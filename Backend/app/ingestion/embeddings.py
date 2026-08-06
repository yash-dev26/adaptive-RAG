from uuid import uuid4
from typing import List

from openai import AuthenticationError
import json

from fastapi import HTTPException
import os
import asyncio

from app.config.redis import redis_client
from app.repository.qdrant import store_in_qdrant
from app.config.server import config
from app.config.providers import get_openai_client
from app.cache.embeddings_cache import _embedding_cache_key
from app.ingestion.sparse_embeddings import gen_sparse_document_embeddings
from app.utils.retry import external_api_retry

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 384
# Smaller batches to reduce per-request latency and improve parallelism
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
# Concurrency limit for processing multiple batches simultaneously
CONCURRENCY_LIMIT = int(os.getenv("EMBED_CONCURRENCY", "4"))
EMBED_TIMEOUT_SEC = float(os.getenv("OPENAI_EMBED_TIMEOUT_SEC", "45"))

@external_api_retry()
async def gen_embeddings(text: str, openai_api_key: str) -> List[float]:
    cache_key = _embedding_cache_key(text)
    cached_embedding = await redis_client.get(cache_key)
    if cached_embedding:
        return json.loads(cached_embedding)

    client = get_openai_client(openai_api_key)
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSION,
        input=text,
        timeout=EMBED_TIMEOUT_SEC,
    )
    embedding = response.data[0].embedding
    await redis_client.set(cache_key, json.dumps(embedding), ex=60 * 60 * 24)
    return embedding

@external_api_retry()
async def _embed_batch(texts: List[str], openai_api_key: str) -> List[List[float]]:
    client = get_openai_client(openai_api_key)
    try :
        response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSION,
        input=texts,
        timeout=EMBED_TIMEOUT_SEC,
        )
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key")
    
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
    

async def gen_embeddingsAndStoreInQdrant(
    chunks: List[str],
    file_id: str,
    user_id: str,
    content_hash: str,
    openai_api_key: str,
) -> dict:

    print(f"[embeddings] Generating hybrid embeddings for {len(chunks)} chunks (batch_size={BATCH_SIZE}, concurrency={CONCURRENCY_LIMIT})...")

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def _process_batch(batch_start: int, batch_chunks: List[dict]):
        batch_texts = [chunk["text"] for chunk in batch_chunks]
        batch_end = batch_start + len(batch_chunks)
        print(f"[embeddings] Processing batch {batch_start}:{batch_end} (size={len(batch_chunks)})")

        try:
            dense_embeddings, sparse_embeddings = await asyncio.gather(
                _embed_batch(batch_texts, openai_api_key),
                asyncio.to_thread(gen_sparse_document_embeddings, batch_texts),
            )
        except Exception as e:
            print(f"[embeddings] Batch failed at {batch_start}:{batch_end} -> {type(e).__name__}: {e}")
            raise

        points = []
        for offset, (chunk_data, dense_vec, sparse_vec) in enumerate(
            zip(batch_chunks, dense_embeddings, sparse_embeddings)
        ):
            idx = batch_start + offset
            points.append(
                {
                    "id": str(uuid4()),
                    "vector": {"dense": dense_vec, "sparse": sparse_vec},
                    "payload": {
                        "chunk": chunk_data["text"],
                        "text": chunk_data["text"],
                        "page": chunk_data["page"],
                        "file_id": file_id,
                        "user_id": user_id,
                        "chunk_index": idx,
                        "content_hash": content_hash,
                    },
                }
            )

        # Store this batch's points in Qdrant
        await store_in_qdrant(config["qdrant_collection_name"], points, file_id, user_id)

    # Create tasks for all batches and run up to CONCURRENCY_LIMIT concurrently
    tasks = []
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[batch_start : batch_start + BATCH_SIZE]

        async def sem_task(bs=batch_start, bc=batch_chunks):
            async with semaphore:
                await _process_batch(bs, bc)

        tasks.append(asyncio.create_task(sem_task()))

    # Await completion of all batches
    await asyncio.gather(*tasks)

    return {
        "status": "success",
        "message": f"Stored {len(chunks)} chunks in collection '{config['qdrant_collection_name']}'.",
        "file_id": file_id,
    }