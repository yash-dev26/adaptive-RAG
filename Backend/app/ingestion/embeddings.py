from http.client import HTTPException
from uuid import uuid4
from typing import List

from openai import APIStatusError, AuthenticationError, RateLimitError, APIConnectionError
import json

from fastapi import HTTPException
import os

from app.config.redis import redis_client
from app.repository.qdrant import store_in_qdrant
from app.config.server import config
from app.config.providers import get_openai_client
from app.cache.embeddings_cache import _embedding_cache_key
from app.ingestion.sparse_embeddings import gen_sparse_document_embeddings
from app.utils.retry import external_api_retry

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 384
BATCH_SIZE = 512
EMBED_TIMEOUT_SEC = float(os.getenv("OPENAI_EMBED_TIMEOUT_SEC", "45"))

@external_api_retry()
def gen_embeddings(text: str, openai_api_key: str) -> List[float]:
    cache_key = _embedding_cache_key(text)
    cached_embedding = redis_client.get(cache_key)
    if cached_embedding:
        return json.loads(cached_embedding)

    client = get_openai_client(openai_api_key)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSION,
        input=text,
        timeout=EMBED_TIMEOUT_SEC,
    )
    embedding = response.data[0].embedding
    redis_client.set(cache_key, json.dumps(embedding), ex=60 * 60 * 24)
    return embedding

@external_api_retry()
def _embed_batch(texts: List[str], openai_api_key: str) -> List[List[float]]:
    client = get_openai_client(openai_api_key)
    try :
        response = client.embeddings.create(
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

    all_dense_embeddings: List[List[float]] = []
    all_sparse_embeddings = []
    print(f"[embeddings] Generating hybrid embeddings for {len(chunks)} chunks...")

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        batch_texts = [chunk["text"] for chunk in batch]
        batch_end = batch_start + len(batch)

        print(f"[embeddings] Dense embedding batch {batch_start}:{batch_end} (size={len(batch)})")
        try:
            dense_embeddings = _embed_batch(batch_texts, openai_api_key)
        except Exception as e:
            print(f"[embeddings] Dense embedding batch failed at {batch_start}:{batch_end} -> {type(e).__name__}: {e}")
            raise

        print(f"[embeddings] Sparse (BM25) embedding batch {batch_start}:{batch_end}")
        sparse_embeddings = gen_sparse_document_embeddings(batch_texts)

        all_dense_embeddings.extend(dense_embeddings)
        all_sparse_embeddings.extend(sparse_embeddings)

    points = [
        {
            "id": str(uuid4()),
            "vector": {
                "dense": dense_vec,
                "sparse": sparse_vec,
            },
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
        for idx, (chunk_data, dense_vec, sparse_vec) in enumerate(
            zip(chunks, all_dense_embeddings, all_sparse_embeddings)
        )
    ]

    print(
        f"[embeddings] Generated {len(points)} hybrid embeddings for "
        f"file_id={file_id}, user_id={user_id} "
        f"in {len(range(0, len(chunks), BATCH_SIZE))} batch(es)."
    )

    print(f"[embeddings] Storing embeddings in Qdrant for file_id={file_id}, user_id={user_id}...")
    return await store_in_qdrant(
        config["qdrant_collection_name"], points, file_id, user_id
    )