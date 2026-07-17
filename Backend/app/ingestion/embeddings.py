from uuid import uuid4
from typing import List
import json
import os

from app.config.redis import redis_client
from app.repository.qdrant import store_in_qdrant
from app.config.server import config
from app.config.providers import get_openai_client
from app.cache.embeddings_cache import _embedding_cache_key

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 384
BATCH_SIZE = 512
EMBED_TIMEOUT_SEC = float(os.getenv("OPENAI_EMBED_TIMEOUT_SEC", "45"))


def gen_embeddings(text: str, openai_api_key: str) -> List[float]:
    # Check if the embedding is already in the cache
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
    # Store the embedding in the cache
    redis_client.set(cache_key, json.dumps(embedding), ex=60 * 60 * 24)  # Cache for 24 hours
    return embedding

def _embed_batch(texts: List[str], openai_api_key: str) -> List[List[float]]:
    client = get_openai_client(openai_api_key)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSION,
        input=texts,
        timeout=EMBED_TIMEOUT_SEC,
    )
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


async def gen_embeddingsAndStoreInQdrant(
    chunks: List[str],
    file_id: str,
    user_id: str,
    content_hash: str,
    openai_api_key: str,
) -> dict:

    all_embeddings: List[List[float]] = []
    print(f"[embeddings] Generating embeddings for {len(chunks)} chunks...")

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        batch_texts = [chunk["text"] for chunk in batch]

        batch_end = batch_start + len(batch)

        print(f"[embeddings] Dense embedding batch {batch_start}:{batch_end} (size={len(batch)})")

        try:
            embeddings = _embed_batch(batch_texts, openai_api_key)
        except Exception as e:
            print(f"[embeddings] Dense embedding batch failed at {batch_start}:{batch_end} -> {type(e).__name__}: {e}")
            raise

        all_embeddings.extend(embeddings)

    points = [
        {
            "id": str(uuid4()),
            "vector": embedding,
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
        for idx, (chunk_data, embedding) in enumerate(zip(chunks, all_embeddings))
    ]

    print(
        f"[embeddings] Generated {len(points)} embeddings for "
        f"file_id={file_id}, user_id={user_id} "
        f"in {len(range(0, len(chunks), BATCH_SIZE))} batch(es)."
    )

    print (f"[embeddings] Storing embeddings in Qdrant for file_id={file_id}, user_id={user_id}...")
    return await store_in_qdrant(
        config["qdrant_collection_name"], points, file_id, user_id
    )
