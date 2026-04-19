from uuid import uuid4
from typing import List
import json

from app.config.redis import redis_client
from app.repository.qdrant import store_in_qdrant
from app.config.server import config
from app.config.openaiConfig import openai_client
from app.cache.embeddings_cache import _embedding_cache_key
from fastembed import SparseTextEmbedding

sparse_model = SparseTextEmbedding()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 384
BATCH_SIZE = 512


def gen_embeddings(text: str) -> List[float]:
    # Check if the embedding is already in the cache
    cache_key = _embedding_cache_key(text)
    cached_embedding = redis_client.get(cache_key)
    if cached_embedding:
        return json.loads(cached_embedding)

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSION,
        input=text,
    )
    embedding =  response.data[0].embedding
    # Store the embedding in the cache
    redis_client.set(cache_key, json.dumps(embedding))
    return embedding

def _embed_batch(texts: List[str]) -> List[List[float]]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSION,
        input=texts,
    )
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


async def gen_embeddingsAndStoreInQdrant(
    chunks: List[str],
    file_id: str,
    user_id: str,
    content_hash: str,
) -> dict:
    
    all_embeddings: List[List[float]] = []

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        embeddings = _embed_batch(batch)
        all_embeddings.extend(embeddings)

    sparse_vectors = list(sparse_model.embed(chunks))

    points = [
        {
            "id": str(uuid4()),
            "vector": embedding,
            "sparse_vector": {
                "text": sparse_vectors[idx]
            },
            "payload": {
                "chunk": chunk,
                "text": chunk,
                "file_id": file_id,
                "user_id": user_id,
                "chunk_index": idx,
                "content_hash": content_hash,
            },
        }
        for idx, (chunk, embedding) in enumerate(zip(chunks, all_embeddings))
    ]

    print(
        f"[embeddings] Generated {len(points)} embeddings for "
        f"file_id={file_id}, user_id={user_id} "
        f"in {len(range(0, len(chunks), BATCH_SIZE))} batch(es)."
    )

    return await store_in_qdrant(
        config["qdrant_collection_name"], points, file_id, user_id
    )

