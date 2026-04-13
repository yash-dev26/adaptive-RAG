from uuid import uuid4
from typing import List

from app.repository.qdrant import store_in_qdrant
from app.config.server import config
from app.config.openaiConfig import openai_client

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 384
BATCH_SIZE = 512


def gen_embeddings(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSION,
        input=text,
    )
    return response.data[0].embedding

def _embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts in a single API call.
    Returns vectors in the same order as `texts`.
    """
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
) -> dict:
    
    all_embeddings: List[List[float]] = []

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        embeddings = _embed_batch(batch)
        all_embeddings.extend(embeddings)

    points = [
        {
            "id": str(uuid4()),
            "vector": embedding,
            "payload": {
                "chunk": chunk,
                "text": chunk,         
                "file_id": file_id,
                "user_id": user_id,
                "chunk_index": idx,
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

