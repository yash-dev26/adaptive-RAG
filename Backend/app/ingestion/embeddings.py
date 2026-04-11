from uuid import uuid4

from app.repository.qdrant import store_in_qdrant
from app.config.server import config
from app.config.openaiConfig import openai_client

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 384


def gen_embeddings(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSION,
        input=text
    )
    return response.data[0].embedding


async def gen_embeddingsAndStoreInQdrant (chunks: str, file_id: str, user_id: str):
    vectors = []

    for chunk in chunks:
        res = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSION,
            input=chunk
        )

        vectors.append({
            "text": chunk,
            "embedding": res.data[0].embedding
        })

    points = [
        {
            "id": str(uuid4()),
            "vector": item["embedding"],
            "payload": {
                "chunk": item["text"],
                "file_id": file_id,
                "user_id": user_id,
                "chunk_index": idx,
            },
        }
        for idx, item in enumerate(vectors)
    ]
    print(f"Generated {len(points)} embeddings for file_id: {file_id}, user_id: {user_id}")

    return await store_in_qdrant(config["qdrant_collection_name"], points, file_id, user_id)