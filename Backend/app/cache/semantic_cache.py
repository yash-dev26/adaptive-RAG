import json
import time
from uuid import uuid4

from qdrant_client.models import IsEmptyCondition, PayloadField

from app.ingestion.embeddings import gen_embeddings
from app.repository.qdrant import qdrant_client
from app.config.server import config


SIMILARITY_THRESHOLD = 0.72  # tune later


def get_semantic_cached_response(query: str, user_id: str, file_id: str | None, openai_api_key: str):
    query_embedding = gen_embeddings(query, openai_api_key)
    current_time = int(time.time())

    must_conditions = [
        {"key": "user_id", "match": {"value": user_id}},
        {
            "key": "expires_at",
            "range": {"gte": current_time}
        },
    ]

    if file_id is not None:
        must_conditions.append({"key": "file_id", "match": {"value": file_id}})
    else:
        
        must_conditions.append(
            IsEmptyCondition(is_empty=PayloadField(key="file_id"))
        )

    results = qdrant_client.query_points(
        collection_name=config["semantic_cache_collection_name"],
        query=query_embedding,
        query_filter={
            "must": must_conditions
        },
        limit=1
    ).points

    if not results:
        return None

    top = results[0]

    print(f"[semantic debug] sim={top.score:.3f} | cached='{top.payload.get('query')}'")

    if top.score > SIMILARITY_THRESHOLD:
        print(f"[semantic cache] HIT ({top.score:.3f})")
        return top.payload.get("response")

    print(f"[semantic cache] MISS ({top.score:.3f})")
    return None


def set_semantic_cache(query, response, user_id, file_id, openai_api_key: str):
    embedding = gen_embeddings(query, openai_api_key)
    now = int(time.time())
    ttl = 36000

    qdrant_client.upsert(
        collection_name=config["semantic_cache_collection_name"],
        points=[
            {
                "id": str(uuid4()),
                "vector": embedding,
                "payload": {
                    "query": query,
                    "response": response,
                    "user_id": user_id,
                    "file_id": file_id,
                    "created_at": now,
                    "expires_at": now + ttl
                }
            }
        ]
    )