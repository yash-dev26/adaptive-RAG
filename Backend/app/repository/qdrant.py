from app.config.qdrantConfig import qdrant_client
from app.config.server import config

def _ensure_payload_index(collection_name: str, field_name: str, field_schema: str):
    qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name=field_name,
        field_schema=field_schema,
    )


def ensure_collections():
    if config["qdrant_collection_name"] not in [collection.name for collection in qdrant_client.get_collections().collections]:
        qdrant_client.create_collection(
            collection_name=config["qdrant_collection_name"],
            vectors_config={
                "size": 384,
                "distance": "Cosine"
            },
            sparse_vectors_config={
                "text": {}   
            }
        )
    else:
        print(f"Collection '{config['qdrant_collection_name']}' exists.")

    if config["semantic_cache_collection_name"] not in [c.name for c in qdrant_client.get_collections().collections]:
        qdrant_client.create_collection(
            collection_name=config["semantic_cache_collection_name"],
            vectors_config={
                "size": 384,
                "distance": "Cosine"
            }
        )  
    else:
        print(f"Collection '{config['semantic_cache_collection_name']}' exists.")

    _ensure_payload_index(config["qdrant_collection_name"], "user_id", "keyword")
    _ensure_payload_index(config["qdrant_collection_name"], "content_hash", "keyword")
    _ensure_payload_index(config["semantic_cache_collection_name"], "user_id", "keyword")
    _ensure_payload_index(config["semantic_cache_collection_name"], "file_id", "keyword")
    _ensure_payload_index(config["semantic_cache_collection_name"], "expires_at", "integer")

async def store_in_qdrant(collection_name: str, data: list[dict], file_id: str, user_id: str):
    qdrant_client.upsert(
        collection_name=collection_name,
        points=data
    )   
    return {"status": "success", "message": f"Data stored in collection '{collection_name}'.", "file_id": file_id}


def find_existing_file_id_by_content_hash(
    collection_name: str,
    user_id: str,
    content_hash: str,
) -> str | None:
    points, _ = qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter={
            "must": [
                {"key": "user_id", "match": {"value": user_id}},
                {"key": "content_hash", "match": {"value": content_hash}},
            ]
        },
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        print(f"[qdrant] No existing file found for user_id={user_id} with content_hash={content_hash}.")
        return None

    payload = points[0].payload or {}
    print(f"[qdrant] Found existing file for user_id={user_id} with content_hash={content_hash}: file_id={payload.get('file_id')}.")
    return payload.get("file_id")

def search_data(collection_name: str, query_vector: list[float], top_k: int = 5):
    search_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    ).points
    return search_result




