from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    PayloadSchemaType,
    SparseVectorParams,
    Modifier,
)
from app.config.qdrantConfig import qdrant_client
from app.config.server import config
from asyncio import sleep

def _ensure_payload_index(collection_name: str, field_name: str, field_schema):
    return qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name=field_name,
        field_schema=field_schema,
    )


async def ensure_collections():
    existing = [c.name for c in (await qdrant_client.get_collections()).collections]

    if config["qdrant_collection_name"] not in existing:
        await qdrant_client.create_collection(
            collection_name=config["qdrant_collection_name"],
            vectors_config={
                "dense": VectorParams(size=384, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                # Modifier.IDF: Qdrant applies IDF weighting server-side using
                # its own corpus stats, since fastembed's BM25 doc vectors only
                # carry saturated term frequency, not IDF.
                "sparse": SparseVectorParams(modifier=Modifier.IDF),
            },
        )
        print(f"[qdrant] Created collection '{config['qdrant_collection_name']}' (hybrid dense+sparse).")
    else:
        print(f"[qdrant] Collection '{config['qdrant_collection_name']}' exists.")

    if config["semantic_cache_collection_name"] not in existing:
        await qdrant_client.create_collection(
            collection_name=config["semantic_cache_collection_name"],
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"[qdrant] Created collection '{config['semantic_cache_collection_name']}'.")
    else:
        print(f"[qdrant] Collection '{config['semantic_cache_collection_name']}' exists.")

    await _ensure_payload_index(config["qdrant_collection_name"], "user_id", PayloadSchemaType.KEYWORD)
    await _ensure_payload_index(config["qdrant_collection_name"], "file_id", PayloadSchemaType.KEYWORD)
    await _ensure_payload_index(config["qdrant_collection_name"], "content_hash", PayloadSchemaType.KEYWORD)
    await _ensure_payload_index(config["semantic_cache_collection_name"], "user_id", PayloadSchemaType.KEYWORD)
    await _ensure_payload_index(config["semantic_cache_collection_name"], "file_id", PayloadSchemaType.KEYWORD)
    await _ensure_payload_index(config["semantic_cache_collection_name"], "expires_at", PayloadSchemaType.INTEGER)
    
async def store_in_qdrant(
    collection_name: str, data: list[dict], file_id: str, user_id: str
):
    points = [
        PointStruct(
            id=item["id"],
            vector=item["vector"],
            payload=item["payload"],
        )
        for item in data
    ]

    # Upsert in smaller batches to avoid HTTP write timeouts for large payloads.
    # Tune CHUNK_SIZE and TIMEOUT_SEC as needed for your environment.
    CHUNK_SIZE = 256
    TIMEOUT_SEC = 90
    MAX_RETRIES = 3

    for i in range(0, len(points), CHUNK_SIZE):
        batch = points[i : i + CHUNK_SIZE]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                await qdrant_client.upsert(collection_name=collection_name, points=batch, timeout=TIMEOUT_SEC)
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait = 2 ** attempt
                    print(f"[qdrant] upsert batch {i}:{i+len(batch)} failed (attempt {attempt}), retrying in {wait}s: {e}")
                    await sleep(wait)
                    continue
                else:
                    print(f"[qdrant] upsert batch {i}:{i+len(batch)} failed (attempt {attempt}), giving up: {e}")
                    raise

    return {
        "status": "success",
        "message": f"Data stored in collection '{collection_name}'.",
        "file_id": file_id,
    }


async def find_existing_file_id_by_content_hash(
    collection_name: str, user_id: str, content_hash: str
) -> str | None:
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    points, _ = await qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="content_hash", match=MatchValue(value=content_hash)),
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        return None

    payload = points[0].payload or {}
    return payload.get("file_id")


async def search_data(collection_name: str, query_vector: list[float], top_k: int = 5):
    """Vector search (used by semantic cache)."""

    response = await qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    )
    return response.points