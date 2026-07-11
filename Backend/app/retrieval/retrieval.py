from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
)

from app.ingestion.embeddings import gen_embeddings
from app.config.qdrantConfig import qdrant_client
from app.config.server import config


def retrieve_relevant_documents(
    query: str,
    openai_api_key: str,
    top_k: int = 5,
    file_id: str | None = None,
    user_id: str | None = None,
) -> list[dict]:
    """
    Vector search using query embedding and optional payload filters.
    Filters by file_id and user_id when provided.
    """
    query_embedding: list[float] = gen_embeddings(query, openai_api_key)

    # ── build optional payload filter ───────────────────────────────────────
    filter_conditions = []
    if file_id:
        filter_conditions.append(FieldCondition(key="file_id", match=MatchValue(value=file_id)))
    if user_id:
        filter_conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
    payload_filter = Filter(must=filter_conditions) if filter_conditions else None

    results = qdrant_client.query_points(
        collection_name=config["qdrant_collection_name"],
        query=query_embedding,
        query_filter=payload_filter,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {
            "text": (hit.payload or {}).get("text") or (hit.payload or {}).get("chunk", ""),
            "score": hit.score,
        }
        for hit in results
    ]
