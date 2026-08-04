from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
)

from app.ingestion.embeddings import gen_embeddings
from app.ingestion.sparse_embeddings import gen_sparse_query_embedding
from app.config.qdrantConfig import qdrant_client
from app.config.server import config
from app.retrieval.fusion import reciprocal_rank_fusion


def _build_filter(file_id: str | None, user_id: str | None):
    filter_conditions = []
    if file_id:
        filter_conditions.append(FieldCondition(key="file_id", match=MatchValue(value=file_id)))
    if user_id:
        filter_conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
    return Filter(must=filter_conditions) if filter_conditions else None


import asyncio

async def retrieve_relevant_documents(
    query: str,
    openai_api_key: str,
    top_k: int = 5,
    file_id: str | None = None,
    user_id: str |None = None,
) -> list[dict]:
    """
    Hybrid search: dense + sparse fused with RRF.
    """

    payload_filter = _build_filter(file_id, user_id)

    # Start embedding request immediately
    dense_embedding_task = asyncio.create_task(
        gen_embeddings(query, openai_api_key)
    )

    # CPU work overlaps with OpenAI request
    sparse_embedding = gen_sparse_query_embedding(query)

    # Wait for OpenAI
    dense_embedding = await dense_embedding_task

    # Run both vector searches concurrently
    dense_result, sparse_result = await asyncio.gather(
        qdrant_client.query_points(
            collection_name=config["qdrant_collection_name"],
            query=dense_embedding,
            using="dense",
            query_filter=payload_filter,
            limit=top_k,
            with_payload=True,
        ),
        qdrant_client.query_points(
            collection_name=config["qdrant_collection_name"],
            query=sparse_embedding,
            using="sparse",
            query_filter=payload_filter,
            limit=top_k,
            with_payload=True,
        ),
    )

    dense_hits = dense_result.points
    sparse_hits = sparse_result.points

    dense_ranking = [
        {
            "text": (hit.payload or {}).get("text")
            or (hit.payload or {}).get("chunk", ""),
            "score": hit.score,
            "source": "dense",
        }
        for hit in dense_hits
    ]

    sparse_ranking = [
        {
            "text": (hit.payload or {}).get("text")
            or (hit.payload or {}).get("chunk", ""),
            "score": hit.score,
            "source": "sparse",
        }
        for hit in sparse_hits
    ]

    if not sparse_ranking:
        return dense_ranking
    if not dense_ranking:
        return sparse_ranking

    return reciprocal_rank_fusion([dense_ranking, sparse_ranking])[:top_k]