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


def retrieve_relevant_documents(
    query: str,
    openai_api_key: str,
    top_k: int = 5,
    file_id: str | None = None,
    user_id: str | None = None,
) -> list[dict]:
    """
    Hybrid search: dense (OpenAI embeddings, semantic) + sparse (BM25, lexical),
    fused client-side with RRF. This runs unconditionally for every query —
    the sparse fan-out is a local computation, not an API call, so there's no
    cost/latency tradeoff worth gating on, unlike the planner/evaluator/
    reranker decisions elsewhere in the graph.
    """
    payload_filter = _build_filter(file_id, user_id)

    dense_embedding = gen_embeddings(query, openai_api_key)
    sparse_embedding = gen_sparse_query_embedding(query)

    dense_hits = qdrant_client.query_points(
        collection_name=config["qdrant_collection_name"],
        query=dense_embedding,
        using="dense",
        query_filter=payload_filter,
        limit=top_k,
        with_payload=True,
    ).points

    sparse_hits = qdrant_client.query_points(
        collection_name=config["qdrant_collection_name"],
        query=sparse_embedding,
        using="sparse",
        query_filter=payload_filter,
        limit=top_k,
        with_payload=True,
    ).points

    dense_ranking = [
        {
            "text": (hit.payload or {}).get("text") or (hit.payload or {}).get("chunk", ""),
            "score": hit.score,
            "source": "dense",
        }
        for hit in dense_hits
    ]
    sparse_ranking = [
        {
            "text": (hit.payload or {}).get("text") or (hit.payload or {}).get("chunk", ""),
            "score": hit.score,
            "source": "sparse",
        }
        for hit in sparse_hits
    ]

    if not sparse_ranking:
        return dense_ranking
    if not dense_ranking:
        return sparse_ranking

    fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking])
    return fused[:top_k]