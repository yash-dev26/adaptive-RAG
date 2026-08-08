from app.schemas.state import GraphState
from app.service.rerankingService import rerank


async def reranking_node(state: GraphState):
    print("[flow] entering reranking_node")

    query = state.rewritten_query or state.query
    docs = state.context or []

    if not docs:
        return state

    reranked = await rerank(query, docs, top_k=4)

    return {
        "context": reranked,
        "scores": [doc.get("score", 0.0) for doc in reranked],
        "rrf_scores": [doc.get("rrf_score", 0.0) for doc in reranked],
    }