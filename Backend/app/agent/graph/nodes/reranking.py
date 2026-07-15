from app.schemas.state import GraphState
from app.service.rerankingService import rerank


def reranking_node(state: GraphState):
    print("[flow] entering reranking_node")

    query = state.rewritten_query or state.query
    docs = state.context or []

    if not docs:
        return state

    return {
        "context": rerank(query, docs, top_k=4)
    }