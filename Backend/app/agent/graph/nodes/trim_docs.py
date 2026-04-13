from app.schemas.state import GraphState


def trim_docs_node(state: GraphState):
    print("[flow] entering trim_docs_node")
    docs = state.context or []
    qdrant_scores = state.scores or []
    rrf_scores = state.rrf_scores or []

    top_docs = docs[:4]  # just take top 4 as-is as reranking will be skipped in this path

    return {
        "context": top_docs,
        "scores": qdrant_scores[:4],
        "rrf_scores": rrf_scores[:4]
    }