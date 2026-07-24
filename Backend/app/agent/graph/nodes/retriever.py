from app.schemas.state import GraphState
from app.retrieval.retrieval import retrieve_relevant_documents
from app.retrieval.fusion import reciprocal_rank_fusion
from app.agent.graph.keys import extract_keys
from langchain_core.runnables import RunnableConfig


def retrieve_node(state: GraphState, config: RunnableConfig) -> dict:
    print("[flow] entering retrieve_node")
    base_query = state.rewritten_query or state.query

    queries = state.queries if (state.rewrite_type == "multi" and state.queries) else [base_query]

    openai_api_key, _ = extract_keys(config)

    all_rankings = [
        retrieve_relevant_documents(
            q,
            top_k=5,
            file_id=state.file_id,
            user_id=state.user_id,
            openai_api_key=openai_api_key,
        )
        for q in queries
    ]

    if not all_rankings:
        fused_docs = []
    elif len(all_rankings) > 1:
        # Each ranking here is already a dense+sparse hybrid ranking for its
        # own rewritten-query variant; this fuses across those variants.
        fused_docs = reciprocal_rank_fusion(all_rankings)
    else:
        fused_docs = all_rankings[0]

    final_docs = fused_docs[:8]

    return {
        "context": final_docs,
        "scores": [doc.get("score", 0.0) for doc in final_docs],
        "rrf_scores": [doc.get("rrf_score", 0.0) for doc in final_docs],
        "rewrite_attempts": (state.rewrite_attempts or 0) + 1,
    }