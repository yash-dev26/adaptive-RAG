from app.schemas.state import GraphState
from app.retrieval.retrieval import retrieve_relevant_documents
from app.retrieval.fusion import reciprocal_rank_fusion
from app.agent.graph.keys import extract_keys
from langchain_core.runnables import RunnableConfig
import asyncio

# Aggregate queries ("list every X", "compare A and B") need broader coverage
# than a single-fact QA query, so we increase the top-k and context-limit for those queries.
DEFAULT_TOP_K = 5
AGGREGATE_TOP_K = 10
DEFAULT_CONTEXT_LIMIT = 8
AGGREGATE_CONTEXT_LIMIT = 14


async def retrieve_node(state: GraphState, config: RunnableConfig) -> dict:
    print("[flow] entering retrieve_node")
    base_query = state.rewritten_query or state.query

    queries = state.queries if (state.rewrite_type == "multi" and state.queries) else [base_query]

    is_aggregate = state.task_type == "aggregate"
    top_k = AGGREGATE_TOP_K if is_aggregate else DEFAULT_TOP_K
    context_limit = AGGREGATE_CONTEXT_LIMIT if is_aggregate else DEFAULT_CONTEXT_LIMIT

    openai_api_key, _ = extract_keys(config)

    tasks = [
        retrieve_relevant_documents(
            q,
            top_k=top_k,
            file_id=state.file_id,
            user_id=state.user_id,
            openai_api_key=openai_api_key,
        )
        for q in queries
    ]

    all_rankings = await asyncio.gather(*tasks)

    if not all_rankings:
        fused_docs = []
    elif len(all_rankings) > 1:
        # Each ranking here is already a dense+sparse hybrid ranking for its
        # own rewritten-query variant; this fuses across those variants.
        fused_docs = reciprocal_rank_fusion(all_rankings)
    else:
        fused_docs = all_rankings[0]

    final_docs = fused_docs[:context_limit]

    return {
        "context": final_docs,
        "scores": [doc.get("score", 0.0) for doc in final_docs],
        "rrf_scores": [doc.get("rrf_score", 0.0) for doc in final_docs],
        "rewrite_attempts": (state.rewrite_attempts or 0) + 1,
    }