from app.schemas.state import GraphState
from app.config.models import MAX_REWRITE_ATTEMPTS


def route_after_evaluator(state: GraphState) -> str:
    action = state.eval_action or "generate"
    attempts = state.rewrite_attempts or 0
    docs = state.context or []
    scores = state.scores or []

    if action == "rewrite_single" and attempts < MAX_REWRITE_ATTEMPTS:
        return "rewrite_single"

    if action == "rewrite_multi" and attempts < MAX_REWRITE_ATTEMPTS:
        return "rewrite_multi"

    if action == "llm_fallback":
        return "web_search" if attempts >= MAX_REWRITE_ATTEMPTS else "llm"

    # Rewrite attempts exhausted — use web search before falling back to LLM.
    if attempts >= MAX_REWRITE_ATTEMPTS and action in {"rewrite_single", "rewrite_multi"}:
        return "web_search"

    # action == "generate" (or anything else): decide whether we can generate directly,
    # trim the context to the top 4, or pay for reranking.
    if len(docs) <= 4:
        return "generate"

    if len(scores) >= 2 and (scores[0] - scores[1] > 0.15):
        return "trim_docs"

    return "do_rerank"