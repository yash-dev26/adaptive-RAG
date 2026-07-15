from app.schemas.state import GraphState


MAX_REWRITE_ATTEMPTS = 2


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
        return "llm"

    # Rewrite attempts exhausted — fall back to LLM rather than looping
    if attempts >= MAX_REWRITE_ATTEMPTS and action in {"rewrite_single", "rewrite_multi"}:
        return "llm"

    # action == "generate" (or anything else): decide whether we can generate directly,
    # trim the context to the top 4, or pay for reranking.
    if len(docs) <= 4:
        return "generate"

    if len(scores) >= 2 and (scores[0] - scores[1] > 0.15):
        return "trim_docs"

    return "do_rerank"