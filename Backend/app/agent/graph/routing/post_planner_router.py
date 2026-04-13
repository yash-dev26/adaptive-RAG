from app.schemas.state import GraphState


MAX_REWRITE_ATTEMPTS = 1


def route_after_evaluator(state: GraphState) -> str:
    action = state.eval_action or "generate"
    attempts = state.rewrite_attempts or 0
    docs = state.context or []
    scores = state.scores or []

    if action == "rewrite_single" and attempts < MAX_REWRITE_ATTEMPTS:
        return "rewrite_single"

    if action == "rewrite_multi" and attempts < MAX_REWRITE_ATTEMPTS:
        return "rewrite_multi"

    if action == "llm_fallback" or attempts >= MAX_REWRITE_ATTEMPTS:
        return "llm"

    # same reranking logic as the node
    if len(docs) <= 3:
        return "trim_docs"

    if len(scores) >= 2 and (scores[0] - scores[1] > 0.15):
        return "trim_docs"

    if state.rewrite_type == "multi":
        return "do_rerank"

    return "do_rerank"