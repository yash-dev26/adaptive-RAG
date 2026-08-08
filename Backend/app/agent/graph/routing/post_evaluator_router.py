from app.schemas.state import GraphState
from app.config.models import MAX_REWRITE_ATTEMPTS


def route_after_evaluator(state: GraphState) -> str:
    action = state.eval_action or "generate"
    attempts = state.rewrite_attempts or 0
    docs = state.context or []
    scores = state.scores or []
    tavily_available = bool(state.tavily_configured)

    if action == "rewrite_single" and attempts < MAX_REWRITE_ATTEMPTS:
        return "rewrite_single"

    if action == "rewrite_multi" and attempts < MAX_REWRITE_ATTEMPTS:
        return "rewrite_multi"

    # Nothing usable is going to come out of retrieval — either the
    # evaluator said so directly (llm_fallback), or we've burned through the
    # rewrite budget. Try Tavily first, but only if the caller actually
    # supplied a key, Without a key, routing
    # straight to "llm" skips a request we already know will no-op and goes
    # straight to a disclaimed general-knowledge answer instead.
    exhausted_rewrites = attempts >= MAX_REWRITE_ATTEMPTS and action in {"rewrite_single", "rewrite_multi"}
    if action == "llm_fallback" or exhausted_rewrites:
        return "web_search" if tavily_available else "llm"

    # action == "generate" (or anything else): decide whether we can generate directly,
    # trim the context to the top 4, or pay for reranking.
    if len(docs) <= 4:
        return "generate"

    if len(scores) >= 2 and (scores[0] - scores[1] > 0.15):
        return "trim_docs"

    return "do_rerank"