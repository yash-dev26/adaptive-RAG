"""
Routes after the pre-retrieval planner.
Only concerns: intent (rag vs llm) and rewrite strategy.
Doc-quality decisions now live in evaluator_router.
"""

from app.schemas.state import GraphState


def route_after_pre_planner(state: GraphState) -> str:
    if state.intent == "llm":
        return "llm"

    if state.rewrite_type == "single":
        return "single_rewrite"
    if state.rewrite_type == "multi":
        return "multi_rewrite"

    return "retrieve"