"""
Routes after the pre-retrieval planner.
Concerns: intent (rag vs llm), task_type (qa/summarize/aggregate), and
rewrite strategy. Doc-quality decisions live in evaluator_router, and only
apply to the "qa" task_type — summarize bypasses retrieval-quality routing
entirely since it isn't answerable via top-k similarity in the first place.
"""

from app.schemas.state import GraphState


def route_after_pre_planner(state: GraphState) -> str:
    if state.intent == "llm" and state.tavily_configured == True:
        return "web_search"
    
    if state.intent == "llm" and state.tavily_configured == False:
        return "llm"

    # summarize needs the whole document, not a top-k similarity search
    # route straight to the dedicated map-reduce summarizer
    if state.task_type == "summarize":
        return "summarize"

    if state.rewrite_type == "single":
        return "single_rewrite"
    if state.rewrite_type == "multi":
        return "multi_rewrite"

    return "retrieve"