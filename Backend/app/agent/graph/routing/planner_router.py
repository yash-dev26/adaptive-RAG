def route_after_planner(state):
    
    if state.rewrite_type == "single":
        return "single-rewrite"
    if state.rewrite_type == "multi":
        return "multi-rewrite"
    if state.intent == "rag":
        return "retrieve"
    return "generate"