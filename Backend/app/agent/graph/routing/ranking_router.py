from app.schemas.state import GraphState

def ranking_decision_node(state: GraphState):
    rewrite_type = state.rewrite_type

    if rewrite_type != "multi" or num_retrieved_docs(state) > 2 or avg_score > 0.80:
        return "no_reranking_needed"
    
    return "reranking_needed"