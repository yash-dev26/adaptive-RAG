from app.schemas.state import GraphState

def llm_node(state: GraphState):
    state.response = "llm"
    return state