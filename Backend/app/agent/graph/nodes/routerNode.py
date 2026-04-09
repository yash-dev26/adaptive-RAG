from app.schemas.state import GraphState

def router_node(state: GraphState):
    state.intent = "router"
    return state