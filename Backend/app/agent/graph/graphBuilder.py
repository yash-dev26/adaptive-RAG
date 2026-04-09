from langgraph.graph import StateGraph, END
from app.agent.graph.nodes.ragNode import rag_node
from app.schemas.state import GraphState
from app.agent.graph.nodes.routerNode import router_node
from app.agent.graph.nodes.llm import llm_node

def build_graph():
    graph = StateGraph(GraphState)
    
    graph.add_node("router", router_node)
    graph.add_node("llm", llm_node)
    graph.add_node("rag", rag_node)

    def route(state: GraphState):
        if isinstance(state, GraphState):
            return state.intent
        return state.get("intent")
    
    graph.add_conditional_edges("router", route, {
    "rag": "rag",
    "llm": "llm",
    })
    
    graph.set_entry_point("router")
    graph.add_edge("llm", END)
    graph.add_edge("rag", END)
    
    return graph.compile()