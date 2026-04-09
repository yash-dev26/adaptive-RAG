from langgraph.graph import StateGraph, END
from app.schemas.state import GraphState
from app.agent.graph.nodes.routerNode import router_node
from app.agent.graph.nodes.llm import llm_node

def build_graph():
    graph = StateGraph(GraphState)
    
    graph.add_node("router", router_node)
    graph.add_node("llm", llm_node)
    
    graph.set_entry_point("router")
    graph.add_edge("router", "llm")
    graph.add_edge("llm", END)
    
    return graph.compile()