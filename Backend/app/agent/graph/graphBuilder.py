from langgraph.graph import StateGraph, END
from app.schemas.state import GraphState

from app.agent.graph.nodes.planner import planner_node
from app.agent.graph.nodes.retriever import retrieve_node
from app.agent.graph.nodes.rag_llm import generate_node
from app.agent.graph.nodes.multiRewrite import multi_query_rewrite_node
from app.agent.graph.nodes.reranking import reranking_node
from app.agent.graph.nodes.singleRewrite import single_query_rewrite_node
from app.agent.graph.nodes.llm import llm_node

from .routing.planner_router import route_after_planner


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("planner", planner_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("multi-rewrite", multi_query_rewrite_node)
    graph.add_node("single-rewrite", single_query_rewrite_node)
    graph.add_node("rerank", reranking_node)
    graph.add_node("llm", llm_node)

    def planner_postprocess(state: GraphState):
        state["needs_retrieval"] = (
            state["intent"] == "rag"
            and state.get("file_id") is None
        )
        return state
    
    
    graph.add_conditional_edges("planner", route_after_planner, {
        "multi-rewrite": "multi-rewrite",
        "single-rewrite": "single-rewrite",
        "retrieve": "retrieve",
        "generate": "generate",
        "llm": "llm"
    })


    graph.add_edge("llm", END)
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("single-rewrite", "retrieve")
    graph.add_edge("single-rewrite", "llm")
    graph.add_edge("multi-rewrite", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    graph.set_entry_point("planner")

    
    return graph.compile()