from langgraph.graph import StateGraph, END
from app.schemas.state import GraphState

from app.agent.graph.nodes.planner import (
    pre_retrieval_planner_node,
    post_retrieval_evaluator_node,
)
from app.agent.graph.nodes.retriever import retrieve_node
from app.agent.graph.nodes.rag_llm import generate_node
from app.agent.graph.nodes.multiRewrite import multi_query_rewrite_node
from app.agent.graph.nodes.singleRewrite import single_query_rewrite_node
from app.agent.graph.nodes.trim_docs import trim_docs_node
from app.agent.graph.nodes.reranking import reranking_node
from app.agent.graph.nodes.llm import llm_node

from app.agent.graph.routing.pre_planner_routes import route_after_pre_planner
from app.agent.graph.routing.post_planner_router import route_after_evaluator
from app.agent.graph.routing.ranking_router import ranking_decision_node
from app.agent.graph.routing.postRewrite_router import route_after_single_rewrite


def build_graph():
    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("pre_planner", pre_retrieval_planner_node)
    graph.add_node("multi_rewrite", multi_query_rewrite_node)
    graph.add_node("single_rewrite", single_query_rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluator", post_retrieval_evaluator_node)
    graph.add_node("trim_docs", trim_docs_node)
    graph.add_node("rerank", reranking_node)
    graph.add_node("generate", generate_node)
    graph.add_node("llm", llm_node)

    # Entry point
    graph.set_entry_point("pre_planner")

    # Edges
    graph.add_conditional_edges(
        "pre_planner",
        route_after_pre_planner,
        {
            "multi_rewrite": "multi_rewrite",
            "single_rewrite": "single_rewrite",
            "retrieve": "retrieve",
            "llm": "llm",
        },
    )

    graph.add_edge("multi_rewrite", "retrieve")

    graph.add_conditional_edges(
        "single_rewrite",
        route_after_single_rewrite,
        {
            "retrieve": "retrieve",
            "llm": "llm",
        },
    )

    graph.add_edge("retrieve", "evaluator")

    graph.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {
            "do_rerank":       "rerank",
            "trim_docs":       "trim_docs",
            "rewrite_single":  "single_rewrite",
            "rewrite_multi":   "multi_rewrite",
            "llm":             "llm",
        },
    )

    graph.add_edge("trim_docs", "generate")
    graph.add_edge("rerank", "generate")

    graph.add_edge("generate", END)
    graph.add_edge("llm", END)

    return graph.compile()