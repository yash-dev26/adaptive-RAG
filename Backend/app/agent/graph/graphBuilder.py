from langgraph.graph import StateGraph, END
from app.schemas.state import GraphState

from app.agent.graph.nodes.planner import pre_retrieval_planner_node
from app.agent.graph.nodes.evaluator import post_retrieval_evaluator_node
from app.agent.graph.nodes.retriever import retrieve_node
from app.agent.graph.nodes.generate import generate_node
from app.agent.graph.nodes.multiRewrite import multi_query_rewrite_node
from app.agent.graph.nodes.singleRewrite import single_query_rewrite_node
from app.agent.graph.nodes.trim_docs import trim_docs_node
from app.agent.graph.nodes.reranking import reranking_node
from app.agent.graph.nodes.summarize import summarize_node
from app.agent.graph.nodes.webSearch import web_search_node

from app.agent.graph.routing.pre_planner_routes import route_after_pre_planner
from app.agent.graph.routing.post_evaluator_router import route_after_evaluator
from app.agent.graph.routing.postRewrite_router import route_after_single_rewrite


def build_graph(checkpointer=None):
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
    graph.add_node("summarize", summarize_node)
    graph.add_node("web_search", web_search_node)

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
            "summarize": "summarize",
            # "llm" is a route *name* meaning "answer without retrieval"
            # it's handled by the same centralized generate node as every
            # other terminal path
            "llm": "generate",
            "web_search": "web_search",
        },
    )

    graph.add_edge("multi_rewrite", "retrieve")

    graph.add_conditional_edges(
        "single_rewrite",
        route_after_single_rewrite,
        {
            "retrieve": "retrieve",
            "llm": "generate",
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
            "llm":             "generate",
            "web_search":      "web_search",
            "generate":        "generate",
        },
    )

    graph.add_edge("trim_docs", "generate")
    graph.add_edge("rerank", "generate")
    graph.add_edge("web_search", "generate")
    graph.add_edge("summarize", "generate")

    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer)