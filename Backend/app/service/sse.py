"""
SSE event formatting + node-name/detail lookups for streaming chat responses.
Kept separate from chatService so the wire-format concerns don't get tangled
up with caching and graph-invocation logic.
"""

import json

STREAMABLE_NODES = {
    "pre_planner",
    "multi_rewrite",
    "single_rewrite",
    "retrieve",
    "evaluator",
    "trim_docs",
    "rerank",
    "generate",
    "llm",
}

NODE_DETAILS = {
    "pre_planner": "Planning whether retrieval is needed.",
    "multi_rewrite": "Expanding the query into multiple retrieval variants.",
    "single_rewrite": "Rewriting the query for a targeted retrieval pass.",
    "retrieve": "Searching the vector store for relevant context.",
    "evaluator": "Evaluating retrieved context quality.",
    "trim_docs": "Trimming context to fit the prompt window.",
    "rerank": "Reranking retrieved documents.",
    "generate": "Generating the final answer from context.",
    "llm": "Generating the final answer without retrieval.",
}


def format_sse(event_type: str, payload: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, default=str)}\n\n"


def node_detail(node_name: str, status: str) -> str:
    # `status` isn't used to vary the message today, but kept as a parameter
    # in case running/done text needs to diverge later.
    return NODE_DETAILS.get(node_name, f"Executing {node_name}.")