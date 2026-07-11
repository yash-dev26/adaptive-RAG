import json
from app.schemas import response
from app.schemas.request import ChatRequest
from app.schemas.state import GraphState
from app.cache.response_cache import get_cached_response, set_cached_response
from app.cache.semantic_cache import get_semantic_cached_response, set_semantic_cache

from uuid import uuid4


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


def _format_sse(event_type: str, payload: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, default=str)}\n\n"


def _extract_event_output(event: dict):
    data = event.get("data") or {}

    if isinstance(data, dict):
        return data.get("output") or data.get("chunk") or data.get("result") or data

    return data


def _extract_node_name(event: dict) -> str | None:
    metadata = event.get("metadata") or {}
    return metadata.get("langgraph_node") or event.get("name")


def _node_detail(node_name: str, status: str) -> str:
    base_detail = NODE_DETAILS.get(node_name, f"Executing {node_name}.")

    if status == "done":
        return base_detail

    return base_detail

def _resolve_thread_id(request: ChatRequest, session_id: str) -> str:
    if request.thread_id:
        return request.thread_id

    if request.file_id:
        return f"{session_id}:{request.file_id}"

    return f"{session_id}:{uuid4()}"


def _build_run_config(thread_id: str, api_keys: dict) -> dict:
    """
    BYOK keys go into `configurable`, NOT into GraphState — GraphState is
    what MongoDBSaver checkpoints on every turn (that's how conversation
    memory works), and we don't want a user's API key written to Mongo.
    `configurable` is per-invocation only; nodes read it via extract_keys().
    """
    return {
        "configurable": {
            "thread_id": thread_id,
            "openai_api_key": api_keys["openai_api_key"],
            "groq_api_key": api_keys.get("groq_api_key"),
        }
    }


async def process_chat(request: ChatRequest, graph, session_id: str, api_keys: dict):
    thread_id = _resolve_thread_id(request, session_id)
    openai_api_key = api_keys["openai_api_key"]

    semantic_hit = get_semantic_cached_response(
        request.query,
        session_id,
        request.file_id,
        openai_api_key,
    )

    if semantic_hit:
        return {
            "response": semantic_hit,
            "thread_id": thread_id,
            "cached": "semantic"
        }

    cached_response = get_cached_response(
        user_id=session_id,
        file_id=request.file_id if request.file_id else None,
        query=request.query
    )

    if cached_response:
        print("[cache] response cache HIT")
        return {
            "response": cached_response,
            "thread_id": thread_id,
            "cached": True,
        }

    print("[cache] response cache MISS")

    state = GraphState(
        user_id=session_id,
        query=request.query,
        file_id=request.file_id if request.file_id else None
    )

    invoke_config = _build_run_config(thread_id, api_keys)

    result = graph.invoke(state, config=invoke_config) if graph else None

    rewritten_query = None
    queries = None

    if isinstance(result, dict):
        rewritten_query = result.get("rewritten_query")
        queries = result.get("queries")
    else:
        rewritten_query = getattr(result, "rewritten_query", None)
        queries = getattr(result, "queries", None)

    if isinstance(result, dict):
        response = result.get("response", "I could not generate a response right now.")
        confidence = result.get("confidence", None)
    else:
        response = getattr(result, "response", "I could not generate a response right now.")
        confidence = getattr(result, "confidence", None)

    if confidence is None or confidence > 0.6:
        # Always store original query (fallback baseline)
        set_semantic_cache(
            request.query,
            response,
            session_id,
            request.file_id,
            openai_api_key,
        )

        # Store rewritten query (single)
        if rewritten_query:
            set_semantic_cache(
                rewritten_query,
                response,
                session_id,
                request.file_id,
                openai_api_key,
            )

        # Store multi queries
        if queries:
            for q in queries:
                set_semantic_cache(
                    q,
                    response,
                    session_id,
                    request.file_id,
                    openai_api_key,
                )


        set_cached_response(
            user_id=session_id,
            file_id=request.file_id,
            query=request.query,
            response=response,
        )
        print(f"[cache] response cached (confidence={confidence})")
    else:
        print(f"[cache] skipped (low confidence={confidence})")

    return {
        "response": response,
        "thread_id": thread_id,
    }


async def stream_chat_events(request: ChatRequest, graph, session_id: str, api_keys: dict):
    thread_id = _resolve_thread_id(request, session_id)
    openai_api_key = api_keys["openai_api_key"]

    yield _format_sse(
        "status",
        {
            "type": "status",
            "step": "cache",
            "message": "Checking semantic and response caches...",
            "thread_id": thread_id,
        },
    )

    semantic_hit = get_semantic_cached_response(
        request.query,
        session_id,
        request.file_id,
        openai_api_key,
    )

    if semantic_hit:
        yield _format_sse(
            "cache_hit",
            {
                "type": "cache_hit",
                "cache": "semantic",
                "message": "Semantic cache hit.",
                "thread_id": thread_id,
            },
        )
        yield _format_sse(
            "final",
            {
                "type": "final",
                "response": semantic_hit,
                "thread_id": thread_id,
                "cached": "semantic",
                "sources": [],
            },
        )
        return

    cached_response = get_cached_response(
        user_id=session_id,
        file_id=request.file_id if request.file_id else None,
        query=request.query,
    )

    if cached_response:
        yield _format_sse(
            "cache_hit",
            {
                "type": "cache_hit",
                "cache": "response",
                "message": "Response cache hit.",
                "thread_id": thread_id,
            },
        )
        yield _format_sse(
            "final",
            {
                "type": "final",
                "response": cached_response,
                "thread_id": thread_id,
                "cached": True,
                "sources": [],
            },
        )
        return

    yield _format_sse(
        "status",
        {
            "type": "status",
            "step": "graph",
            "message": "Cache miss. Executing graph...",
            "thread_id": thread_id,
        },
    )

    state = GraphState(
        user_id=session_id,
        query=request.query,
        file_id=request.file_id if request.file_id else None,
    )

    invoke_config = _build_run_config(thread_id, api_keys)

    response_text = None
    confidence = None

    try:
        async for chunk in graph.astream(state, config=invoke_config, stream_mode="updates", version="v2"):
            if not isinstance(chunk, dict):
                continue

            updates = chunk.get("data") if chunk.get("type") == "updates" else chunk
            if not isinstance(updates, dict):
                continue

            for node_name, output in updates.items():
                if node_name not in STREAMABLE_NODES:
                    continue

                yield _format_sse(
                    "node",
                    {
                        "type": "node",
                        "node": node_name,
                        "status": "running",
                        "detail": _node_detail(node_name, "running"),
                    },
                )

                if isinstance(output, dict):
                    if output.get("response"):
                        response_text = output.get("response")
                    if output.get("confidence") is not None:
                        confidence = output.get("confidence")
                    if output.get("rewritten_query"):
                        yield _format_sse(
                            "status",
                            {
                                "type": "status",
                                "step": "rewrite",
                                "message": f"Rewritten query: {output.get('rewritten_query')}",
                                "thread_id": thread_id,
                            },
                        )

                yield _format_sse(
                    "node",
                    {
                        "type": "node",
                        "node": node_name,
                        "status": "done",
                        "detail": _node_detail(node_name, "done"),
                    },
                )

        if response_text is None:
            response_text = "I could not generate a response right now."

        # Capture the final graph state to extract context/sources
        sources = []
        try:
            final_state = graph.get_state(config=invoke_config)
            if final_state and hasattr(final_state, 'values'):
                final_state_dict = final_state.values
            else:
                final_state_dict = final_state if isinstance(final_state, dict) else {}

            context = final_state_dict.get("context") if isinstance(final_state_dict, dict) else None
            if context and isinstance(context, list):
                sources = [
                    {
                        "text": item.get("text") if isinstance(item, dict) else item,
                        "score": item.get("qdrant_score") if isinstance(item, dict) else 0,
                        "payload": item if isinstance(item, dict) else {},
                    }
                    for item in context
                ]
        except Exception as e:
            print(f"[chat] Warning: could not retrieve final state for sources: {e}")

        if confidence is None or confidence > 0.6:
            set_semantic_cache(
                request.query,
                response_text,
                session_id,
                request.file_id,
                openai_api_key,
            )

            set_cached_response(
                user_id=session_id,
                file_id=request.file_id,
                query=request.query,
                response=response_text,
            )

        yield _format_sse(
            "final",
            {
                "type": "final",
                "response": response_text,
                "thread_id": thread_id,
                "cached": False,
                "sources": sources,
            },
        )
    except Exception as exc:
        yield _format_sse(
            "error",
            {
                "type": "error",
                "message": str(exc),
                "thread_id": thread_id,
            },
        )
