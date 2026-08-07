import asyncio

from app.cache.response_cache import get_cached_response, set_cached_response
from app.cache.semantic_cache import get_semantic_cached_response, set_semantic_cache
from app.schemas.request import ChatRequest
from app.schemas.state import GraphState
from app.service.graphRunner import build_run_config, resolve_thread_id
from app.service.sse import STREAMABLE_NODES, format_sse, node_detail
from app.service.threadService import record_turn


async def stream_chat_events(request: ChatRequest, graph, session_id: str, api_keys: dict):
    thread_id = resolve_thread_id(request, session_id)
    openai_api_key = api_keys["openai_api_key"]

    await record_turn(thread_id, session_id, request.file_id, request.file_name, request.query)

    yield format_sse(
        "status",
        {
            "type": "status",
            "step": "cache",
            "message": "Checking semantic and response caches...",
            "thread_id": thread_id,
        },
    )

    semantic_hit = await get_semantic_cached_response(
        request.query,
        session_id,
        request.file_id,
        openai_api_key,
    )

    if semantic_hit:
        yield format_sse(
            "cache_hit",
            {
                "type": "cache_hit",
                "cache": "semantic",
                "message": "Semantic cache hit.",
                "thread_id": thread_id,
            },
        )
        yield format_sse(
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

    cached_response = await get_cached_response(
        user_id=session_id,
        file_id=request.file_id if request.file_id else None,
        query=request.query,
    )

    if cached_response:
        yield format_sse(
            "cache_hit",
            {
                "type": "cache_hit",
                "cache": "response",
                "message": "Response cache hit.",
                "thread_id": thread_id,
            },
        )
        yield format_sse(
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

    yield format_sse(
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

    invoke_config = build_run_config(thread_id, api_keys)
    token_queue = asyncio.Queue()
    invoke_config["configurable"]["token_queue"] = token_queue

    response_text = None
    confidence = None

    async def run_graph():
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

                    await token_queue.put(
                        (
                            "node_update",
                            {
                                "type": "node",
                                "node": node_name,
                                "status": "running",
                                "detail": node_detail(node_name, "running"),
                                "output": output,
                            },
                        )
                    )

                    if isinstance(output, dict) and output.get("rewritten_query"):
                        await token_queue.put(
                            (
                                "status",
                                {
                                    "type": "status",
                                    "step": "rewrite",
                                    "message": f"Rewritten query: {output.get('rewritten_query')}",
                                    "thread_id": thread_id,
                                },
                            )
                        )

                    if isinstance(output, dict):
                        if output.get("response"):
                            token_queue.put_nowait(("response", output.get("response")))
                        if output.get("confidence") is not None:
                            token_queue.put_nowait(("confidence", output.get("confidence")))

                    await token_queue.put(
                        (
                            "node_update",
                            {
                                "type": "node",
                                "node": node_name,
                                "status": "done",
                                "detail": node_detail(node_name, "done"),
                                "output": output,
                            },
                        )
                    )
        except Exception as exc:
            await token_queue.put(
                (
                    "error",
                    {
                        "type": "error",
                        "message": str(exc),
                        "thread_id": thread_id,
                    },
                )
            )
        finally:
            await token_queue.put(("done", None))

    graph_task = asyncio.create_task(run_graph())

    while True:
        kind, payload = await token_queue.get()

        if kind == "done":
            break

        if kind == "token":
            yield format_sse("token", {"text": payload})
            continue

        if kind == "response":
            response_text = payload
            continue

        if kind == "confidence":
            confidence = payload
            continue

        if kind == "node_update":
            yield format_sse(
                "node",
                {
                    "type": "node",
                    "node": payload.get("node"),
                    "status": payload.get("status"),
                    "detail": payload.get("detail"),
                },
            )
            continue

        if kind == "status":
            yield format_sse("status", payload)
            continue

        if kind == "error":
            yield format_sse("error", payload)
            break

    await graph_task

    if response_text is None:
        response_text = "I could not generate a response right now."

    sources = []
    try:
        final_state = graph.get_state(config=invoke_config)
        if final_state and hasattr(final_state, "values"):
            final_state_dict = final_state.values
        else:
            final_state_dict = final_state if isinstance(final_state, dict) else {}

        context = final_state_dict.get("context") if isinstance(final_state_dict, dict) else None
        if context and isinstance(context, list):
            sources = [
                {
                    "text": item.get("text") if isinstance(item, dict) else item,
                    "score": item.get("qdrant_score", item.get("score", 0)) if isinstance(item, dict) else 0,
                    "payload": item if isinstance(item, dict) else {},
                }
                for item in context
            ]
    except Exception as exc:
        print(f"[chat] Warning: could not retrieve final state for sources: {exc}")

    if confidence is None or confidence > 0.6:
        await set_semantic_cache(
            request.query,
            response_text,
            session_id,
            request.file_id,
            openai_api_key,
        )

        await set_cached_response(
            user_id=session_id,
            file_id=request.file_id,
            query=request.query,
            response=response_text,
        )

    yield format_sse(
        "final",
        {
            "type": "final",
            "response": response_text,
            "thread_id": thread_id,
            "cached": False,
            "sources": sources,
        },
    )