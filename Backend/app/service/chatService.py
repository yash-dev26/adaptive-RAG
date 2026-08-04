from app.schemas.request import ChatRequest
from app.schemas.state import GraphState
from app.cache.response_cache import get_cached_response, set_cached_response
from app.cache.semantic_cache import get_semantic_cached_response, set_semantic_cache
from app.service.threadService import record_turn
from app.service.graphRunner import resolve_thread_id, build_run_config
from app.service.sse import STREAMABLE_NODES, format_sse, node_detail


async def process_chat(request: ChatRequest, graph, session_id: str, api_keys: dict):
    thread_id = resolve_thread_id(request, session_id)
    openai_api_key = api_keys["openai_api_key"]

    await record_turn(thread_id, session_id, request.file_id, request.file_name, request.query)

    semantic_hit = await get_semantic_cached_response(
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

    cached_response = await get_cached_response(
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

    invoke_config = build_run_config(thread_id, api_keys)

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
        #Always store original query(fallback baseline)
        await set_semantic_cache(
            request.query,
            response,
            session_id,
            request.file_id,
            openai_api_key,
        )

        #Store rewritten query(single)
        if rewritten_query:
            await set_semantic_cache(
                rewritten_query,
                response,
                session_id,
                request.file_id,
                openai_api_key,
            )

        #Store multi queries
        if queries:
            for q in queries:
                await set_semantic_cache(
                    q,
                    response,
                    session_id,
                    request.file_id,
                    openai_api_key,
                )

        await set_cached_response(
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

                yield format_sse(
                    "node",
                    {
                        "type": "node",
                        "node": node_name,
                        "status": "running",
                        "detail": node_detail(node_name, "running"),
                    },
                )

                if isinstance(output, dict):
                    if output.get("response"):
                        response_text = output.get("response")
                    if output.get("confidence") is not None:
                        confidence = output.get("confidence")
                    if output.get("rewritten_query"):
                        yield format_sse(
                            "status",
                            {
                                "type": "status",
                                "step": "rewrite",
                                "message": f"Rewritten query: {output.get('rewritten_query')}",
                                "thread_id": thread_id,
                            },
                        )

                yield format_sse(
                    "node",
                    {
                        "type": "node",
                        "node": node_name,
                        "status": "done",
                        "detail": node_detail(node_name, "done"),
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
    except Exception as exc:
        yield format_sse(
            "error",
            {
                "type": "error",
                "message": str(exc),
                "thread_id": thread_id,
            },
        )