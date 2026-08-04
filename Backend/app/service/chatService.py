from app.schemas.request import ChatRequest
from app.schemas.state import GraphState
from app.cache.response_cache import get_cached_response, set_cached_response
from app.cache.semantic_cache import get_semantic_cached_response, set_semantic_cache
from app.service.threadService import record_turn
from app.service.graphRunner import resolve_thread_id, build_run_config
import asyncio

async def process_chat(request: ChatRequest, graph, session_id: str, api_keys: dict):
    thread_id = resolve_thread_id(request, session_id)
    openai_api_key = api_keys["openai_api_key"]

    await record_turn(thread_id, session_id, request.file_id, request.file_name, request.query)

    semantic_hit, cached_response = await asyncio.gather(
    get_semantic_cached_response(request.query, session_id, request.file_id, openai_api_key),
    get_cached_response(user_id=session_id, file_id=request.file_id, query=request.query),
    )

    if semantic_hit:
        return {"response": semantic_hit, "thread_id": thread_id, "cached": "semantic"}
    if cached_response:
        return {"response": cached_response, "thread_id": thread_id, "cached": True}
    
    print("[cache] response cache MISS")

    state = GraphState(
        user_id=session_id,
        query=request.query,
        file_id=request.file_id if request.file_id else None
    )

    invoke_config = build_run_config(thread_id, api_keys)

    result = await graph.ainvoke(state, config=invoke_config) if graph else None

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
