from app.schemas import response
from app.schemas.request import ChatRequest
from app.schemas.state import GraphState
from app.cache.response_cache import get_cached_response, set_cached_response
from app.cache.semantic_cache import get_semantic_cached_response, set_semantic_cache

from uuid import uuid4

def _resolve_thread_id(request: ChatRequest) -> str:
    if request.thread_id:
        return request.thread_id

    if request.file_id:
        return f"{request.user_id}:{request.file_id}"

    return f"{request.user_id}:{uuid4()}"

async def process_chat(request: ChatRequest, graph):
    thread_id = _resolve_thread_id(request)

    semantic_hit = get_semantic_cached_response(request.query)

    if semantic_hit:
        return {
            "response": semantic_hit,
            "thread_id": thread_id,
            "cached": "semantic"
        }

    cached_response = get_cached_response(
        user_id=request.user_id,
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
        user_id=request.user_id,
        query=request.query,
        file_id=request.file_id if request.file_id else None
    )

    invoke_config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = graph.invoke(state, config=invoke_config) if graph else None

    if isinstance(result, dict):
        response = result.get("response", "I could not generate a response right now.")
        confidence = result.get("confidence", None)
    else:
        response = getattr(result, "response", "I could not generate a response right now.")
        confidence = getattr(result, "confidence", None)

    if confidence is None or confidence > 0.6:
        set_semantic_cache(request.query, response)
        set_cached_response(
            user_id=request.user_id,
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