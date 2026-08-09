from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse

from app.schemas.request import ChatRequest
from app.service.chatService import process_chat
from app.service.chatStream import stream_chat_events

from app.auth.session import get_session_id, get_api_keys
from app.config.rate_limiter import limiter

router = APIRouter()

@router.post("/")
@limiter.limit("8/minute")
async def chat(
    payload: ChatRequest,
    request: Request,
    session_id: str = Depends(get_session_id),
    api_keys: dict = Depends(get_api_keys),
):
    result = await process_chat(payload, request.app.state.graph, session_id, api_keys)
    return result

@router.post("/stream")
@limiter.limit("8/minute")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    session_id: str = Depends(get_session_id),
    api_keys: dict = Depends(get_api_keys),
):
    return StreamingResponse(
        stream_chat_events(payload, request.app.state.graph, session_id, api_keys),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )