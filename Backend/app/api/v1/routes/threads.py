from fastapi import APIRouter, Depends, Request

from app.auth.session import get_session_id
from app.schemas.thread import ThreadListResponse, ThreadMessagesResponse
from app.service.threadService import list_threads_for_session, get_thread_messages

router = APIRouter()


@router.get("/", response_model=ThreadListResponse)
async def list_threads(session_id: str = Depends(get_session_id)):
    return ThreadListResponse(threads=list_threads_for_session(session_id))


@router.get("/{thread_id}/messages", response_model=ThreadMessagesResponse)
async def thread_messages(
    thread_id: str,
    request: Request,
    session_id: str = Depends(get_session_id),
):
    messages = get_thread_messages(thread_id, session_id, request.app.state.graph)
    return ThreadMessagesResponse(thread_id=thread_id, messages=messages)