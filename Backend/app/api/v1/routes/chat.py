from fastapi import APIRouter

from app.schemas.request import ChatRequest
from app.service.chatService import process_chat

router = APIRouter()

@router.post("/")
async def chat(request: ChatRequest):
    result = await process_chat(request)
    return {"response": result}

