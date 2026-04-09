from fastapi import APIRouter

from app.api.v1.routes import chat, health, ingest

router = APIRouter()

router.include_router(prefix="/ingest", router=ingest.router, tags=["ingest"])
router.include_router(prefix="/chat", router=chat.router, tags=["chat"])
router.include_router(prefix="/health", router=health.router, tags=["health"])