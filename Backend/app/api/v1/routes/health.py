from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config.mongo import get_mongo_client
from app.config.redis import redis_client
from app.config.qdrantConfig import qdrant_client

router = APIRouter()


async def _check_mongo() -> bool:
    try:
        await get_mongo_client().admin.command("ping")
        return True
    except Exception:
        return False


async def _check_redis() -> bool:
    try:
        return bool(await redis_client.ping())
    except Exception:
        return False


async def _check_qdrant() -> bool:
    try:
        await qdrant_client.get_collections()
        return True
    except Exception:
        return False


@router.get("/")
async def health():
    """
    Note: this does a live round-trip to each dependency on every call
    """
    checks = {
        "mongo": await _check_mongo(),
        "redis": await _check_redis(),
        "qdrant": await _check_qdrant(),
    }
    all_healthy = all(checks.values())

    payload = {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
    }

    return JSONResponse(status_code=200 if all_healthy else 503, content=payload)