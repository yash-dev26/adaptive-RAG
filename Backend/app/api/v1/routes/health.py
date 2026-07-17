from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config.mongo import get_mongo_client
from app.config.redis import redis_client
from app.config.qdrantConfig import qdrant_client

router = APIRouter()


def _check_mongo() -> bool:
    try:
        get_mongo_client().admin.command("ping")
        return True
    except Exception:
        return False


def _check_redis() -> bool:
    try:
        return bool(redis_client.ping())
    except Exception:
        return False


def _check_qdrant() -> bool:
    try:
        qdrant_client.get_collections()
        return True
    except Exception:
        return False


@router.get("/")
async def health():
    """
    Note: this does a live round-trip to each dependency on every call
    """
    checks = {
        "mongo": _check_mongo(),
        "redis": _check_redis(),
        "qdrant": _check_qdrant(),
    }
    all_healthy = all(checks.values())

    payload = {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
    }

    return JSONResponse(status_code=200 if all_healthy else 503, content=payload)