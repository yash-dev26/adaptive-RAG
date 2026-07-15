from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router
from app.agent.graph.graphBuilder import build_graph
from app.config.checkpointer import connect_to_mongodb
from app.cache.semantic_cleanup import cleanup_semantic_cache
import asyncio

from app.repository.qdrant import ensure_collections
from app.repository.chatSessions import ensure_indexes as ensure_chat_session_indexes
from app.config.server import config
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from app.config.rate_limiter import limiter


async def custom_rate_limit_handler(request, exc):
    """Custom rate limit handler to return clear JSON response."""
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Maximum 8 requests per minute allowed."},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    with connect_to_mongodb() as checkpointer:
        print("Successfully connected to MongoDB.")
        app.state.graph = build_graph(checkpointer=checkpointer)
        ensure_collections()
        ensure_chat_session_indexes()
        cleanup_task = asyncio.create_task(cleanup_semantic_cache())

        yield

        cleanup_task.cancel()

app = FastAPI(lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_handler)

origins = config["CORS_ORIGINS"].split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   # GET, POST, etc.
    allow_headers=["*"],
)

app.include_router(prefix="/api/v1", router=router)