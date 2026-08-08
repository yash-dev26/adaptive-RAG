"""
Helpers for turning a ChatRequest into a LangGraph invocation: resolving the
thread id and building the run-scoped `config` dict. Kept separate from
chatService so the graph-invocation plumbing isn't tangled up with caching
and SSE formatting.
"""

from uuid import uuid4
from app.schemas.request import ChatRequest
from app.config.server import config as server_config


def resolve_thread_id(request: ChatRequest, session_id: str) -> str:
    if request.thread_id:
        return request.thread_id

    if request.file_id:
        return f"{session_id}:{request.file_id}"

    return f"{session_id}:{uuid4()}"


def build_run_config(thread_id: str, api_keys: dict) -> dict:
    """
    BYOK keys go into `configurable`, NOT into GraphState — GraphState is
    what MongoDBSaver checkpoints on every turn (that's how conversation
    memory works), and we don't want a user's API key written to Mongo.
    `configurable` is per-invocation only; nodes read it via extract_keys().

    Tavily is server-side only (not BYOK, see app/auth/session.py) — always
    sourced from server_config, never from the request.
    """
    return {
        "configurable": {
            "thread_id": thread_id,
            "openai_api_key": api_keys["openai_api_key"],
            "groq_api_key": api_keys.get("groq_api_key"),
            "tavily_api_key": server_config.get("tavily_api_key"),
        }
    }