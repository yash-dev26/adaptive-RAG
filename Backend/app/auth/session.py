from fastapi import Header, HTTPException


def get_session_id(
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> str:
    """
    Identity for BYOK mode.

    There is no login. The frontend generates a random id once, persists it
    in localStorage, and sends it on every request. It exists only to scope
    Qdrant payloads (user_id filter) so one browser's uploads/chat history
    stay isolated from another's.
    """
    if not x_session_id or not x_session_id.strip():
        raise HTTPException(status_code=400, detail="Missing X-Session-Id header")
    return x_session_id.strip()


def get_api_keys(
    x_openai_key: str | None = Header(default=None, alias="X-OpenAI-Key"),
    x_groq_key: str | None = Header(default=None, alias="X-Groq-Key"),
) -> dict:
    """
    BYOK credentials, read fresh on every request.

    Keys live only for the duration of this request (passed as function
    args / LangGraph `configurable`, never written to state, cache, logs,
    or Mongo checkpoints).
    """
    if not x_openai_key or not x_openai_key.strip():
        raise HTTPException(
            status_code=401,
            detail="Missing X-OpenAI-Key header. An OpenAI API key is required.",
        )

    return {
        "openai_api_key": x_openai_key.strip(),
        "groq_api_key": x_groq_key.strip() if x_groq_key and x_groq_key.strip() else None,
    }
