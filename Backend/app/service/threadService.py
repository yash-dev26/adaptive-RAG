from fastapi import HTTPException

from app.repository import chatSessions
from app.schemas.thread import ThreadSummary, ThreadMessage

TITLE_MAX_LEN = 48

_ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system"}


def _truncate(text: str) -> str:
    if len(text) <= TITLE_MAX_LEN:
        return text
    return text[: TITLE_MAX_LEN - 1].rstrip() + "…"


def derive_title_from_query(query: str) -> str:
    normalized = " ".join(query.strip().split())
    return _truncate(normalized) if normalized else "New chat"


def derive_title_from_filename(file_name: str) -> str:
    # Strip the extension — "quarterly_report.pdf" -> "quarterly_report"
    # reads better as a chat title than the raw filename.
    stem = file_name.rsplit(".", 1)[0].strip() or file_name
    return _truncate(stem)


def record_turn(thread_id: str, session_id: str, file_id: str | None, file_name: str | None, query: str) -> None:
    
    existing = chatSessions.get_thread(thread_id)
    already_had_file = bool(existing and existing.get("file_id"))

    title_override = None
    if existing is None:
        title_override = derive_title_from_filename(file_name) if file_name else derive_title_from_query(query)
    elif file_name and not already_had_file:
        title_override = derive_title_from_filename(file_name)

    chatSessions.upsert_thread(
        thread_id=thread_id,
        session_id=session_id,
        file_id=file_id,
        file_name=file_name,
        title=title_override,
    )


def list_threads_for_session(session_id: str) -> list[ThreadSummary]:
    rows = chatSessions.list_threads(session_id)
    return [
        ThreadSummary(
            thread_id=row["thread_id"],
            file_id=row.get("file_id"),
            file_name=row.get("file_name"),
            title=row.get("title") or "New chat",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


def get_thread_messages(thread_id: str, session_id: str, graph) -> list[ThreadMessage]:
    owner = chatSessions.get_thread_owner(thread_id)

    # Same 404 for "doesn't exist" and "exists but isn't yours" — confirming
    # a thread_id belongs to someone else is itself a small information leak.
    if owner is None or owner != session_id:
        raise HTTPException(status_code=404, detail="Thread not found")

    state = graph.get_state(config={"configurable": {"thread_id": thread_id}})
    raw_messages = (state.values or {}).get("messages", []) if state else []

    return [
        ThreadMessage(
            role=_ROLE_MAP.get(getattr(message, "type", ""), "user"),
            content=getattr(message, "content", ""),
        )
        for message in raw_messages
        if getattr(message, "content", "").strip()
    ]