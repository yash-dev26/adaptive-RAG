from datetime import datetime, timezone
from typing import Optional

from app.config.mongo import get_app_db

COLLECTION_NAME = "chat_sessions"


def _collection():
    return get_app_db()[COLLECTION_NAME]


def ensure_indexes() -> None:
    """Called once at app startup (see main.py), not per-request."""
    collection = _collection()
    collection.create_index([("session_id", 1), ("updated_at", -1)])
    collection.create_index("thread_id", unique=True)


def get_thread(thread_id: str) -> Optional[dict]:
    return _collection().find_one({"thread_id": thread_id})


def upsert_thread(
    thread_id: str,
    session_id: str,
    file_id: Optional[str],
    file_name: Optional[str],
    title: Optional[str] = None,
) -> None:
    
    now = datetime.now(timezone.utc)

    set_fields = {"updated_at": now}
    if file_id is not None:
        set_fields["file_id"] = file_id
        set_fields["file_name"] = file_name
    if title is not None:
        set_fields["title"] = title

    set_on_insert = {
        "thread_id": thread_id,
        "session_id": session_id,
        "created_at": now,
    }
   
    if title is None:
        set_on_insert["title"] = "New chat"

    _collection().update_one(
        {"thread_id": thread_id},
        {"$set": set_fields, "$setOnInsert": set_on_insert},
        upsert=True,
    )


def list_threads(session_id: str) -> list[dict]:
    cursor = _collection().find({"session_id": session_id}).sort("updated_at", -1)
    return list(cursor)


def get_thread_owner(thread_id: str) -> Optional[str]:
    doc = _collection().find_one({"thread_id": thread_id}, {"session_id": 1})
    return doc["session_id"] if doc else None