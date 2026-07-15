from typing import Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    # user_id is intentionally NOT here as it's derived server-side from the
    # X-Session-Id header (see app/auth/session.py) so a client can't set
    # someone else's id in the body and read their documents.
    query: str
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    thread_id: Optional[str] = None


class IngestRequest(BaseModel):
    file_id: str
    user_id: str
    file_path: str
    content_hash: str