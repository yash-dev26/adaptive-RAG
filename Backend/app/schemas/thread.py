from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel


class ThreadSummary(BaseModel):
    thread_id: str
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    title: str
    created_at: datetime
    updated_at: datetime


class ThreadListResponse(BaseModel):
    threads: List[ThreadSummary]


class ThreadMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ThreadMessagesResponse(BaseModel):
    thread_id: str
    messages: List[ThreadMessage]