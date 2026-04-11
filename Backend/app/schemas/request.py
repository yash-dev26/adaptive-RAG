from typing import Optional

from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_id: str
    query: str
    file_id: Optional[str] = None

class IngestRequest(BaseModel):
    file_id : str
    user_id: str
    
    
    