from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_id: str
    query: str

class IngestRequest(BaseModel):
    file_id : str
    user_id: str
    
    
    