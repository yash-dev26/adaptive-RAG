from pydantic import BaseModel

class ChatResponse(BaseModel):
    response: str

class IngestResponse(BaseModel):
    file_id: str
    status: str
    message: str
    