from uuid import uuid4
from app.schemas.request import IngestRequest
from app.service.ingestService import ingest_data
from fastapi import APIRouter

router = APIRouter()

@router.post("/")
async def ingest(user_id: str):
    file_id = str(uuid4())

    file_path = "D:\Scalable Rag Agent\Backend\Sample Data\yash-dev26-langgraph-checkpointer-support-8a5edab282632443 (1).txt" # temporary hardcoded path for testing

    await ingest_data(
        IngestRequest(
            user_id=user_id,
            file_id=file_id,
            file_path=file_path
        )
    )

    return {
        "file_id": file_id,
        "message": "Ingested successfully"
    }