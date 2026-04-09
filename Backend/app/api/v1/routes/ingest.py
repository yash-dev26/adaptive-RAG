from fastapi import APIRouter, UploadFile, File
from app.service.ingestService import ingest_data
import os
import shutil
from types import SimpleNamespace

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/")
async def ingest( user_id: str = "default_user_id", file_id: str = "default_file_id"):
    try:
        file_path = r"D:\Scalable Rag Agent\Backend\Sample Data\yash-dev26-langgraph-checkpointer-support-8a5edab282632443 (1).txt"
        # file_path = f"{UPLOAD_DIR}/{request.file.filename}"

        # with open(file_path, "wb") as buffer:
        #     shutil.copyfileobj(request.file, buffer)

        # pipeline
        result = await ingest_data(request=SimpleNamespace(pdf_file=file_path, user_id=user_id, file_id=file_id))

        
        return {"message": "PDF processed successfully"}
    except Exception as e:
        return {"error": str(e)}
    