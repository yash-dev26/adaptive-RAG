from app.ingestion.embeddings import gen_embeddingsAndStoreInQdrant
from app.ingestion.chunking import load_file, split_text
from app.config.server import config
from app.repository.qdrant import find_existing_file_id_by_content_hash
from fastapi import HTTPException

async def ingest_data(request, openai_api_key: str):
    print(f"[ingest] Starting ingestion for user_id={request.user_id}, file_id={request.file_id}")
    existing_file_id = find_existing_file_id_by_content_hash(
        collection_name=config["qdrant_collection_name"],
        user_id=request.user_id,
        content_hash=request.content_hash,
    )

    if existing_file_id:
        print(
            f"[ingest] Duplicate file detected for user_id={request.user_id}; "
            f"existing file_id={existing_file_id}. Skipping re-ingestion."
        )
        return {
            "status": "success",
            "duplicate": True,
            "file_id": existing_file_id,
            "message": "Duplicate file already ingested.",
        }

    try:
        text = load_file(request.file_path)
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file could not be parsed")

    chunks = split_text(text)
    print(f"Split text into chunks: {len(chunks)} chunks")

    result = await gen_embeddingsAndStoreInQdrant(
        chunks,
        request.file_id,
        request.user_id,
        request.content_hash,
        openai_api_key,
    )
    return result
