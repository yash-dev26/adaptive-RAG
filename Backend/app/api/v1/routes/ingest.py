from uuid import uuid4
from pathlib import Path
import hashlib
import asyncio
import os
from app.schemas.request import IngestRequest
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi import Depends
from app.auth.session import get_session_id, get_api_keys
from app.config.rate_limiter import limiter

# Background in-process ingest: parse/chunk upload and run the embedding
# pipeline asynchronously so the route returns quickly while batches
# are processed concurrently by the embedding pipeline.
from app.ingestion.chunking import load_file, split_text
from app.ingestion.embeddings import gen_embeddingsAndStoreInQdrant

router = APIRouter()

MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
UPLOAD_READ_CHUNK_SIZE = 1024 * 1024  # 1 MB per read

# asyncio.create_task() only holds a *weak* reference to the task it
# returns — if nothing else references it, it's eligible for garbage
# collection mid-execution and can silently disappear before finishing.
# Keeping a strong reference here (and dropping it via add_done_callback
# once the task finishes) is the fix recommended by the asyncio docs.
_background_tasks: set[asyncio.Task] = set()


async def _stream_to_disk(file: UploadFile, dest_path: Path, max_bytes: int) -> str:
    """
    Reads the upload in 1MB chunks and writes each straight to dest_path
    as it arrives, instead of buffering the whole file in memory first.
    The content hash is computed incrementally over the same chunks, so
    there's never a second full-size copy of the file sitting in RAM.

    If the size cap is exceeded partway through, the partial file on disk
    is removed before raising, so no truncated file is left behind.
    """
    hasher = hashlib.sha256()
    total = 0

    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = await file.read(UPLOAD_READ_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {max_bytes // (1024 * 1024)}MB upload limit",
                    )
                hasher.update(chunk)
                out.write(chunk)
    except HTTPException:
        dest_path.unlink(missing_ok=True)
        raise
    except Exception:
        dest_path.unlink(missing_ok=True)
        raise

    return hasher.hexdigest()


@router.post("/")
@limiter.limit("2/minute")
async def ingest(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id),
    api_keys: dict = Depends(get_api_keys),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".pdf", ".txt", ".docx"]:
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and DOCX uploads are supported")

    file_id = str(uuid4())
    upload_dir = Path(__file__).resolve().parents[4] / "uploaded_files"
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_path = upload_dir / f"{file_id}{suffix}"

    content_hash = await _stream_to_disk(file, stored_path, MAX_UPLOAD_SIZE_BYTES)

    print(f"[ingest] Received file upload: session_id={session_id}, file_id={file_id}, filename={file.filename}, content_hash={content_hash}")

    async def _background_process(path: Path, f_id, user_id, c_hash, openai_key):
        try:
            # Parsing (esp. pymupdf4llm for PDFs) and splitting are sync,
            # CPU/IO-bound calls. Running them inline on the event loop
            # would freeze every other request (chat, SSE, health checks)
            # for the duration. asyncio.to_thread keeps the loop free.
            parsed = await asyncio.to_thread(load_file, str(path))
            chunks_list = await asyncio.to_thread(split_text, parsed)
            print(f"[ingest][background] Parsed and split into {len(chunks_list)} chunks for file_id={f_id}")

            await gen_embeddingsAndStoreInQdrant(chunks_list, f_id, user_id, c_hash, openai_key)
            print(f"[ingest][background] Completed ingestion for file_id={f_id}")
        except Exception as e:
            print(f"[ingest][background] Ingestion failed for file_id={f_id}: {e}")
        finally:
            try:
                if path.exists():
                    path.unlink()
                    print(f"[ingest][background] Removed temp file {path}")
            except Exception as ex:
                print(f"[ingest][background] Failed to remove temp file {path}: {ex}")

    # Schedule background processing (won't survive process restart) and
    # keep a strong reference so it can't be garbage-collected mid-run.
    task = asyncio.create_task(
        _background_process(stored_path, file_id, session_id, content_hash, api_keys["openai_api_key"])
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    ingest_result = {
        "status": "queued",
        "file_id": file_id,
        "message": "Ingestion started in background; results will be available when complete.",
        "duplicate": False,
    }

    return {
        "file_id": ingest_result.get("file_id", file_id),
        "filename": file.filename,
        "message": ingest_result.get("message", "Ingested successfully"),
        "duplicate": ingest_result.get("duplicate", False),
    }