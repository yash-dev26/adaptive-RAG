"""
Ingests the fixture document(s) in evaluation/fixtures/ through the app's
real ingestion pipeline (same chunking + hybrid embedding + Qdrant upsert
code the /api/v1/ingest route uses), so both eval scripts exercise the
actual retrieval path rather than a hand-built stand-in for it.

Requires the same environment the app itself needs to run (Qdrant, Redis,
Mongo reachable; OPENAI_API_KEY set) — this is dev/CI tooling, not
something that runs inside the app's own request path, so it talks to
those services directly rather than going through a running server.

Usage (from Backend/):
    python -m evaluation.seed_fixtures

Writes evaluation/fixtures/file_ids.json, e.g.:
    {"acme_handbook": "3f2e1a9c-...-file-id"}

Both run_routing_eval.py and run_ragas_eval.py read that file to resolve
a fixture name (used in cases/*.json as "file_ref") to a real file_id.
Re-running this script is safe: gen_embeddingsAndStoreInQdrant computes a
content hash and the ingestion pipeline treats a hash match for the same
user_id as a duplicate — see find_existing_file_id_by_content_hash in
app.repository.qdrant — so re-seeding won't create duplicate Qdrant
points, it'll just resolve to the same file_id again.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path

from app.ingestion.chunking import load_file, split_text
from app.ingestion.embeddings import gen_embeddingsAndStoreInQdrant
from app.repository.qdrant import find_existing_file_id_by_content_hash, ensure_collections
from app.config.server import config as server_config

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FILE_IDS_PATH = FIXTURES_DIR / "file_ids.json"

# Fixed, dedicated user_id for eval-seeded documents so they never
# collide with a real user's documents and are trivial to identify /
# clean up in Qdrant if needed.
EVAL_USER_ID = "eval-fixtures-user"


async def _seed_one(txt_path: Path, openai_api_key: str) -> str:
    content_hash = hashlib.sha256(txt_path.read_bytes()).hexdigest()

    existing_file_id = await find_existing_file_id_by_content_hash(
        collection_name=server_config["qdrant_collection_name"],
        user_id=EVAL_USER_ID,
        content_hash=content_hash,
    )
    if existing_file_id:
        print(f"[seed] {txt_path.name}: already ingested -> file_id={existing_file_id}")
        return existing_file_id

    from uuid import uuid4
    file_id = str(uuid4())

    pages = load_file(str(txt_path))
    chunks = split_text(pages)
    print(f"[seed] {txt_path.name}: {len(chunks)} chunk(s), embedding + upserting...")

    await gen_embeddingsAndStoreInQdrant(chunks, file_id, EVAL_USER_ID, content_hash, openai_api_key)
    print(f"[seed] {txt_path.name}: done -> file_id={file_id}")
    return file_id


async def main() -> None:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise SystemExit(
            "OPENAI_API_KEY must be set in the environment to seed fixtures "
            "(used for embeddings, same as the app's normal ingestion path)."
        )

    # Normally created once by the FastAPI app's own startup lifespan
    # (main.py -> ensure_collections()), as a background task whose
    # failures are silently swallowed. This script doesn't depend on the
    # app having been started successfully first — idempotent, safe to
    # call every run (checks for existing collections before creating).
    print("[seed] ensuring Qdrant collections exist...")
    await ensure_collections()

    txt_fixtures = sorted(FIXTURES_DIR.glob("*.txt"))
    if not txt_fixtures:
        raise SystemExit(f"No .txt fixtures found in {FIXTURES_DIR}")

    file_ids = {}
    for txt_path in txt_fixtures:
        fixture_name = txt_path.stem
        file_ids[fixture_name] = await _seed_one(txt_path, openai_api_key)

    FILE_IDS_PATH.write_text(json.dumps(file_ids, indent=2) + "\n")
    print(f"\n[seed] wrote {FILE_IDS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())