"""
Retrieval helper.

Fixes
-----
- Collection name now comes from config (not hard-coded "rag-collection").
- Payload key is "chunk" during ingestion but aliased to "text" too; reads both.
"""

from app.ingestion.embeddings import gen_embeddings
from app.repository.qdrant import search_data
from app.config.server import config


def retrieve_relevant_documents(query: str, top_k: int = 5) -> list[dict]:
    embedding = gen_embeddings(query)
    results = search_data(
        collection_name=config["qdrant_collection_name"],
        query_vector=embedding,
        top_k=top_k,
    )

    documents = []
    for hit in results:
        payload = hit.payload or {}
        # ingestion stores under "chunk" key (with "text" alias added in new code)
        text = payload.get("text") or payload.get("chunk", "")
        documents.append({"text": text, "score": hit.score})

    return documents