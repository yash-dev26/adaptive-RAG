from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, Document,
)

from app.ingestion.embeddings import gen_embeddings
from app.config.qdrantConfig import qdrant_client
from app.config.server import config
from app.retrieval.fusion import reciprocal_rank_fusion

BM25_MODEL_NAME = "Qdrant/bm25"


def _build_filter(file_id: str | None, user_id: str | None):
    filter_conditions = []
    if file_id:
        filter_conditions.append(FieldCondition(key="file_id", match=MatchValue(value=file_id)))
    if user_id:
        filter_conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
    return Filter(must=filter_conditions) if filter_conditions else None


import asyncio

async def retrieve_relevant_documents(
    query: str,
    openai_api_key: str,
    top_k: int = 5,
    file_id: str | None = None,
    user_id: str |None = None,
) -> list[dict]:
    """
    Hybrid search: dense + sparse fused with RRF.

    Dense stays client-side (OpenAI's text-embedding-3-small, via
    gen_embeddings). Sparse is generated server-side by Qdrant itself —
    passing a `Document(model="Qdrant/bm25")` instead of a precomputed
    vector — so there's no local BM25 model/runtime to load, race on, or
    OOM on in this process. See ingestion/embeddings.py for the same
    change on the write side, and config/qdrantConfig.py for the
    `cloud_inference=True` client flag this requires.
    """

    payload_filter = _build_filter(file_id, user_id)

    dense_embedding = await gen_embeddings(query, openai_api_key)

    # Run both vector searches concurrently. The sparse side does its
    # BM25 inference inside this call, on Qdrant's side, not before it —
    # there's no separate local step to overlap with the dense request
    # anymore.
    dense_result, sparse_result = await asyncio.gather(
        qdrant_client.query_points(
            collection_name=config["qdrant_collection_name"],
            query=dense_embedding,
            using="dense",
            query_filter=payload_filter,
            limit=top_k,
            with_payload=True,
        ),
        qdrant_client.query_points(
            collection_name=config["qdrant_collection_name"],
            query=Document(text=query, model=BM25_MODEL_NAME),
            using="sparse",
            query_filter=payload_filter,
            limit=top_k,
            with_payload=True,
        ),
    )

    dense_hits = dense_result.points
    sparse_hits = sparse_result.points

    dense_ranking = [
        {
            "text": (hit.payload or {}).get("text")
            or (hit.payload or {}).get("chunk", ""),
            "score": hit.score,
            "source": "dense",
        }
        for hit in dense_hits
    ]

    sparse_ranking = [
        {
            "text": (hit.payload or {}).get("text")
            or (hit.payload or {}).get("chunk", ""),
            "score": hit.score,
            "source": "sparse",
        }
        for hit in sparse_hits
    ]

    if not sparse_ranking:
        return dense_ranking
    if not dense_ranking:
        return sparse_ranking

    return reciprocal_rank_fusion([dense_ranking, sparse_ranking])[:top_k]


async def fetch_all_chunks_for_file(
    file_id: str,
    user_id: str | None = None,
    page_size: int = 256,
) -> list[dict]:
    """
    Scroll every chunk belonging to a file, in
    original document order. Summarization needs the full chunk set instead.
    """
    payload_filter = _build_filter(file_id, user_id)

    all_points = []
    offset = None
    while True:
        points, next_offset = await qdrant_client.scroll(
            collection_name=config["qdrant_collection_name"],
            scroll_filter=payload_filter,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset

    chunks = [
        {
            "text": (p.payload or {}).get("text") or (p.payload or {}).get("chunk", ""),
            "chunk_index": (p.payload or {}).get("chunk_index", 0),
            "page": (p.payload or {}).get("page"),
        }
        for p in all_points
    ]
    chunks.sort(key=lambda c: c["chunk_index"])
    _dedupe_overlap(chunks)
    return chunks


def _dedupe_overlap(chunks: list[dict], max_overlap: int = 100) -> None:
    """
    Remove overlapping text between adjacent chunks. This is a heuristic to avoid
    duplication in the final text when chunks are concatenated because the textSplitter has chunk overlap and will increase the number of tokens in the final text. 
    """
    for i in range(1, len(chunks)):
        prev_text = chunks[i - 1]["text"]
        curr_text = chunks[i]["text"]
        window = min(max_overlap, len(prev_text), len(curr_text))

        # Try the longest possible overlap first, shrink until an exact
        # suffix-of-prev == prefix-of-curr match is found (or none exists).
        for overlap_len in range(window, 0, -1):
            if prev_text[-overlap_len:] == curr_text[:overlap_len]:
                chunks[i]["text"] = curr_text[overlap_len:]
                break