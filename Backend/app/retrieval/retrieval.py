from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
)

from app.ingestion.embeddings import gen_embeddings
from app.ingestion.sparse_embeddings import gen_sparse_query_embedding
from app.config.qdrantConfig import qdrant_client
from app.config.server import config
from app.retrieval.fusion import reciprocal_rank_fusion


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
    """

    payload_filter = _build_filter(file_id, user_id)

    # Start embedding request immediately
    dense_embedding_task = asyncio.create_task(
        gen_embeddings(query, openai_api_key)
    )

    # CPU work overlaps with OpenAI request
    sparse_embedding = gen_sparse_query_embedding(query)

    # Wait for OpenAI
    dense_embedding = await dense_embedding_task

    # Run both vector searches concurrently
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
            query=sparse_embedding,
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