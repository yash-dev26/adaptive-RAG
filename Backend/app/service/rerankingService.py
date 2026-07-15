import logging

from app.config.reranking_models import (
    RERANKER_MODEL,
    get_reranker_client,
)

logger = logging.getLogger(__name__)


def rerank(query: str, docs: list[dict], top_k: int = 4) -> list[dict]:
    """
    Reranks retrieved documents and returns the top_k most relevant.
    Falls back to retrieval order if reranking fails.
    """

    if not docs:
        return []

    try:
        client = get_reranker_client()

        response = client.rerank(
            model=RERANKER_MODEL,
            query=query,
            documents=[doc["text"] for doc in docs],
            top_n=top_k,
        )

        reranked_docs = []

        for result in response.results:
            doc = docs[result.index].copy()
            doc["rerank_score"] = result.relevance_score
            reranked_docs.append(doc)

        return reranked_docs

    except Exception:
        logger.exception("Failed to rerank documents.")
        return docs[:top_k]