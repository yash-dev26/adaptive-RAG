import logging

from app.config.rerankerModel import RERANKER_MODEL, get_reranker
from app.utils.retry import external_api_retry
logger = logging.getLogger(__name__)


@external_api_retry(max_attempts=2)
async def _do_rerank(client, model, query, docs, top_k):
    return await client.rerank(model=model, query=query, documents=docs, top_n=top_k)
async def rerank(query: str, docs: list[dict], top_k: int = 4) -> list[dict]:
    """
    Reranks retrieved documents and returns the top_k most relevant.
    Falls back to retrieval order if reranking fails.
    """

    if not docs:
        return []

    try:
        client = get_reranker()

        response = await _do_rerank(client, RERANKER_MODEL, query, [d["text"] for d in docs], top_k)

        reranked_docs = []

        for result in response.results:
            doc = docs[result.index].copy()
            doc["rerank_score"] = result.relevance_score
            reranked_docs.append(doc)

        return reranked_docs

    except Exception:
        logger.exception("Failed to rerank documents.")
        return docs[:top_k]