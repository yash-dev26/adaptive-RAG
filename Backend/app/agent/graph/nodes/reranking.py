import logging

from app.schemas.state import GraphState
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_reranker_model = None
_reranker_init_failed = False

_RERANKER_MODEL_CANDIDATES = [
    "BAAI/bge-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
]


def _get_reranker_model():
    global _reranker_model, _reranker_init_failed

    if _reranker_model is not None:
        return _reranker_model

    if _reranker_init_failed:
        return None

    for model_id in _RERANKER_MODEL_CANDIDATES:
        try:
            _reranker_model = CrossEncoder(model_id, max_length=512)
            logger.info("Loaded reranker model: %s", model_id)
            return _reranker_model
        except Exception as exc:
            logger.warning("Failed to load reranker model '%s': %s", model_id, exc)

    _reranker_init_failed = True
    logger.error("All reranker model candidates failed. Falling back to retrieved order.")
    return None


def reranking_node(state: GraphState):
    print("[flow] entering reranking_node")
    query = state.rewritten_query or state.query
    docs = state.context or []

    if not docs:
        return state

    pairs = [(query, doc) for doc in docs]

    model = _get_reranker_model()
    if model is None:
        return {"context": docs[:4]}

    scores = model.predict(pairs)

    scored_docs = list(zip(docs, scores))
    ranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    top_docs = [doc for doc, _ in ranked[:4]]

    return {
        "context": top_docs
    }