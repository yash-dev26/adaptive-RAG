from functools import lru_cache
from typing import List

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

SPARSE_MODEL_NAME = "Qdrant/bm25"


@lru_cache(maxsize=1)
def _get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def gen_sparse_document_embeddings(texts: List[str]) -> List[SparseVector]:
    """BM25 embeddings for chunks going INTO the index at ingest time."""
    model = _get_sparse_model()
    embeddings = list(model.embed(texts))
    return [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in embeddings
    ]


def gen_sparse_query_embedding(text: str) -> SparseVector:
    """
    BM25 weights query terms differently than document terms (no document-
    length normalization on the query side) — fastembed exposes query_embed
    specifically for this, don't reuse embed() for queries.
    """
    model = _get_sparse_model()
    embedding = next(model.query_embed(text))
    return SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())