from app.ingestion.embeddings import gen_embeddings
from app.repository.qdrant import search_data
from app.config.server import config

from fastembed import SparseTextEmbedding
from app.config.qdrantConfig import qdrant_client

sparse_model = SparseTextEmbedding()

def retrieve_relevant_documents(query: str, top_k: int = 5) -> list[dict]:
    dense_embedding = gen_embeddings(query)
    sparse_embedding = list(sparse_model.embed([query]))[0]

    results = qdrant_client.query_points(
        collection_name=config["qdrant_collection_name"],
        query={
            "vector": dense_embedding,
            "sparse_vector": {
                "text": sparse_embedding
            }
        },
        limit=top_k
    ).points

    documents = []
    for hit in results:
        payload = hit.payload or {}
        text = payload.get("text") or payload.get("chunk", "")
        documents.append({
            "text": text,
            "score": hit.score   # now hybrid score 🔥
        })

    return documents