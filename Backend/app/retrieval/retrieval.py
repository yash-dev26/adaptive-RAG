from app.repository.qdrant import qdrant_client
from app.ingestion.embeddings import gen_embeddings

def retrieve_relevant_documents(query: str, top_k: int = 5):

    embedding = gen_embeddings(query)
    results = qdrant_client.search(
        collection_name="rag-collection",
        query_vector=embedding,
        limit=top_k
    )

    return [hit.payload["text"] for hit in results]