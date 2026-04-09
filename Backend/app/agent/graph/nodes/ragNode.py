from app.schemas.state import GraphState
from app.rag.retrieval import retrieve_relevant_documents

def rag_node(state: GraphState):
    query = state.query if isinstance(state, GraphState) else state.get("query", "")
    relevant_docs = retrieve_relevant_documents(query)
    return {"retrieved_docs": relevant_docs}
