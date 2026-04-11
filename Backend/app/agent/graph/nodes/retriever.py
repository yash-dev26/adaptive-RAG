from app.schemas.state import GraphState
from app.retrieval.retrieval import retrieve_relevant_documents

def retrieve_node(state: GraphState):
    docs = retrieve_relevant_documents(state.query)

    return {
        "context": docs
    }

