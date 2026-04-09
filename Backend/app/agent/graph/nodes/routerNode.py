from app.schemas.state import GraphState

def router_node(state: GraphState):
    file_id = state.file_id if isinstance(state, GraphState) else state.get("file_id")

    if file_id:
        return {
            "response": "file_id received, routing to RAG node",
            "intent": "rag",
        }
    else:
        return {
            "response": "no file_id, routing to LLM node",
            "intent": "llm",
        }