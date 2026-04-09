from app.agent.graph.graphBuilder import build_graph
from app.schemas.request import ChatRequest
from app.schemas.state import GraphState

graph = build_graph()

async def process_chat(request: ChatRequest):
    initial_state = GraphState(user_id=request.user_id, query=request.query)
    final_state = graph.invoke(initial_state.model_dump())

    if isinstance(final_state, dict):
        return final_state.get("response")

    return getattr(final_state, "response", None)