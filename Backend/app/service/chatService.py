from app.agent.graph.graphBuilder import build_graph
from app.schemas.request import ChatRequest
from app.schemas.state import GraphState

graph = build_graph()


async def process_chat(request: ChatRequest):
    state = GraphState(
        user_id=request.user_id,
        query=request.query,
        file_id=request.file_id if request.file_id else None
    )

    result = graph.invoke(state)

    if isinstance(result, dict):
        return result.get("response") or "I could not generate a response right now."

    return getattr(result, "response", None) or "I could not generate a response right now."