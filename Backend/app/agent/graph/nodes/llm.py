from app.schemas.state import GraphState
from app.config.openaiConfig import openai_client

def llm_node(state: GraphState):
    query = state.query if isinstance(state, GraphState) else state.get("query", "")
    messages = state.messages if isinstance(state, GraphState) else state.get("messages", [])

    if not messages:
        messages = [{"role": "user", "content": query}]

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return {"response": response.choices[0].message.content}