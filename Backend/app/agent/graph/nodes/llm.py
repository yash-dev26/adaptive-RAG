from app.schemas.state import GraphState
from app.config.openaiConfig import openai_client

def llm_node(state: GraphState):

    SYSTEM_PROMPT = """You are a helpful assistant that answers user queries."""

    messages = state.messages or [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state.query}
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )

    return {"response": response.choices[0].message.content}