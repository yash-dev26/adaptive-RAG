from app.schemas.state import GraphState
from app.config.openaiConfig import openai_client

def generate_node(state: GraphState):
    query = state.query
    context = state.context
    
    if context:
        context_text = "\n\n".join(context)

        

        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that generates responses based on user queries and retrieved context. Use the provided context to answer the user's query as accurately and helpfully as possible. If the context does not contain relevant information, tell the user that you do not have enough information to answer that question."
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuery:\n{query}"
            }
        ]
    else:
        messages = state.messages or [
            {"role": "user", "content": query}
        ]

    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )

    return {
        "response": response.choices[0].message.content
    }