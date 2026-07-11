from app.schemas.state import GraphState
from app.service.LLMProviders import generate_completion
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from app.config.models import GENERATION_MODEL, GENERATION_PROVIDER
from app.agent.graph.keys import extract_keys
from langchain_core.runnables import RunnableConfig

def _to_openai_message(message) -> dict:
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    return {
        "role": role_map.get(getattr(message, "type", ""), "user"),
        "content": getattr(message, "content", ""),
    }

def generate_node(state: GraphState, config: RunnableConfig):
    print("[flow] entering generate_node")
    query = state.query
    context = state.context

    if context:
        context_text = "\n\n".join(doc["text"] for doc in context)
        messages = [
            SystemMessage(
                content="You are an AI assistant that generates responses based on user queries and retrieved context. Use the provided context to answer the user's query as accurately and helpfully as possible. If the context does not contain relevant information, tell the user that you do not have enough information to answer that question."
            ),
            HumanMessage(content=f"Context:\n{context_text}\n\nQuery:\n{query}"),
        ]
    else:
        messages = [HumanMessage(content=query)]

    openai_api_key, groq_api_key = extract_keys(config)

    response = generate_completion(
        provider=GENERATION_PROVIDER,
        model=GENERATION_MODEL,
        openai_api_key=openai_api_key,
        groq_api_key=groq_api_key,
        messages=[_to_openai_message(m) for m in messages],
    )

    assistant_text = response

    return {
        "messages": [HumanMessage(content=query), AIMessage(content=assistant_text)],
        "response": assistant_text,
    }
