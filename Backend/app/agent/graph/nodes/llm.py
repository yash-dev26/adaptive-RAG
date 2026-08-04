from app.schemas.state import GraphState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from app.service.LLMProviders import stream_completion
from app.config.models import GENERATION_MODEL, GENERATION_PROVIDER
from app.agent.graph.keys import extract_keys
from langchain_core.runnables import RunnableConfig

FALLBACK_DISCLAIMER = (
    "I couldn't find a good enough answer in the document for this, "
    "so here's a general-knowledge response instead:\n\n"
)


def _to_openai_message(message) -> dict:
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    return {
        "role": role_map.get(getattr(message, "type", ""), "user"),
        "content": getattr(message, "content", ""),
    }

async def llm_node(state: GraphState, config: RunnableConfig):
    print("[flow] entering llm_node")

    SYSTEM_PROMPT = """You are a helpful assistant that answers user queries."""

    messages = list(state.messages or [])

    if not messages:
        messages.append(SystemMessage(content=SYSTEM_PROMPT))

    if not messages or getattr(messages[-1], "type", "") != "human" or getattr(messages[-1], "content", "") != state.query:
        messages.append(HumanMessage(content=state.query))

    openai_messages = [_to_openai_message(message) for message in messages]

    openai_api_key, groq_api_key = extract_keys(config)
    queue = (config or {}).get("configurable", {}).get("token_queue") if isinstance(config, dict) else None
    full_text = ""

    async for delta in stream_completion(
        provider=GENERATION_PROVIDER,
        model=GENERATION_MODEL,
        openai_api_key=openai_api_key,
        groq_api_key=groq_api_key,
        messages=openai_messages,
    ):
        full_text += delta
        if queue:
            await queue.put(("token", delta))

    if state.intent == "rag":
        full_text = FALLBACK_DISCLAIMER + full_text

    return {
        "messages": [HumanMessage(content=state.query), AIMessage(content=full_text)],
        "response": full_text,
    }