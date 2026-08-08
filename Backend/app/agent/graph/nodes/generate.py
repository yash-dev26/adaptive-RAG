"""Centralized final-answer generation."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agent.graph.keys import extract_keys
from app.config.models import GENERATION_MODEL, GENERATION_PROVIDER
from app.schemas.state import GraphState
from app.service.LLMProviders import stream_completion

BARE_SYSTEM_PROMPT = "You are a helpful assistant that answers user queries."

RAG_SYSTEM_PROMPT = (
    "You are an AI assistant that generates responses based on user queries and "
    "retrieved context. Use the provided context to answer the user's query as "
    "accurately and helpfully as possible.\n\n"
    "Instructions:\n"
    "- Read ALL retrieved documents before answering.\n"
    "- The answer may require combining information from multiple documents.\n"
    "- Do NOT rely only on the highest-scoring document.\n"
    "- Connect evidence across documents when necessary.\n"
    "- Explain cause-and-effect if the question asks \"why\" or \"how\".\n"
    "- If multiple documents contribute different parts of the answer, synthesize "
    "them into one coherent response.\n"
    "- If the information is not present in the retrieved documents, explicitly say "
    "that it is not mentioned."
)

SUMMARY_SYSTEM_PROMPT = (
    "You are a helpful assistant that turns a condensed document summary into the "
    "final user-facing answer. Keep the response accurate, concise, and clearly written."
)

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


def _build_messages(state: GraphState) -> tuple[list, bool]:
    """Return (messages_to_send, needs_fallback_disclaimer)."""
    context = state.context or []

    if state.task_type == "summarize" and state.summary_text:
        return (
            [
                SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"User request:\n{state.query}\n\n"
                        f"Condensed document summary:\n{state.summary_text}"
                    )
                ),
            ],
            False,
        )

    if context:
        context_text = "\n\n".join(
            f"[Doc {i + 1}] {doc['text']}" for i, doc in enumerate(context)
        )
        return (
            [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                HumanMessage(content=f"Context:\n{context_text}\n\nQuery:\n{state.query}"),
            ],
            False,
        )

    # Bare / general-knowledge path. Reuse conversation history so chitchat
    # and general-knowledge threads stay coherent across turns. If we ended
    # up here after a RAG intent failed to ground on anything — rewrite
    # budget exhausted, no Tavily key, or a web search that came back empty —
    # flag the disclaimer so the user knows this answer isn't from their
    # document.
    messages = list(state.messages or [])
    if not messages:
        messages.append(SystemMessage(content=BARE_SYSTEM_PROMPT))

    last = messages[-1] if messages else None
    if last is None or getattr(last, "type", "") != "human" or getattr(last, "content", "") != state.query:
        messages.append(HumanMessage(content=state.query))

    return messages, state.intent == "rag"


async def generate_node(state: GraphState, config: RunnableConfig) -> dict:
    print("[flow] entering generate_node")

    messages, needs_disclaimer = _build_messages(state)
    openai_messages = [_to_openai_message(m) for m in messages]

    openai_api_key, groq_api_key = extract_keys(config)
    queue = (config or {}).get("configurable", {}).get("token_queue") if isinstance(config, dict) else None

    full_text = ""
    if needs_disclaimer:
        full_text += FALLBACK_DISCLAIMER
        if queue:
            await queue.put(("token", FALLBACK_DISCLAIMER))

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

    return {
        "messages": [HumanMessage(content=state.query), AIMessage(content=full_text)],
        "response": full_text,
    }