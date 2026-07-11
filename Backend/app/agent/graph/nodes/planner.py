import json
from app.schemas.state import GraphState
from app.service.LLMProviders import generate_completion
from app.agent.graph.keys import extract_keys
from langchain_core.runnables import RunnableConfig
from app.config.models import PLANNER_MODEL, PLANNER_PROVIDER

_CLASSIFY_PROMPT = """\
You are a query intent classifier for a retrieval system.

Classify the user query into exactly one of three intents:
- "needs_retrieval": the query is asking about specific content, facts, or details that likely exist in an uploaded document
- "general_knowledge": the query is a factual or reasoning question answerable from general knowledge, with no document context required
- "chitchat": the query is a greeting, filler, thanks, or social exchange with no information need

Return ONLY valid JSON with no explanation:
{"intent": "needs_retrieval" | "general_knowledge" | "chitchat"}
"""

def _classify_intent(query: str, openai_api_key: str, groq_api_key: str | None) -> str:
    try:
        response_text = generate_completion(
            provider=PLANNER_PROVIDER,
            model=PLANNER_MODEL,
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
            messages=[
                {"role": "system", "content": _CLASSIFY_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response_text)
        intent = result.get("intent", "needs_retrieval")
        if intent not in {"needs_retrieval", "general_knowledge", "chitchat"}:
            return "needs_retrieval"
        return intent
    except Exception as e:
        print(f"[planner] intent classification failed ({e}), defaulting to needs_retrieval")
        return "needs_retrieval"


def _is_ambiguous(query: str) -> bool:
    ambiguous_signals = {"it", "this", "that", "they", "he", "she", "the thing", "stuff"}
    words = set(query.lower().split())
    return bool(words & ambiguous_signals)


def pre_retrieval_planner_node(state: GraphState, config: RunnableConfig) -> dict:
    print("[flow] entering pre_retrieval_planner_node")
    query = state.query.strip()

    if not state.file_id:
        return {"intent": "llm", "rewrite_type": "none"}

    openai_api_key, groq_api_key = extract_keys(config)

    llm_intent = _classify_intent(query, openai_api_key, groq_api_key)
    print(f"[planner] classified intent: {llm_intent}")

    if llm_intent in {"chitchat", "general_knowledge"}:
        return {"intent": "llm", "rewrite_type": "none"}

    if _is_ambiguous(query.lower()):
        return {"intent": "rag", "rewrite_type": "single"}

    return {"intent": "rag", "rewrite_type": "none"}