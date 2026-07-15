import json

from app.agent.graph.keys import extract_keys
from app.schemas.state import GraphState
from langchain_core.runnables import RunnableConfig
from app.service.LLMProviders import generate_completion
from app.config.models import PLANNER_PROVIDER, PLANNER_MODEL

_CHITCHAT_PHRASES = {
    "hi", "hii", "hiii", "hello", "hey", "heya", "yo",
    "good morning", "good afternoon", "good evening", "good night",
    "thanks", "thank you", "thx", "ty",
    "ok", "okay", "cool", "nice", "great", "awesome", "got it",
    "bye", "goodbye", "see you", "cya", "take care",
    "how are you", "what's up", "whats up", "sup",
}

_CLASSIFY_PROMPT = """\
You are a routing classifier for a Retrieval-Augmented Generation (RAG) system.
Your task is to determine whether the user's query should be answered using the uploaded document or using general knowledge.

Classify the query into exactly one of two intents:

- "needs_retrieval"
- "general_knowledge"

Return ONLY valid JSON:

{"intent":"needs_retrieval"}

or

{"intent":"general_knowledge"}

Decision Rules

Choose "general_knowledge" ONLY if the query is clearly independent of the uploaded document and is better answered directly from general knowledge.

Examples:
- What is 2 + 2?
- Capital of France?
- Who is the current President of the United States?
- What time is it in Tokyo?
- Translate "hello" to Spanish.
- Today's weather in Delhi.
- Current Bitcoin price.

For ALL other informational queries, prefer "needs_retrieval".

This includes:
- Definitions
- Explanations
- Summaries
- Comparisons
- Technical questions
- Questions about concepts
- Questions that could reasonably be answered from an uploaded document
- Questions referring to "this", "the document", "it", or previous context

NOTE: When in doubt, choose "needs_retrieval".

Do NOT answer the question.
Do NOT explain your reasoning.
Output ONLY valid JSON.
"""


def _is_chitchat(query: str) -> bool:
    normalized = query.strip().lower().rstrip("!.? ")
    if not normalized:
        return True
    return normalized in _CHITCHAT_PHRASES


def _is_ambiguous(query: str) -> bool:
    ambiguous_signals = {"it", "this", "that", "they", "he", "she", "the thing", "stuff"}
    words = set(query.lower().split())
    return bool(words & ambiguous_signals)

def _classify_intent(
    query: str,
    openai_api_key: str,
    groq_api_key: str | None,
) -> str:
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

        if intent not in {
            "needs_retrieval",
            "general_knowledge",
            "chitchat",
        }:
            return "needs_retrieval"

        return intent

    except Exception as e:
        print(
            f"[planner] intent classification failed ({e}), "
            "defaulting to needs_retrieval"
        )
        return "needs_retrieval"



def pre_retrieval_planner_node(state: GraphState, config: RunnableConfig) -> dict:
    print("[flow] entering pre_retrieval_planner_node")
    query = state.query.strip()

    if not state.file_id:
        return {"intent": "llm", "rewrite_type": "none"}

    if _is_chitchat(query):
        print("[planner] heuristic match: chitchat, skipping retrieval")
        return {"intent": "llm", "rewrite_type": "none"}

    if _is_ambiguous(query.lower()):
        return {"intent": "rag", "rewrite_type": "single"}


    openai_api_key, groq_api_key = extract_keys(config)

    intent = _classify_intent(
        query=query,
        openai_api_key=openai_api_key,
        groq_api_key=groq_api_key,
    )

    print(f"[planner] classified intent: {intent}")

    if intent in {"general_knowledge", "chitchat"}:
        return {
            "intent": "llm",
            "rewrite_type": "none",
        }

    return {
        "intent": "rag",
        "rewrite_type": "none",
    }