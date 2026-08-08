import json
import re

from app.agent.graph.keys import extract_keys, extract_tavily_key
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

# Cheap regex fast-paths, checked before the LLM classifier.
# summarize/aggregate queries are common
# and near-unambiguous from surface form alone, so there's no reason to pay
# for an LLM call to classify them.
_SUMMARIZE_PATTERN = re.compile(
    r"\b(summar(y|ize|ise|ized|ised)|tl;?dr|overview of|gist of|"
    r"key (takeaways|points)|main points)\b",
    re.IGNORECASE,
)

_AGGREGATE_PATTERN = re.compile(
    r"\b(list (all|every)|every (single )?\w+ mentioned|enumerate|"
    r"all (the )?\w+ (in|from|mentioned)|compare .+ and .+|"
    r"advantages and disadvantages|pros and cons)\b",
    re.IGNORECASE,
)

_CLASSIFY_PROMPT = """\
You are a routing classifier for a Retrieval-Augmented Generation (RAG) system.
Your task is to determine (1) whether the user's query should be answered using
the uploaded document or using general knowledge, and if using the document,
(2) what kind of retrieval task it is.

Return ONLY valid JSON:

{"intent":"needs_retrieval","task_type":"qa"}
{"intent":"needs_retrieval","task_type":"summarize"}
{"intent":"needs_retrieval","task_type":"aggregate"}
{"intent":"general_knowledge","task_type":null}

Decision Rules

Choose "general_knowledge" ONLY if the query is clearly independent of the
uploaded document and is better answered directly from general knowledge.

Examples:
- What is 2 + 2?
- Capital of France?
- Who is the current President of the United States?
- What time is it in Tokyo?
- Translate "hello" to Spanish.
- Today's weather in Delhi.
- Current Bitcoin price.

For ALL other informational queries, use "needs_retrieval" and pick a task_type:

- "summarize": the user wants a synthesis/overview of the whole document, or a
  large section of it, not an answer grounded in a handful of specific
  passages. Examples: "summarize this", "what's the tl;dr", "give me an
  overview", "what are the key takeaways".

- "aggregate": the user wants an exhaustive or comparative sweep across the
  document — every instance of something, or a structured comparison of
  multiple concepts/entities. Examples: "list every API mentioned",
  "compare Docker and Kubernetes", "what are the pros and cons", "summarize
  advantages and disadvantages of each approach".

- "qa": everything else that's about the document — a specific fact,
  definition, explanation, or a question referring to "this", "the
  document", "it", or previous context, where a handful of relevant
  passages is enough to answer.

NOTE: When in doubt between "qa" and one of the others, choose "qa" as it's
the safer default since it still retrieves and can partially answer broader
questions, whereas the reverse that is mis-classifying a narrow question as
"summarize" wastes a full-document pass.

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


def _heuristic_task_type(query: str) -> str | None:
    """Cheap regex classification. Returns None if no confident match, in
    which case the caller falls through to the LLM classifier."""
    if _SUMMARIZE_PATTERN.search(query):
        return "summarize"
    if _AGGREGATE_PATTERN.search(query):
        return "aggregate"
    return None


async def _classify_intent(
    query: str,
    openai_api_key: str,
    groq_api_key: str | None,
) -> tuple[str, str | None]:
    try:
        response_text = await generate_completion(
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
        task_type = result.get("task_type")

        if intent not in {"needs_retrieval", "general_knowledge", "chitchat"}:
            intent = "needs_retrieval"

        if task_type not in {"qa", "summarize", "aggregate", None}:
            task_type = "qa"

        return intent, task_type

    except Exception as e:
        print(
            f"[planner] intent classification failed ({e}), "
            "defaulting to needs_retrieval/qa"
        )
        return "needs_retrieval", "qa"


async def pre_retrieval_planner_node(state: GraphState, config: RunnableConfig) -> dict:
    print("[flow] entering pre_retrieval_planner_node")
    query = state.query.strip()

    # Computed once per turn and stamped onto every return below. This is the
    # only place in the graph that reads the Tavily key (via config, never
    # persisted), so it's also the only place that can tell the evaluator's
    # router whether the web-search fallback branch is even reachable.
    tavily_configured = extract_tavily_key(config) is not None

    if not state.file_id:
        return {"intent": "llm", "rewrite_type": "none", "task_type": None, "tavily_configured": tavily_configured}

    if _is_chitchat(query):
        print("[planner] heuristic match: chitchat, skipping retrieval")
        return {"intent": "llm", "rewrite_type": "none", "task_type": None, "tavily_configured": tavily_configured}

    heuristic_task_type = _heuristic_task_type(query)
    if heuristic_task_type == "summarize":
        print("[planner] heuristic match: summarize")
        return {"intent": "rag", "rewrite_type": "none", "task_type": "summarize", "tavily_configured": tavily_configured}

    if heuristic_task_type == "aggregate":
        print("[planner] heuristic match: aggregate")
        # Aggregate queries route through the existing multi-rewrite fan-out
        # for broader coverage
        return {"intent": "rag", "rewrite_type": "multi", "task_type": "aggregate", "tavily_configured": tavily_configured}

    if _is_ambiguous(query.lower()):
        return {"intent": "rag", "rewrite_type": "single", "task_type": "qa", "tavily_configured": tavily_configured}

    openai_api_key, groq_api_key = extract_keys(config)

    intent, task_type = await _classify_intent(
        query=query,
        openai_api_key=openai_api_key,
        groq_api_key=groq_api_key,
    )

    print(f"[planner] classified intent: {intent}, task_type: {task_type}")

    if intent in {"general_knowledge", "chitchat"}:
        return {"intent": "llm", "rewrite_type": "none", "task_type": None, "tavily_configured": tavily_configured}

    # LLM classifier agreed the query is retrieval-worthy; give aggregate
    # queries the same multi-rewrite fan-out the heuristic path uses.
    return {
        "intent": "rag",
        "rewrite_type": "multi" if task_type == "aggregate" else "none",
        "task_type": task_type or "qa",
        "tavily_configured": tavily_configured,
    }