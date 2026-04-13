"""
Post-retrieval planner (evaluator).

Runs AFTER retrieval. Inspects the actual retrieved documents and decides:
  - Are the docs good enough to generate from?
  - Should we rewrite and re-retrieve?
  - Is the query actually a general knowledge question (skip RAG entirely)?

State contract
--------------
Reads:   query, context (list[str]), scores (list[float])
Writes:  intent, rewrite_type, confidence, eval_action
"""

import json
from app.config.openaiConfig import openai_client
from app.schemas.state import GraphState


def needs_retrieval(query: str) -> bool:
    """
    Cheap heuristic: does this query even need a document lookup?
    Returns False for obvious general-knowledge / chitchat queries so we can
    skip retrieval entirely and go straight to the LLM node.
    """
    greetings = {"hi", "hello", "hey", "thanks", "thank you", "bye", "ok", "okay" }
    return query.strip().lower() not in greetings


# ---------------------------------------------------------------------------
# Pre-retrieval intent node  (entry-point — sets intent + rewrite_type)
# ---------------------------------------------------------------------------

# nodes/planner.py — no LLM call needed

GENERAL_PATTERNS = {
    "hi", "hello", "hey", "thanks", "what is", "how do i",
    "explain", "define", "what does", "who is", "when was"
}

def pre_retrieval_planner_node(state: GraphState) -> dict:
    print("[flow] entering pre_retrieval_planner_node")
    query = state.query.strip().lower()
    words = query.split()

    if not needs_retrieval(query):
        return {"intent": "llm", "rewrite_type": "none"}

    if not state.file_id:
        return {"intent": "llm", "rewrite_type": "none"}

    if len(words) <= 3:
        return {"intent": "rag", "rewrite_type": "multi"}

    if _is_ambiguous(query):
        return {"intent": "rag", "rewrite_type": "single"}

    return {"intent": "rag", "rewrite_type": "none"}


def _is_ambiguous(query: str) -> bool:
    ambiguous_signals = {"it", "this", "that", "they", "he", "she", "the thing", "stuff"}
    words = set(query.lower().split())
    return bool(words & ambiguous_signals)


def post_retrieval_evaluator_node(state: GraphState) -> dict:
    print("[flow] entering post_retrieval_evaluator_node")
    """
    Called AFTER retrieval.
    Evaluates the actual retrieved documents and decides next action:

    eval_action:
      "generate"        → docs are relevant, proceed to generate
      "rewrite_single"  → docs exist but seem off-topic; try a clarified query
      "rewrite_multi"   → docs are weak across the board; try diverse sub-queries
      "llm_fallback"    → no useful docs found; answer from general knowledge
    """
    query = state.query
    docs = state.context or []
    scores = state.scores or []

    # --- Fast path: no docs returned at all ---
    if not docs:
        print("[evaluator] No docs retrieved → llm_fallback")
        return {"eval_action": "llm_fallback", "confidence": 0.0}

    # --- Fast path: scores are clearly strong (top score > 0.82) ---
    if scores and scores[0] > 0.82:
        print(f"[evaluator] Strong top score {scores[0]:.3f} → generate")
        return {"eval_action": "generate", "confidence": scores[0]}

    # --- LLM-based evaluation on marginal cases ---
    # Pass the query + top-3 doc snippets (first 300 chars each)
    snippets = "\n\n".join(
        f"[Doc {i+1}] {doc[:300]}" for i, doc in enumerate(docs[:3])
    )
    avg_score = sum(scores[:3]) / len(scores[:3]) if scores else 0.0

    EVAL_PROMPT = f"""
You are evaluating whether retrieved documents are relevant enough to answer a user query.

User query: {query}

Retrieved document snippets:
{snippets}

Average retrieval score: {avg_score:.3f}  (range 0-1, higher = more similar)

Decide the best action. Return ONLY valid JSON:
{{
  "eval_action": "generate" | "rewrite_single" | "rewrite_multi" | "llm_fallback",
  "confidence": 0.0 - 1.0,
  "reason": "one-line explanation"
}}

Guidelines:
- "generate"       → docs clearly address the query
- "rewrite_single" → docs exist but query may have been misunderstood; one rewrite likely helps
- "rewrite_multi"  → docs are marginally relevant; diverse sub-queries would improve coverage
- "llm_fallback"   → docs are completely off-topic; general knowledge is better
"""

    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": EVAL_PROMPT}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        result = json.loads(response.choices[0].message.content)
        eval_action = result.get("eval_action", "generate")
        confidence = float(result.get("confidence", 0.5))
        reason = result.get("reason", "")
    except Exception:
        eval_action, confidence, reason = "generate", 0.5, "parse error"

    print(f"[evaluator] action={eval_action}  confidence={confidence:.2f}  reason={reason}")
    return {"eval_action": eval_action, "confidence": confidence}