import json
from app.schemas.state import GraphState
from app.service.LLMProviders import generate_completion
from app.agent.graph.keys import extract_keys
from langchain_core.runnables import RunnableConfig
from app.config.models import EVALUATOR_MODEL, EVALUATOR_PROVIDER

HIGH_CONFIDENCE_THRESHOLD = 0.72
LOW_CONFIDENCE_THRESHOLD = 0.45
MAX_REWRITE_ATTEMPTS = 2

def post_retrieval_evaluator_node(state: GraphState, config: RunnableConfig) -> dict:
    print("[flow] entering post_retrieval_evaluator_node")
    query = state.query
    docs = state.context or []
    scores = state.scores or []
    attempts = state.attempts or 0

    if not docs:
        print("[evaluator] No docs retrieved → llm_fallback")
        return {"eval_action": "llm_fallback", "confidence": 0.0}
    
    top_score = scores[0] if scores else 0.0
    second_score = scores[1] if len(scores) > 1 else 0.0
    score_gap = top_score - second_score

    print(
        f"[evaluator] top={top_score:.3f} "
        f"second={second_score:.3f} "
        f"gap={score_gap:.3f}"
    )

    if top_score >= HIGH_CONFIDENCE_THRESHOLD:
        print(f"[evaluator] Strong retrieval ({top_score:.3f}) → generate")

        return {
            "eval_action": "generate",
            "confidence": top_score,
        }


    if top_score >= 0.62 and score_gap >= 0.10:
        print(
            f"[evaluator] Good top result + strong gap "
            f"({top_score:.3f}, gap={score_gap:.3f}) → generate"
        )

        return {
            "eval_action": "generate",
            "confidence": top_score,
        }
    
    if top_score < LOW_CONFIDENCE_THRESHOLD:
        print(f"[evaluator] Weak retrieval ({top_score:.3f})")

        # Avoid infinite rewrite loops
        if attempts >= MAX_REWRITE_ATTEMPTS:
            print("[evaluator] Rewrite limit reached → llm_fallback")

            return {
                "eval_action": "llm_fallback",
                "confidence": top_score,
            }
    # Pass the query + top-3 doc snippets (first 300 chars each)
    snippets = "\n\n".join(
        f"[Doc {i+1}] {doc['text'][:300]}" for i, doc in enumerate(docs[:3])
    )

    EVAL_PROMPT = f"""
You are evaluating whether retrieved documents are relevant enough to answer a user query.

User query: {query}

Retrieved document snippets:
{snippets}

Top Similarity Score:
{top_score:.3f}

Score Gap Between Top Results:
{score_gap:.3f}

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
- "llm_fallback"   → docs are completely off-topic; general knowledge is better; they must be completely off topic
"""

    openai_api_key, groq_api_key = extract_keys(config)

    response_text = generate_completion(
        provider=EVALUATOR_PROVIDER,
        model=EVALUATOR_MODEL,
        openai_api_key=openai_api_key,
        groq_api_key=groq_api_key,
        messages=[{"role": "user", "content": EVAL_PROMPT}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        result = json.loads(response_text)
        eval_action = result.get("eval_action", "generate")
        confidence = float(result.get("confidence", 0.5))
        reason = result.get("reason", "")
    except Exception:
        eval_action, confidence, reason = "generate", 0.5, "parse error"

    print(f"[evaluator] action={eval_action}  confidence={confidence:.2f}  reason={reason}")
    return {"eval_action": eval_action, "confidence": confidence}