import json
from app.schemas.state import GraphState
from app.service.LLMProviders import generate_completion
from app.agent.graph.keys import extract_keys
from langchain_core.runnables import RunnableConfig
from app.config.models import EVALUATOR_MODEL, EVALUATOR_PROVIDER, MAX_REWRITE_ATTEMPTS

HIGH_CONFIDENCE_THRESHOLD = 0.70
LOW_CONFIDENCE_THRESHOLD = 0.45


async def post_retrieval_evaluator_node(state: GraphState, config: RunnableConfig) -> dict:
    print("[flow] entering post_retrieval_evaluator_node")
    query = state.query
    docs = state.context or []
    scores = state.scores or []
    attempts = state.rewrite_attempts or 0

    top_score = scores[0] if scores else 0.0
    second_score = scores[1] if len(scores) > 1 else 0.0
    score_gap = top_score - second_score

    print(
        f"[evaluator] top={top_score:.3f} second={second_score:.3f} "
        f"gap={score_gap:.3f} attempts={attempts} task_type={state.task_type}"
    )

    # Aggregate queries ("list every X", "compare A and B") are deliberately
    # broad, a low top-1 similarity score is *expected* (no single chunk will be
    # "the" answer) and doesn't indicate bad retrieval the way it does for a
    # single-fact QA query. The score-gap/rewrite logic below was designed to
    # answer "is there one good answer here", which is the wrong question for
    # this task_type. Just check we got something back and generate.
    if state.task_type == "aggregate":
        if docs:
            print(f"[evaluator] aggregate task, {len(docs)} docs retrieved → generate")
            return {"eval_action": "generate", "confidence": top_score}
        print("[evaluator] aggregate task, no docs → llm_fallback")
        return {"eval_action": "llm_fallback", "confidence": top_score}

    # Once retries are exhausted, the router will stop the rewrite loop and
    # hand control to the fallback path instead of paying for another rewrite.
    if attempts >= MAX_REWRITE_ATTEMPTS:
        if docs and top_score >= LOW_CONFIDENCE_THRESHOLD:
            print(f"[evaluator] Retries exhausted, usable docs ({top_score:.3f}) → generate")
            return {"eval_action": "generate", "confidence": top_score}
        print("[evaluator] Retries exhausted, no usable docs → llm_fallback")
        return {"eval_action": "llm_fallback", "confidence": top_score}

    if not docs:
        print("[evaluator] No docs retrieved → rewrite_single")
        return {"eval_action": "rewrite_single", "confidence": 0.0}

    if top_score >= HIGH_CONFIDENCE_THRESHOLD:
        print(f"[evaluator] Strong retrieval ({top_score:.3f}) → generate")
        return {"eval_action": "generate", "confidence": top_score}

    if top_score >= 0.60 and score_gap >= 0.10:
        print(f"[evaluator] Good top result + strong gap ({top_score:.3f}, gap={score_gap:.3f}) → generate")
        return {"eval_action": "generate", "confidence": top_score}

    if top_score < LOW_CONFIDENCE_THRESHOLD:
        print(f"[evaluator] Weak retrieval ({top_score:.3f}) → rewrite_single")
        return {"eval_action": "rewrite_single", "confidence": top_score}

    snippets = "\n\n".join(
        f"[Doc {i + 1}] {doc['text']}" for i, doc in enumerate(docs[:3])
    )

    EVAL_PROMPT = f"""
You are evaluating the quality of retrieved documents for a Retrieval-Augmented Generation (RAG) system.

Your task is NOT to answer the user's question.

Your task is ONLY to determine whether the retrieved documents are sufficient for answering the query.

User Query:
{query}

Retrieved Documents:

{snippets}

Retrieval Statistics:
- Top similarity score: {top_score:.3f}
- Second similarity score: {second_score:.3f}
- Score gap: {score_gap:.3f}

Return ONLY valid JSON.

{{
    "eval_action": "generate" | "rewrite_single" | "rewrite_multi" | "llm_fallback",
    "reason": "<short reason>"
}}

Decision Rules

1. generate

Choose this if the retrieved documents contain enough information to answer the user's question.

The documents do NOT need to be perfect.
If a reasonable answer can be produced using them, choose "generate".

2. rewrite_single

Choose this ONLY if the query itself appears vague, ambiguous, incomplete, or poorly phrased.

Examples:
- "What does it mean?"
- "Explain this."
- "How does it work?"
- "Tell me more."

3. rewrite_multi

Choose this ONLY if:

- the query contains multiple sub-questions,
- or multiple concepts,
- or the retrieved documents each cover different pieces of the query.

Examples:
- Compare Docker and Kubernetes.
- Explain OAuth, JWT and sessions.
- List every API mentioned.
- Summarize advantages and disadvantages.

4. llm_fallback

Choose this ONLY if the retrieved documents are clearly unrelated to the user's query.

Important Rules

- Prefer "generate" whenever the documents are usable.
- Do NOT choose rewrite simply because better documents may exist.
- Judge ONLY the retrieved evidence.
- Do NOT answer the user's question.
- Return ONLY JSON.
"""
    openai_api_key, groq_api_key = extract_keys(config)

    response_text = await generate_completion(
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
        reason = result.get("reason", "")
    except Exception:
        eval_action, reason = "generate", "parse error"

    print(f"[evaluator] action={eval_action}  confidence(top_score)={top_score:.2f}  reason={reason}")
    return {"eval_action": eval_action, "confidence": top_score}