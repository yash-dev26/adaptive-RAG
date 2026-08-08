"""Prompt for app/agent/graph/nodes/evaluator.py."""


def build_eval_prompt(query: str, snippets: str, top_score: float, second_score: float, score_gap: float) -> str:
    return f"""
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
