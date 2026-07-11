import json
from app.schemas.state import GraphState
from app.service.LLMProviders import generate_completion
from app.config.models import REWRITE_MODEL, REWRITE_PROVIDER
from app.agent.graph.keys import extract_keys
from langchain_core.runnables import RunnableConfig

def multi_query_rewrite_node(state: GraphState, config: RunnableConfig) -> dict:
    print("[flow] entering multi_query_rewrite_node")
    query = state.query

    SYSTEM_PROMPT = """
You are a query rewriting assistant for a retrieval system.

Generate 3 alternative queries that improve document retrieval.

Guidelines:
- Keep queries semantically similar to the original
- Make them more specific and clear
- Focus on improving search relevance
- Do NOT answer the query

Return ONLY valid JSON in this exact shape:
{
  "queries": ["query 1", "query 2", "query 3"]
}
"""

    openai_api_key, groq_api_key = extract_keys(config)

    response_text = generate_completion(
        provider=REWRITE_PROVIDER,
        model=REWRITE_MODEL,
        openai_api_key=openai_api_key,
        groq_api_key=groq_api_key,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.5,
    )
    try:
        data = json.loads(response_text)
        queries = data.get("queries", [])
        if not isinstance(queries, list) or not queries:
            queries = [query]
    except Exception:
        queries = [query]

    return {"queries": queries, "rewrite_type": "multi"}
