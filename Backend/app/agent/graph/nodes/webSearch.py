from langchain_core.runnables import RunnableConfig

from app.agent.graph.keys import extract_tavily_key
from app.agent.tools.tavily import search_web
from app.schemas.state import GraphState


def _result_to_context_item(result: dict) -> dict | None:
    title = result.get("title", "")
    content = result.get("content") or result.get("raw_content") or result.get("snippet") or ""
    url = result.get("url", "")
    text_parts = [part for part in [title, content, f"Source: {url}" if url else ""] if part]
    if not text_parts:
        return None
    return {
        "text": "\n".join(text_parts),
        "title": title,
        "url": url,
        "source": "web",
        "score": result.get("score", 0),
    }


async def web_search_node(state: GraphState, config: RunnableConfig) -> dict:
    """Fallback retrieval via Tavily.

    If the search still comes back empty (network failure, no
    results for the query, key rejected), `context` stays empty and the
    centralized `generate` node falls back to a disclaimed general-knowledge
    answer instead of silently generating on nothing.
    """
    print("[flow] entering web_search_node")

    query = state.rewritten_query or state.query
    tavily_api_key = extract_tavily_key(config)

    results = await search_web(query=query, api_key=tavily_api_key, max_results=5)
    context = [item for item in (_result_to_context_item(r) for r in results) if item]

    print(f"[web_search] {len(context)} usable result(s) for query={query!r}")

    return {
        "context": context,
        "confidence": 0.65 if context else 0.0,
    }