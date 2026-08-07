from langchain_core.runnables import RunnableConfig

from app.agent.graph.keys import extract_tavily_key
from app.agent.tools.tavily import search_web
from app.schemas.state import GraphState


async def web_search_node(state: GraphState, config: RunnableConfig) -> dict:
    print("[flow] entering web_search_node")

    query = state.rewritten_query or state.query
    tavily_api_key = extract_tavily_key(config)

    results = await search_web(query=query, api_key=tavily_api_key, max_results=5)

    context = []
    for result in results:
        title = result.get("title", "")
        content = result.get("content") or result.get("raw_content") or result.get("snippet") or ""
        url = result.get("url", "")
        text_parts = [part for part in [title, content, f"Source: {url}" if url else ""] if part]
        if not text_parts:
            continue
        context.append(
            {
                "text": "\n".join(text_parts),
                "title": title,
                "url": url,
                "source": "web",
                "score": result.get("score", 0),
            }
        )

    return {
        "context": context,
        "confidence": 0.65 if context else 0.0,
    }