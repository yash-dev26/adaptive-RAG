"""Lightweight Tavily web-search wrapper used for graph fallback retrieval."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from urllib import error, request


TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def _search_web_sync(query: str, api_key: str, max_results: int) -> list[dict[str, Any]]:
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_images": False,
        "include_raw_content": True,
    }

    req = request.Request(
        TAVILY_SEARCH_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(req, timeout=20) as resp:
        body = resp.read().decode("utf-8")
        data = json.loads(body)

    results = data.get("results", []) if isinstance(data, dict) else []
    return results if isinstance(results, list) else []


async def search_web(query: str, api_key: str | None = None, max_results: int = 5) -> list[dict[str, Any]]:
    resolved_api_key = api_key or os.getenv("TAVILY_API_KEY")
    if not resolved_api_key:
        print("[tavily] missing API key; skipping web search fallback")
        return []

    try:
        return await asyncio.to_thread(_search_web_sync, query, resolved_api_key, max_results)
    except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[tavily] search failed: {exc}")
        return []