"""Tavily web-search client used as the graph's fallback retrieval path.

Tavily is a not a hard dependency of the graph, and
`generate_node` treats an empty context as "nothing to ground on" and
answers with a disclaimed general-knowledge reply instead.
"""

from __future__ import annotations

import httpx

from app.utils.retry import external_api_retry

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_TIMEOUT_SECONDS = 15.0


@external_api_retry(max_attempts=2, min_wait=1, max_wait=4)
async def _post_search(client: httpx.AsyncClient, payload: dict) -> dict:
    """Issue the search request. Connection-level failures (timeouts, resets)
    are retried automatically by `external_api_retry`; HTTP status errors are
    not retried here since a 4xx (bad key, bad request) won't fix itself and
    Tavily doesn't warrant burning the fixed retry budget on a fallback path.
    """
    response = await client.post(TAVILY_SEARCH_URL, json=payload)
    response.raise_for_status()
    return response.json()


async def search_web(query: str, api_key: str | None, max_results: int = 5) -> list[dict]:
    """Run a Tavily search and return the raw list of result dicts.

    Returns an empty list on any failure — see module docstring.
    """
    if not api_key:
        # Defensive only: routing should not send us here without a key.
        print("[tavily] search_web called with no API key; returning no results")
        return []

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_images": False,
        "include_raw_content": True,
    }

    try:
        async with httpx.AsyncClient(timeout=TAVILY_TIMEOUT_SECONDS) as client:
            data = await _post_search(client, payload)
    except httpx.HTTPStatusError as exc:
        print(f"[tavily] search failed with status {exc.response.status_code}")
        return []
    except httpx.HTTPError as exc:
        print(f"[tavily] request failed: {exc}")
        return []
    except ValueError as exc:
        # response.json() raised — malformed body
        print(f"[tavily] failed to parse response: {exc}")
        return []

    results = data.get("results") if isinstance(data, dict) else None
    return results if isinstance(results, list) else []