import cohere

from app.config.server import config

RERANKER_MODEL = "rerank-v3.5"

_client = None


def get_reranker():
    global _client

    if _client is None:
        _client = cohere.ClientV2(
            api_key=config.get("cohere_api_key")
        )

    return _client