import hashlib
import json
from app.config.redis import redis_client


def _response_cache_key(user_id: str, file_id: str | None, query: str) -> str:
    raw = f"{user_id}:{file_id}:{query}"
    return "response:" + hashlib.sha256(raw.encode()).hexdigest()


def get_cached_response(user_id: str, file_id: str | None, query: str):
    key = _response_cache_key(user_id, file_id, query)
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    return None


def set_cached_response(user_id: str, file_id: str | None, query: str, response: str):
    key = _response_cache_key(user_id, file_id, query)
    redis_client.set(key, json.dumps(response), ex=36000)  # Cache expires in 10 hours