import hashlib

# create a unique cache key for the every text
def _embedding_cache_key(text: str) -> str:
    return "embedding:" + hashlib.sha256(text.encode()).hexdigest()