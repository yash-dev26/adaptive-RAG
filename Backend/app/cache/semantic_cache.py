import json
import uuid
import numpy as np

from app.config.redis import redis_client
from app.ingestion.embeddings import gen_embeddings


SIMILARITY_THRESHOLD = 0.80  # tune later


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_semantic_cached_response(query: str):

    query_embedding = gen_embeddings(query)

    keys = redis_client.keys("semantic:*")

    best_sim = 0
    best_response = None
    best_query = None

    for key in keys:
        raw = redis_client.get(key)
        if not raw:
            continue  

        data = json.loads(raw)

        sim = cosine_similarity(query_embedding, data["embedding"])

        print(f"[semantic debug] sim={sim:.3f} | cached='{data['query']}'")

        if sim > best_sim:
            best_sim = sim
            best_response = data["response"]
            best_query = data["query"]

    if best_sim > SIMILARITY_THRESHOLD:
        print(f"[semantic cache] HIT (best_sim={best_sim:.3f}, matched='{best_query}')")
        return best_response

    print(f"[semantic cache] MISS (best_sim={best_sim:.3f})")
    return None

def set_semantic_cache(query: str, response: str):
    embedding = gen_embeddings(query)

    key = f"semantic:{uuid.uuid4()}"

    redis_client.set(
        key,
        json.dumps({
            "query": query,
            "embedding": embedding,
            "response": response
        }),
        ex=3600
    )