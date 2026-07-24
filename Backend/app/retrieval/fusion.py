from typing import List


def reciprocal_rank_fusion(rankings: List[List[dict]], k: int = 60) -> List[dict]:
    """
    Fuses N ranked lists of {"text", "score", ...} dicts by rank position,
    not raw score magnitude. That's what lets this same function fuse:
      1. dense + sparse rankings (different scoring scales entirely), and
      2. rankings across rewritten-query variants (multi-rewrite)
    without needing to normalize scores onto a common scale first.

    k=60 is the standard RRF constant from the original paper (Cormack et al.) —
    bumped from the old k=20 used for multi-query-only fusion.
    """
    scores: dict = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            doc_id = hash(doc["text"])
            if doc_id not in scores:
                scores[doc_id] = {"doc": doc, "rrf_score": 0.0, "source_scores": {}}
            scores[doc_id]["rrf_score"] += 1 / (k + rank + 1)
            source = doc.get("source", "default")
            if source not in scores[doc_id]["source_scores"]:
                scores[doc_id]["source_scores"][source] = doc.get("score", 0.0)

    ranked = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return [
        {
            "text": item["doc"]["text"],
            # preserve dense cosine score specifically — evaluator.py's
            # confidence thresholds depend on this being a real similarity,
            # not an RRF score.
            "score": item["source_scores"].get("dense", item["doc"].get("score", 0.0)),
            "rrf_score": item["rrf_score"],
            "source_scores": item["source_scores"],
        }
        for item in ranked
    ]