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
    # Priority for which sighting of a doc gets to supply its output
    # "score" (evaluator.py's confidence thresholds assume this is a real
    # 0-1 cosine similarity, never an RRF or sparse/BM25-scale number):
    #   0 = explicitly "dense"                    — best, use it
    #   1 = unlabeled ("default")                 — this function's own
    #       output already resolved its "score" to the dense value, so an
    #       unlabeled sighting (i.e. re-fusing an already-fused ranking,
    #       which is exactly what happens when retrieve_node fuses per-
    #       query-variant results for multi-rewrite/aggregate) is just as
    #       trustworthy as an explicit "dense" one
    #   2 = "sparse"                               — wrong scale, only used
    #       as a last resort when no dense-equivalent value ever shows up
    #       for this doc at all (e.g. a doc dense search never returned)
    _SOURCE_QUALITY = {"dense": 0, "sparse": 2}

    scores: dict = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            doc_id = hash(doc["text"])
            if doc_id not in scores:
                scores[doc_id] = {
                    "doc": doc,
                    "rrf_score": 0.0,
                    "source_scores": {},
                    "dense_score": None,
                    "dense_score_rank": None,
                }
            entry = scores[doc_id]
            entry["rrf_score"] += 1 / (k + rank + 1)

            source = doc.get("source", "default")
            if source not in entry["source_scores"]:
                entry["source_scores"][source] = doc.get("score", 0.0)

            quality = _SOURCE_QUALITY.get(source, 1)
            if entry["dense_score_rank"] is None or quality < entry["dense_score_rank"]:
                entry["dense_score"] = doc.get("score", entry["dense_score"] or 0.0)
                entry["dense_score_rank"] = quality

    ranked = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return [
        {
            "text": item["doc"]["text"],
            # preserve dense cosine score specifically — evaluator.py's
            # confidence thresholds depend on this being a real similarity,
            # not an RRF score.
            "score": item["dense_score"] if item["dense_score"] is not None else item["doc"].get("score", 0.0),
            "rrf_score": item["rrf_score"],
            "source_scores": item["source_scores"],
        }
        for item in ranked
    ]