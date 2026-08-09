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
    #   2 = "sparse"                               — wrong scale (raw
    #       BM25-style), and NEVER allowed to populate dense_score below.
    #       A doc that was only ever found via sparse search has no real
    #       cosine similarity to report; leaving dense_score as None (and
    #       therefore the output "score" as 0.0) is the correct signal —
    #       previously a sparse-only sighting's raw score leaked into
    #       dense_score on first sight, silently feeding a BM25-scale
    #       number into the evaluator's cosine-similarity thresholds and
    #       pushing otherwise-fine retrievals into unnecessary rewrites
    #       and web-search fallback.
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
            # quality < 2 excludes "sparse" entirely — only a dense or
            # already-resolved ("default") sighting may set dense_score.
            if quality < 2 and (entry["dense_score_rank"] is None or quality < entry["dense_score_rank"]):
                entry["dense_score"] = doc.get("score", entry["dense_score"] or 0.0)
                entry["dense_score_rank"] = quality

    ranked = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return [
        {
            "text": item["doc"]["text"],
            # dense_score is None for a doc that was only ever matched via
            # sparse search — report 0.0 rather than falling back to a
            # BM25-scale number, so the evaluator's cosine-similarity
            # thresholds only ever see a real cosine score or an honest
            # "no dense match" signal.
            "score": item["dense_score"] if item["dense_score"] is not None else 0.0,
            "rrf_score": item["rrf_score"],
            "source_scores": item["source_scores"],
        }
        for item in ranked
    ]