"""
Hybrid search: BM25 (exact keyword match) + FAISS (semantic match), merged with
Reciprocal Rank Fusion (RRF).

Why: vector search finds meaning but misses exact terms ("Article 99", "Annex III") —
it has no notion of an exact string match, only semantic similarity. BM25 finds exact
keyword/term overlap but misses paraphrases. This corpus (a legal text full of article
numbers and defined terms) benefits from both: our own questions.json has an
"exact_term" stress-test category specifically testing this.

No API calls — BM25 runs entirely locally, same as re-ranking.
"""

import re

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"\w+")
RRF_K = 60  # standard RRF constant; de-emphasizes rank position differences deep in the list


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    """Built once from the same chunks as the FAISS index, kept alongside it."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query: str, k: int) -> list[dict]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{**self.chunks[i], "bm25_score": float(scores[i])} for i in ranked if scores[i] > 0]


def reciprocal_rank_fusion(*rankings: list[dict], k: int = RRF_K) -> list[dict]:
    """Merge any number of ranked chunk lists (each already best-first) into one,
    by rank position rather than raw score — FAISS cosine and BM25 scores live on
    different, incomparable scales, so RRF sidesteps that by using rank only."""
    rrf_score: dict[int, float] = {}
    by_id: dict[int, dict] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, 1):
            cid = chunk["id"]
            # merge, don't replace — a chunk found by both FAISS and BM25 must keep
            # both its "score" (cosine) and "bm25_score" fields, not lose one
            by_id[cid] = {**by_id.get(cid, {}), **chunk}
            rrf_score[cid] = rrf_score.get(cid, 0.0) + 1.0 / (k + rank)
    merged = sorted(rrf_score, key=lambda cid: rrf_score[cid], reverse=True)
    return [{**by_id[cid], "rrf_score": rrf_score[cid]} for cid in merged]
