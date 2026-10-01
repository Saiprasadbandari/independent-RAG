"""
Cache-only evaluation: reuses real embeddings and re-rank scores already
computed and paid for in earlier sessions (data/cache/retrieval.json,
data/cache/rerank.json). Makes ZERO new API calls -- any question not
already cached is skipped entirely, never triggers a live call.

This is a deliberately reduced-scope run (the team was asked to stop
spending). It reports real recall@k/MRR/confidence-threshold numbers on
whichever questions happen to already be cached from real past use, with
the gap documented honestly rather than filled with fake/mock numbers.

Usage:
    python -m scripts.cache_only_eval

Writes data/cache_only_eval.json.
"""

import json
from pathlib import Path
from statistics import mean

from app.utils import DATA_DIR, CACHE_DIR, normalize_query

QUESTIONS_PATH = DATA_DIR / "questions.json"
OUT_PATH = DATA_DIR / "cache_only_eval.json"
ANSWERABLE = {"factual", "cross-reference", "stress-test"}
SHOULD_REFUSE_CATEGORIES = {"out-of-scope", "ambiguous"}
KS = (1, 3, 5)


def page_key(chunk: dict) -> tuple[str, int]:
    return chunk["source"], chunk["page"]


def expected_pages(question: dict) -> set[tuple[str, int]]:
    return {page_key(c) for c in question.get("expected_chunks") or []}


def should_refuse(question: dict) -> bool:
    if question["category"] in SHOULD_REFUSE_CATEGORIES:
        return True
    return question.get("sub_type") == "on_topic_unanswerable"


def load_cache(name: str) -> dict:
    path = CACHE_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("entries", {})


def main():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    retrieval_cache = load_cache("retrieval.json")
    rerank_cache = load_cache("rerank.json")

    # need the actual chunk text (source/page) to turn cached (id, score) pairs
    # back into real chunk dicts -- load chunks.json, zero API cost, local only
    chunks = json.loads((DATA_DIR / "chunks.json").read_text(encoding="utf-8"))
    chunks_by_id = {c["id"]: c for c in chunks}

    rows = []
    skipped = 0
    for i, q in enumerate(questions):
        key = normalize_query(q["question"])
        hit = retrieval_cache.get(key)
        if hit is None:
            skipped += 1
            continue  # NOT cached -- skip, never call the API

        candidates = [{**chunks_by_id[cid], "score": s} for cid, s in hit]
        row = {"id": i, "question": q["question"], "category": q["category"],
              "sub_type": q.get("sub_type"), "should_refuse": should_refuse(q)}

        judged = rerank_cache.get(key)
        if judged:
            ranked = sorted(
                [{**c, "relevance": judged.get(str(c["id"]), 0.0)} for c in candidates],
                key=lambda c: c["relevance"], reverse=True,
            )
            row["top_rerank_score"] = max((c["relevance"] for c in ranked), default=0.0)
        else:
            ranked = candidates

        expected = expected_pages(q)
        if q["category"] in ANSWERABLE and expected:
            keys_in_order = [page_key(c) for c in ranked]
            first = next((r for r, k in enumerate(keys_in_order, 1) if k in expected), None)
            row["mrr"] = 1 / first if first else 0.0
            for k in KS:
                row[f"recall@{k}"] = len(expected & set(keys_in_order[:k])) / len(expected)

        rows.append(row)

    answerable_rows = [r for r in rows if "mrr" in r]
    summary = {
        "total_questions": len(questions),
        "cached_and_evaluated": len(rows),
        "skipped_not_cached": skipped,
        "new_api_calls_made": 0,
        "recall": {f"recall@{k}": mean(r[f"recall@{k}"] for r in answerable_rows) if answerable_rows else None
                  for k in KS},
        "mrr": mean(r["mrr"] for r in answerable_rows) if answerable_rows else None,
    }

    refuse_scores = [r["top_rerank_score"] for r in rows if r.get("should_refuse") and "top_rerank_score" in r]
    answer_scores = [r["top_rerank_score"] for r in rows if not r.get("should_refuse") and "top_rerank_score" in r]
    if refuse_scores and answer_scores:
        threshold = (max(refuse_scores) + min(answer_scores)) / 2
        summary["confidence_threshold"] = {
            "threshold": threshold,
            "refuse_group_n": len(refuse_scores),
            "answer_group_n": len(answer_scores),
        }
    else:
        summary["confidence_threshold"] = None
        summary["confidence_threshold_note"] = (
            f"Not enough cached refuse-group questions ({len(refuse_scores)}) or "
            f"answer-group questions ({len(answer_scores)}) to pick a threshold yet."
        )

    OUT_PATH.write_text(json.dumps({"summary": summary, "questions": rows}, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    print(f"Evaluated {summary['cached_and_evaluated']}/{summary['total_questions']} questions "
          f"using ONLY already-cached data (0 new API calls).")
    print(f"Skipped {skipped} questions -- not cached, would have required a live call.")
    if answerable_rows:
        print(f"recall@1={summary['recall']['recall@1']:.0%} recall@5={summary['recall']['recall@5']:.0%} "
              f"mrr={summary['mrr']:.3f}  (on {len(answerable_rows)} answerable cached questions)")
    print(f"written: {OUT_PATH}")


if __name__ == "__main__":
    main()
