"""
Eval runner — Assignment 2, tool 3 (mandatory, RAG day 2).

Two modes:
  retrieval  -- search + rerank only, no answer generated. Cheap: one embedding
                call per question, no chat calls. Used for recall@k/MRR/latency
                and the ablation table.
  full       -- the complete pipeline (answers generated too), for correctness.
                Delegates to scripts/evaluate.py, which already does this well.

Also runs the ablation table: vector-only -> +reranking -> +BM25, each
retrieval-only, bypassing the disk cache so every row is measured fresh
(reusing RAG's cache across different ablation configs would silently mix
results from different pipelines under the same cache key).

Usage:
    python -m scripts.eval_runner --mode retrieval
    python -m scripts.eval_runner --mode full
    python -m scripts.eval_runner --ablation

Writes data/eval_results.json (this is the spec-required path; --mode full
also keeps writing scripts/evaluate.py's own richer data/eval/report.md).
"""

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from app.hybrid import reciprocal_rank_fusion
from app.utils import DATA_DIR

QUESTIONS_PATH = DATA_DIR / "questions.json"
OUT_PATH = DATA_DIR / "eval_results.json"
ANSWERABLE = {"factual", "cross-reference", "stress-test"}
KS = (1, 3, 5)
VECTOR_K = 15


def page_key(chunk: dict) -> tuple[str, int]:
    return chunk["source"], chunk["page"]


def expected_pages(question: dict) -> set[tuple[str, int]]:
    return {page_key(c) for c in question.get("expected_chunks") or []}


def ranking_metrics(ranked: list[dict], expected: set) -> dict[str, float]:
    keys = [page_key(c) for c in ranked]
    first = next((rank for rank, key in enumerate(keys, 1) if key in expected), None)
    metrics = {"mrr": 1 / first if first else 0.0}
    for k in KS:
        metrics[f"recall@{k}"] = len(expected & set(keys[:k])) / len(expected)
    return metrics


def build_rag():
    from dotenv import load_dotenv
    from openai import OpenAI

    from app.embedding import Retriever, load_index
    from app.rag import RAG
    from app.utils import CostTracker

    load_dotenv()
    client = OpenAI()
    tracker = CostTracker()
    index, chunks = load_index()
    return RAG(client, Retriever(client, index, chunks, tracker), tracker)


def retrieve_fresh(rag, query: str, use_bm25: bool, use_rerank: bool) -> list[dict]:
    """Retrieval for one question, bypassing every disk cache -- always live."""
    vector_hits = rag.retriever.search(query, k=VECTOR_K)
    candidates = vector_hits
    if use_bm25 and rag.bm25:
        keyword_hits = rag.bm25.search(query, k=VECTOR_K)
        candidates = reciprocal_rank_fusion(vector_hits, keyword_hits)
        for c in candidates:
            c.setdefault("score", 0.30)

    if use_rerank and rag.reranker:
        relevance = rag.reranker.score(query, candidates)
        for c, r in zip(candidates, relevance):
            c["relevance"] = r
        return sorted(candidates, key=lambda c: c["relevance"], reverse=True)
    return sorted(candidates, key=lambda c: c["score"], reverse=True)


def run_retrieval_only(rag, questions: list[dict], use_bm25: bool = True,
                       use_rerank: bool = True) -> tuple[list[dict], dict]:
    rows = []
    for i, q in enumerate(questions):
        if q["category"] not in ANSWERABLE or not q.get("expected_chunks"):
            continue
        expected = expected_pages(q)
        t0 = time.time()
        ranked = retrieve_fresh(rag, q["question"], use_bm25, use_rerank)
        latency = time.time() - t0
        rows.append({"id": i, "question": q["question"], "category": q["category"],
                     "sub_type": q.get("sub_type"), "latency": latency,
                     **ranking_metrics(ranked, expected)})

    summary = {
        "questions": len(rows),
        "avg_latency": mean(r["latency"] for r in rows) if rows else 0.0,
        "mrr": mean(r["mrr"] for r in rows) if rows else 0.0,
        **{f"recall@{k}": mean(r[f"recall@{k}"] for r in rows) if rows else 0.0 for k in KS},
    }
    return rows, summary


def run_ablation(rag, questions: list[dict]) -> dict:
    configs = [
        ("Vector only", False, False),
        ("+ reranking", False, True),
        ("+ BM25 (hybrid)", True, True),
    ]
    results = {}
    for name, use_bm25, use_rerank in configs:
        print(f"\nablation: {name}...")
        _, summary = run_retrieval_only(rag, questions, use_bm25, use_rerank)
        results[name] = summary
        print(f"  recall@1={summary['recall@1']:.0%} recall@5={summary['recall@5']:.0%} "
              f"mrr={summary['mrr']:.3f} latency={summary['avg_latency']:.2f}s")
    return results


def ablation_table_md(results: dict) -> str:
    rows = [[name, f"{s['recall@1']:.0%}", f"{s['recall@5']:.0%}", f"{s['mrr']:.3f}",
             f"{s['avg_latency']:.2f}s"] for name, s in results.items()]
    lines = ["| Version | Recall@1 | Recall@5 | MRR | Latency |", "|---|---|---|---|---|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Eval runner: recall@k, MRR, latency, ablation")
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    parser.add_argument("--ablation", action="store_true", help="run the vector/+rerank/+BM25 ablation table")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[:args.limit]

    rag = build_rag()
    output = {"mode": args.mode}

    if args.ablation:
        ablation = run_ablation(rag, questions)
        output["ablation"] = ablation
        print("\n" + ablation_table_md(ablation))

    if args.mode == "retrieval":
        rows, summary = run_retrieval_only(rag, questions)
        output["summary"] = summary
        output["questions"] = rows
        print(f"\nretrieval-only: {summary['questions']} questions, recall@1={summary['recall@1']:.0%}, "
              f"recall@5={summary['recall@5']:.0%}, mrr={summary['mrr']:.3f}, "
              f"avg latency={summary['avg_latency']:.2f}s")
    else:
        from scripts.evaluate import run as run_full, summarize
        full_rows = []
        for i, q in enumerate(questions):
            from scripts.evaluate import evaluate_question
            full_rows.append(evaluate_question(rag, q, i))
        full_summary = summarize(full_rows)
        output["summary"] = full_summary
        output["questions"] = full_rows
        print(f"\nfull: correct={full_summary['correct']:.0%}, cost=${full_summary['cost']:.4f}")

    OUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=list), encoding="utf-8")
    print(f"\nwritten: {OUT_PATH}")


if __name__ == "__main__":
    main()
