"""
One-shot, minimal-cost run to finish RAG day 2's remaining real numbers:
ablation table (12-question sample, 3 configs = 36 calls) + confidence
threshold, reusing the ablation's own "+ BM25 (hybrid)" top scores as the
threshold's answer-group instead of a separate pass (+6 calls for the
refuse-group questions only). Total: exactly 42 API calls.

Writes data/day2_final.json. Not part of the graded app -- a one-time
report generator.
"""

import json
import sys
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from app.embedding import Retriever, load_index
from app.rag import RAG
from app.utils import CostTracker, DATA_DIR
from dotenv import load_dotenv
from openai import OpenAI

import eval_runner as er
import confidence_threshold as ct

load_dotenv()
client = OpenAI()
tracker = CostTracker()
index, chunks = load_index()
rag = RAG(client, Retriever(client, index, chunks, tracker), tracker)

questions = json.loads((DATA_DIR / "questions.json").read_text())
sample_12 = questions[:12]
refuse_group = [q for q in questions if ct.should_refuse(q)]

start_cost = tracker.total
CONFIGS = [("Vector only", False, False), ("+ reranking", False, True), ("+ BM25 (hybrid)", True, True)]

ablation = {}
hybrid_top_scores = []  # captured once, from the "+ BM25 (hybrid)" pass -- reused for confidence threshold

print(f"=== Ablation on {len(sample_12)} questions, 3 configs (36 calls) ===")
for name, use_bm25, use_rerank in CONFIGS:
    rows = []
    for i, q in enumerate(sample_12):
        expected = er.expected_pages(q)
        ranked = er.retrieve_fresh(rag, q["question"], use_bm25, use_rerank)
        rows.append({"id": i, **er.ranking_metrics(ranked, expected)})
        if name == "+ BM25 (hybrid)":
            hybrid_top_scores.append(max((c.get("relevance", c.get("score", 0)) for c in ranked), default=0.0))
    summary = {"recall@1": mean(r["recall@1"] for r in rows), "recall@5": mean(r["recall@5"] for r in rows),
              "mrr": mean(r["mrr"] for r in rows)}
    ablation[name] = summary
    print(f"  {name}: recall@1={summary['recall@1']:.0%} recall@5={summary['recall@5']:.0%} mrr={summary['mrr']:.3f}")

print(f"\n=== Confidence threshold: {len(refuse_group)} refuse-group questions (6 new calls) ===")
refuse_scores = []
for q in refuse_group:
    score = ct.top_rerank_score(rag, q["question"])
    refuse_scores.append(score)
    print(f"  refuse_score={score:.3f}  {q['question'][:60]}")

threshold = (max(refuse_scores) + min(hybrid_top_scores)) / 2 if refuse_scores and hybrid_top_scores else None

cost_used = tracker.total - start_cost
print(f"\nTotal cost this run: ${cost_used:.6f}")
if threshold is not None:
    print(f"threshold: {threshold:.3f}")

out = {
    "ablation": ablation,
    "confidence_threshold": threshold,
    "refuse_scores": refuse_scores,
    "hybrid_answer_scores": hybrid_top_scores,
    "cost_this_run": cost_used,
}
(DATA_DIR / "day2_final.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("written: data/day2_final.json")
