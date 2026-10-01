"""
Local-only wiring check for eval_runner.py and confidence_threshold.py, using
the mock client instead of a real API key. Zero cost. Proves the mechanics run
end-to-end (retrieval, BM25, re-ranking, metric computation, file writing) --
NOT that the numbers are meaningful, since the mock's embeddings are random.
NOT part of the graded app.

Run: python scripts/test_eval_tools_local.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from mock_client import MockOpenAI

from app.embedding import Retriever, load_index
from app.rag import RAG
from app.utils import CostTracker

print("Loading real FAISS index...")
index, chunks = load_index()

client = MockOpenAI()
tracker = CostTracker(path=Path("/tmp/mock_cost.json"))

print("Building RAG (mock client, real BM25 + real local re-ranker)...")
rag = RAG(client, Retriever(client, index, chunks, tracker), tracker)

questions = json.loads(Path("data/questions.json").read_text())[:6]  # small slice, this is a wiring check

print("\n--- eval_runner: run_retrieval_only ---")
import eval_runner
rows, summary = eval_runner.run_retrieval_only(rag, questions)
print(f"rows={len(rows)} summary_keys={list(summary.keys())}")
assert rows, "expected at least one row"
assert "recall@1" in summary and "mrr" in summary
print("OK -- retrieval-only mechanics work")

print("\n--- eval_runner: run_ablation (3 configs) ---")
ablation = eval_runner.run_ablation(rag, questions)
assert set(ablation.keys()) == {"Vector only", "+ reranking", "+ BM25 (hybrid)"}
print("OK -- ablation mechanics work, table:")
print(eval_runner.ablation_table_md(ablation))

print("\n--- confidence_threshold ---")
import confidence_threshold as ct
crows = []
for i, q in enumerate(questions):
    score = ct.top_rerank_score(rag, q["question"])
    crows.append({"id": i, "should_refuse": ct.should_refuse(q), "top_score": score})
t = ct.pick_threshold(crows)
report = ct.evaluate_threshold(crows, t)
print(f"threshold={t:.3f} report={report}")
assert "correct_refusal_rate" in report or report["refuse_group_size"] == 0
print("OK -- confidence threshold mechanics work")

print("\nAll three scripts run end-to-end with zero API cost.")
print("Scores are meaningless (mock embeddings are random) -- this only proves the wiring.")
