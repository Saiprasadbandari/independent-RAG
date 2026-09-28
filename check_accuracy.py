"""
Quick accuracy check: run every question in data/questions.json through the
live RAG pipeline and check whether retrieval found an expected page.

Run from the repo root:
    python check_accuracy.py
"""

import json
from collections import Counter

from openai import OpenAI
from dotenv import load_dotenv

from app.embedding import Retriever, load_index
from app.rag import RAG
from app.utils import CostTracker

load_dotenv()

with open("data/questions.json") as f:
    questions = json.load(f)

client = OpenAI()
tracker = CostTracker()
index, chunks = load_index()
rag = RAG(client, Retriever(client, index, chunks, tracker), tracker)


def expected_pages(q):
    ec = q.get("expected_chunks")
    return {c["page"] for c in ec} if ec else None


results = []
for q in questions:
    answer = rag.ask(q["question"])
    retrieved_pages = {c["page"] for c in answer.retrieved}
    exp = expected_pages(q)

    if q["category"] in ("out-of-scope", "ambiguous"):
        verdict = "REFUSED" if not answer.in_scope else "DID NOT REFUSE"
    elif exp is None:
        verdict = "N/A"
    else:
        verdict = "HIT" if (exp & retrieved_pages) else "MISS"

    results.append({
        "category": q["category"],
        "sub_type": q.get("sub_type"),
        "priority": q.get("priority", False),
        "question": q["question"],
        "expected_pages": sorted(exp) if exp else None,
        "retrieved_pages": sorted(retrieved_pages),
        "verdict": verdict,
        "answer": answer.text,
    })

with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("VERDICT COUNTS:", dict(Counter(r["verdict"] for r in results)))
print(f"TOTAL COST: {tracker.summary()}")

print("\n--- MISSES ---")
for r in results:
    if r["verdict"] == "MISS":
        print(f"[{r['category']}/{r['sub_type']}] {r['question']}")
        print(f"  expected: {r['expected_pages']} | retrieved: {r['retrieved_pages']}")

print("\n--- PRIORITY question results ---")
for r in results:
    if r["priority"]:
        print(f"[{r['verdict']}] {r['question']}")

print("\nFull details written to eval_results.json")
