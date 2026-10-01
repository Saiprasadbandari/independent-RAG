"""
Confidence threshold — RAG day 2, item 1 (mandatory).

For every question, log the re-ranker's top score (the cross-encoder score,
since it produces the final ranking -- FAISS and BM25 scores live on different
scales and can't be compared with it). Using two groups of questions --
questions that SHOULD be refused (out-of-scope, ambiguous, on_topic_unanswerable)
and questions that SHOULD be answered (factual, cross-reference, stress-test,
excluding on_topic_unanswerable) -- find a score threshold that separates them,
and report how many refusals are correct vs. wrong at that threshold.

Usage:
    python -m scripts.confidence_threshold

Writes data/confidence_threshold.json and prints the report.
"""

import json
from pathlib import Path

from app.utils import DATA_DIR

QUESTIONS_PATH = DATA_DIR / "questions.json"
OUT_PATH = DATA_DIR / "confidence_threshold.json"
SHOULD_REFUSE_CATEGORIES = {"out-of-scope", "ambiguous"}


def should_refuse(question: dict) -> bool:
    if question["category"] in SHOULD_REFUSE_CATEGORIES:
        return True
    return question.get("sub_type") == "on_topic_unanswerable"


def top_rerank_score(rag, question_text: str) -> float:
    """The re-ranker's top score for a question -- 0.0 if nothing was even
    worth re-ranking (no candidate cleared the pre-rerank similarity filter)."""
    from app.rag import relevant_chunks

    candidates = rag.retrieve(question_text)
    plausible = relevant_chunks(candidates)
    if not plausible or not rag.reranker:
        return 0.0
    ranked = rag.rerank(question_text, plausible)
    return max((c["relevance"] for c in ranked), default=0.0)


def evaluate_threshold(rows: list[dict], threshold: float) -> dict:
    refuse_group = [r for r in rows if r["should_refuse"]]
    answer_group = [r for r in rows if not r["should_refuse"]]
    correct_refusals = sum(1 for r in refuse_group if r["top_score"] < threshold)
    wrong_refusals = sum(1 for r in answer_group if r["top_score"] < threshold)
    return {
        "threshold": threshold,
        "correct_refusals": correct_refusals,
        "refuse_group_size": len(refuse_group),
        "correct_refusal_rate": correct_refusals / len(refuse_group) if refuse_group else None,
        "wrong_refusals": wrong_refusals,
        "answer_group_size": len(answer_group),
        "false_refusal_rate": wrong_refusals / len(answer_group) if answer_group else None,
    }


def pick_threshold(rows: list[dict]) -> float:
    """Midpoint between the highest score in the 'should refuse' group and the
    lowest score in the 'should answer' group -- the simplest separating cut."""
    refuse_scores = [r["top_score"] for r in rows if r["should_refuse"]]
    answer_scores = [r["top_score"] for r in rows if not r["should_refuse"]]
    if not refuse_scores or not answer_scores:
        return 0.1
    return (max(refuse_scores) + min(answer_scores)) / 2


def main():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    from scripts.eval_runner import build_rag
    rag = build_rag()

    rows = []
    for i, q in enumerate(questions):
        score = top_rerank_score(rag, q["question"])
        rows.append({"id": i, "question": q["question"], "category": q["category"],
                     "sub_type": q.get("sub_type"), "should_refuse": should_refuse(q),
                     "top_score": score})
        print(f"[{i + 1}/{len(questions)}] top_score={score:.3f} refuse={rows[-1]['should_refuse']:<5} "
              f"{q['question'][:60]}")

    threshold = pick_threshold(rows)
    report = evaluate_threshold(rows, threshold)
    report["questions"] = rows

    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nchosen threshold: {threshold:.3f}")
    print(f"correct refusals: {report['correct_refusals']}/{report['refuse_group_size']} "
          f"({report['correct_refusal_rate']:.0%})" if report['refuse_group_size'] else "no refuse-group questions")
    print(f"false refusals:   {report['wrong_refusals']}/{report['answer_group_size']} "
          f"({report['false_refusal_rate']:.0%})" if report['answer_group_size'] else "no answer-group questions")
    print(f"\nwritten: {OUT_PATH}")


if __name__ == "__main__":
    main()
