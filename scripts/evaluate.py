"""Evaluate the RAG pipeline on data/questions.json and reflect on where it fails.

Run from the repo root, after the index is built:
    python -m scripts.evaluate               # every question
    python -m scripts.evaluate --limit 5     # quick run on the first 5
    python -m scripts.evaluate --fresh       # ignore cached answers (costs API calls)
    python -m scripts.evaluate --mode hybrid # same run modes as app.main

Writes data/eval/results.json (one row per question) and data/eval/report.md.

What it measures:
- retrieval: Hit@k, Recall@k and MRR against expected_chunks pages, for FAISS order and
  for the final (re-ranked) order, plus recall of the passages actually sent to the model
- generation: citation accuracy, refusal accuracy, token overlap with expected_answer
- reflection: whether each confidence band is right as often as it claims, and which
  pipeline stage (retrieval / ranking / generation) each failure comes from
"""

import argparse
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean

from app.utils import DATA_DIR

QUESTIONS_PATH = DATA_DIR / "questions.json"
OUT_DIR = DATA_DIR / "eval"
ANSWERABLE = {"factual", "cross-reference", "stress-test"}
KS = (1, 3, 8)
STOPWORDS = {"a", "an", "the", "of", "to", "in", "and", "or", "is", "are", "for", "on", "by", "with",
             "be", "as", "that", "this", "it", "its", "which", "at", "from", "shall"}


def page_key(chunk: dict) -> tuple[str, int]:
    return chunk["source"], chunk["page"]


def expected_pages(question: dict) -> set[tuple[str, int]]:
    return {page_key(c) for c in question.get("expected_chunks") or []}


def ranking_metrics(ranked: list[dict], expected: set) -> dict[str, float]:
    """Chunk-level ranking: a chunk counts as a hit when its page is an expected page."""
    keys = [page_key(c) for c in ranked]
    first = next((rank for rank, key in enumerate(keys, 1) if key in expected), None)
    metrics = {"mrr": 1 / first if first else 0.0}
    for k in KS:
        found = expected & set(keys[:k])
        metrics[f"hit@{k}"] = float(bool(found))
        metrics[f"recall@{k}"] = len(found) / len(expected)
    return metrics


def tokens(text: str) -> list[str]:
    text = re.sub(r"\[[^\]]*\]", " ", text.lower())
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in STOPWORDS]


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = tokens(prediction), tokens(reference)
    if not pred or not ref:
        return 0.0
    common = sum(min(pred.count(t), ref.count(t)) for t in set(pred))
    if not common:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def failure_stage(answerable: bool, answer, expected: set) -> str:
    """Which part of the pipeline the question was lost in."""
    if not answerable:
        return "ok" if not answer.in_scope else "answered unanswerable"
    candidates = {page_key(c) for c in answer.retrieved}
    context = {page_key(c) for c in answer.retrieved[:answer.context_size]}
    cited = {page_key(s) for s in answer.sources}
    if not expected & candidates:
        return "retrieval"
    if not expected & context:
        return "ranking"
    if not answer.in_scope or not expected & cited:
        return "generation"
    return "ok"


def evaluate_question(rag, question: dict, index: int) -> dict:
    # follow-up questions carry the earlier user questions in "history": replay them first so the
    # follow-up is asked inside the same conversation, like in the app
    history = []
    for earlier in question.get("history") or []:
        previous = rag.ask(earlier, history or None)
        history.append((previous.standalone_question or earlier, previous.text))
    answer = rag.ask(question["question"], history or None)
    answerable = question["category"] in ANSWERABLE
    expected = expected_pages(question)
    row = {
        "id": index,
        "question": question["question"],
        "category": question["category"],
        "sub_type": question.get("sub_type"),
        "priority": bool(question.get("priority")),
        "difficulty": question.get("difficulty"),
        "answerable": answerable,
        "expected_pages": sorted(expected),
        "history": question.get("history") or [],
        "rewritten_query": answer.standalone_question if answer.standalone_question != question["question"] else None,
        "answered": answer.in_scope,
        "answer": answer.text,
        "truncated": answer.truncated,
        "cited_pages": sorted({page_key(s) for s in answer.sources}),
        "confidence": answer.confidence,
        "confidence_label": answer.confidence_label,
        "confidence_notes": answer.confidence_notes,
        "cost": answer.cost,
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
        "cached": answer.cached,
        "stage": failure_stage(answerable, answer, expected),
    }
    row["correct"] = row["stage"] == "ok"

    if answerable and expected:
        faiss_order = sorted(answer.retrieved, key=lambda c: c["score"], reverse=True)
        row["retrieval_faiss"] = ranking_metrics(faiss_order, expected)
        row["retrieval_final"] = ranking_metrics(answer.retrieved, expected)
        context = {page_key(c) for c in answer.retrieved[:answer.context_size]}
        row["context_recall"] = len(expected & context) / len(expected)
        cited = {page_key(s) for s in answer.sources}
        row["citation_precision"] = len(expected & cited) / len(cited) if cited else 0.0
        row["citation_hit"] = float(bool(expected & cited))
        row["answer_f1"] = token_f1(answer.text, question.get("expected_answer") or "")
    return row


def avg(rows: list[dict], field: str, sub: str | None = None) -> float | None:
    values = [r[field][sub] if sub else r[field] for r in rows if field in r]
    return mean(values) if values else None


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def num(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    summary = {
        "questions": len(rows),
        "correct": avg(rows, "correct"),
        "refusal_accuracy": mean(not r["answered"] for r in unanswerable) if unanswerable else None,
        "false_refusal_rate": mean(not r["answered"] for r in answerable) if answerable else None,
        "context_recall": avg(answerable, "context_recall"),
        "citation_precision": avg(answerable, "citation_precision"),
        "citation_hit": avg(answerable, "citation_hit"),
        "answer_f1": avg(answerable, "answer_f1"),
        "truncated": sum(r["truncated"] for r in rows),
        "cost": sum(r["cost"] for r in rows),
        "input_tokens": avg([r for r in rows if r["input_tokens"]], "input_tokens"),
        "output_tokens": avg([r for r in rows if r["input_tokens"]], "output_tokens"),
        "retrieval": {
            order: {m: avg(answerable, f"retrieval_{order}", m)
                    for m in ["mrr"] + [f"{kind}@{k}" for kind in ("hit", "recall") for k in KS]}
            for order in ("faiss", "final")
        },
        "stages": {stage: sum(r["stage"] == stage for r in rows)
                   for stage in ("ok", "retrieval", "ranking", "generation", "answered unanswerable")},
    }

    by_label = defaultdict(list)
    for r in rows:
        by_label[r["confidence_label"]].append(r)
    summary["calibration"] = {
        label: {"n": len(group), "mean_confidence": mean(r["confidence"] for r in group),
                "accuracy": mean(r["correct"] for r in group)}
        for label, group in by_label.items() if label
    }
    return summary


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def group_rows(rows: list[dict], field: str) -> list[list]:
    groups = defaultdict(list)
    for r in rows:
        groups[r[field] or "—"].append(r)
    return [[name, len(g), pct(avg(g, "correct")), pct(avg(g, "context_recall")),
             pct(avg(g, "citation_hit")), num(avg(g, "answer_f1")), pct(avg(g, "confidence"))]
            for name, g in sorted(groups.items())]


def write_report(rows: list[dict], summary: dict, path: Path) -> None:
    s = summary
    retrieval_rows = [[metric, num(s["retrieval"]["faiss"][metric]), num(s["retrieval"]["final"][metric])]
                      for metric in s["retrieval"]["final"]]
    calibration_rows = [[label, c["n"], pct(c["mean_confidence"]), pct(c["accuracy"])]
                        for label, c in sorted(s["calibration"].items(),
                                               key=lambda kv: ["high", "medium", "low"].index(kv[0]))]
    stage_rows = [[stage, n] for stage, n in s["stages"].items()]
    group_headers = ["group", "n", "correct", "context recall", "cited expected page", "answer F1",
                     "mean confidence"]
    overconfident = [r for r in rows if r["confidence_label"] == "high" and not r["correct"]]
    priority = [[r["id"], r["question"][:70], r["stage"], pct(r["confidence"])] for r in rows if r["priority"]]
    failures = [[r["id"], r["category"], r["stage"], r["question"][:70],
                 ", ".join(f"p.{p}" for _, p in r["expected_pages"]) or "—",
                 ", ".join(f"p.{p}" for _, p in r["cited_pages"]) or "—"]
                for r in rows if not r["correct"]]

    report = f"""# RAG evaluation

{s['questions']} questions · correct {pct(s['correct'])} · cost ${s['cost']:.4f}

## Overall

{table(["metric", "value"], [
    ["correct (right page cited / correctly refused)", pct(s["correct"])],
    ["refusal accuracy (out-of-scope + ambiguous refused)", pct(s["refusal_accuracy"])],
    ["false refusals (answerable but refused)", pct(s["false_refusal_rate"])],
    ["context recall (expected pages sent to the model)", pct(s["context_recall"])],
    ["citation precision (cited pages that were expected)", pct(s["citation_precision"])],
    ["cited an expected page", pct(s["citation_hit"])],
    ["answer token F1 vs expected_answer", num(s["answer_f1"])],
    ["answers cut off at max_tokens", s["truncated"]],
    ["avg tokens per generated answer (in / out)",
     "—" if s["input_tokens"] is None else f"{s['input_tokens']:.0f} / {s['output_tokens']:.0f}"],
])}

## Retrieval quality (answerable questions)

Chunk-level ranking against `expected_chunks` pages. "FAISS" is embedding-similarity order,
"final" is after re-ranking.

{table(["metric", "FAISS", "final"], retrieval_rows)}

## Where failures happen

- **retrieval**: no expected page among the FAISS candidates
- **ranking**: an expected page was retrieved but didn't make it into the model's context
- **generation**: the right page was in context, but the answer refused or cited something else

{table(["stage", "questions"], stage_rows)}

## Confidence calibration

A trustworthy confidence score is right more often in higher bands.

{table(["band", "n", "mean confidence", "actually correct"], calibration_rows)}

High confidence but wrong: {len(overconfident)} question(s){"" if not overconfident else ": " + ", ".join(str(r["id"]) for r in overconfident)}

## By category

{table(group_headers, group_rows(rows, "category"))}

## By sub-type

{table(group_headers, group_rows([r for r in rows if r["sub_type"]], "sub_type"))}

## Priority questions

{table(["id", "question", "result", "confidence"], priority) if priority else "_None marked priority._"}

## Failures

{table(["id", "category", "stage", "question", "expected", "cited"], failures) if failures else "_None._"}
"""
    path.write_text(report, encoding="utf-8")


def build_rag(cache_dir: Path | None, mode: str = "api"):
    from dotenv import load_dotenv

    from app.embedding import Retriever, load_index
    from app.local_models import make_client
    from app.rag import RAG
    from app.utils import CostTracker

    load_dotenv()
    client = make_client(mode)
    tracker = CostTracker()
    index, chunks = load_index()
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    return RAG(client, Retriever(client, index, chunks, tracker), tracker, **kwargs)


def run(rag, questions: list[dict], out_dir: Path = OUT_DIR) -> dict:
    rows = []
    for i, question in enumerate(questions):
        row = evaluate_question(rag, question, i)
        rows.append(row)
        print(f"[{i + 1}/{len(questions)}] {row['stage']:<22} {row['confidence_label']:<6} {question['question'][:60]}")

    summary = summarize(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "questions": rows}, indent=2,
                                                     ensure_ascii=False, default=list), encoding="utf-8")
    write_report(rows, summary, out_dir / "report.md")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on data/questions.json")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--fresh", action="store_true", help="bypass cached answers (costs API calls)")
    parser.add_argument("--mode", choices=("api", "hybrid", "local"), default="api")
    args = parser.parse_args()

    if not QUESTIONS_PATH.exists():
        raise SystemExit(f"{QUESTIONS_PATH} not found — get data/ from Teams.")
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))[:args.limit]
    with tempfile.TemporaryDirectory() as tmp:
        rag = build_rag(Path(tmp) if args.fresh else None, args.mode)
        summary = run(rag, questions)

    print(f"\ncorrect {pct(summary['correct'])} · context recall {pct(summary['context_recall'])} · "
          f"refusal accuracy {pct(summary['refusal_accuracy'])} · cost ${summary['cost']:.4f}")
    print(f"report: {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
