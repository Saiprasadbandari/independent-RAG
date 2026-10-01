"""Eval runner (course Evaluation toolkit, RAG day 2): runs data/questions.json through the
pipeline and writes a scorecard.

    uv run python -m scripts.eval_runner --retrieval-only           # search + re-rank, no answers
    uv run python -m scripts.eval_runner                            # full: answers generated too
    uv run python -m scripts.eval_runner --retrieval-only --ablation # the ablation table

Options: --mode api|hybrid|local (as in app.main) · --limit N · --fresh (ignore caches) ·
--no-rerank / --no-bm25 / --no-rewrite (switch one improvement off for a single run).

Writes data/eval_results.json (summary + one row per question) and data/eval/report.md;
--ablation writes data/eval/ablation.md.

Scoring follows the toolkit: a retrieved chunk is a hit when its (source, page) is in the
question's expected_chunks. Recall@k (k = 1, 3, 5) is the share of questions with a hit in the
top k after re-ranking, MRR the average of 1 / rank of the first hit (0 when none), both only
on questions with expected_chunks. Out-of-scope and on_topic_unanswerable questions are scored
on whether the system refused. Retrieval-only mode only embeds the questions (plus one rewrite
call per follow-up), so in api mode it costs next to nothing.
"""

import argparse
import json
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from app.utils import DATA_DIR

QUESTIONS_PATH = DATA_DIR / "questions.json"
RESULTS_PATH = DATA_DIR / "eval_results.json"
REPORT_PATH = DATA_DIR / "eval" / "report.md"
ABLATION_PATH = DATA_DIR / "eval" / "ablation.md"
KS = (1, 3, 5)
# candidate refusal thresholds on the top re-ranker score
THRESHOLDS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
STOPWORDS = {"a", "an", "the", "of", "to", "in", "and", "or", "is", "are", "for", "on", "by", "with",
             "be", "as", "that", "this", "it", "its", "which", "at", "from", "shall"}


# ---------------------------------------------------------------------------- scoring helpers

def page_key(chunk: dict) -> tuple[str, int]:
    return chunk["source"], chunk["page"]


def expected_pages(question: dict) -> set[tuple[str, int]]:
    return {page_key(c) for c in question.get("expected_chunks") or []}


def should_refuse(question: dict) -> bool:
    return question["category"] == "out-of-scope" or question.get("sub_type") == "on_topic_unanswerable"


def first_hit_rank(ranked: list[dict], expected: set) -> int | None:
    return next((rank for rank, c in enumerate(ranked, 1) if page_key(c) in expected), None)


def tokens(text: str) -> list[str]:
    text = re.sub(r"\[[^\]]*\]", " ", text.lower())
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in STOPWORDS]


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = tokens(prediction), tokens(reference)
    common = sum(min(pred.count(t), ref.count(t)) for t in set(pred)) if pred and ref else 0
    if not common:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def compact(chunk: dict) -> dict:
    out = {"source": chunk["source"], "page": chunk["page"], "score": round(chunk["score"], 4)}
    if "relevance" in chunk:
        out["relevance"] = round(chunk["relevance"], 4)
    if chunk.get("keyword_only"):
        out["keyword_only"] = True
    if chunk.get("reference"):
        out["reference"] = chunk["reference"]
    return out


# ---------------------------------------------------------------------------- running questions

def run_question(rag, question: dict, index: int, full: bool) -> dict:
    """One question through retrieval (and, in full mode, generation), with latency."""
    from app.rag import sanitize_question

    started = time.perf_counter()
    if full:
        # replay the earlier turns of a follow-up conversation, with real answers
        history = []
        for earlier in question.get("history") or []:
            previous = rag.ask(earlier, history or None)
            history.append((previous.standalone_question or earlier, previous.text))
        started = time.perf_counter()  # latency of the question itself, not its history
        answer = rag.ask(question["question"], history or None)
        standalone = answer.standalone_question
    else:
        # retrieval only: earlier questions are enough context to rewrite a follow-up
        history = [(earlier, "") for earlier in question.get("history") or []]
        asked = sanitize_question(question["question"])
        standalone = rag.rewrite(asked, history)
        answer = None
    candidates, ranked, context = rag.search(standalone)
    latency = time.perf_counter() - started

    expected = expected_pages(question)
    rank = first_hit_rank(ranked, expected) if expected else None
    refused = (not answer.in_scope) if full else not context
    row = {
        "id": index,
        "question": question["question"],
        "category": question["category"],
        "sub_type": question.get("sub_type"),
        "difficulty": question.get("difficulty"),
        "priority": bool(question.get("priority")),
        "history": question.get("history") or [],
        "rewritten_query": standalone if standalone != question["question"].strip() else None,
        "expected_pages": sorted(expected),
        "retrieved": [compact(c) for c in ranked[:10]],
        "first_expected_rank": rank,
        "in_context": bool(expected & {page_key(c) for c in context}) if expected else None,
        "top_rerank_score": round(ranked[0]["relevance"], 4) if ranked and "relevance" in ranked[0] else 0.0,
        "refused": refused,
        "latency_s": round(latency, 3),
    }

    if expected:
        row["stage"] = ("ok" if rank and rank <= max(KS) else
                        "retrieval" if not expected & {page_key(c) for c in candidates} else "ranking")
    elif should_refuse(question):
        row["stage"] = "ok" if refused else "answered unanswerable"
    else:
        row["stage"] = "ambiguous"

    if full:
        cited = {page_key(s) for s in answer.sources}
        if expected and row["stage"] == "ok":
            # retrieval found it; the answer still has to use it
            row["stage"] = "ok" if expected & cited and answer.in_scope else "generation"
        row.update({
            "answer": answer.text,
            "cited_pages": sorted(cited),
            "citation_hit": bool(expected & cited) if expected else None,
            "answer_f1": round(token_f1(answer.text, question.get("expected_answer") or ""), 3) if expected else None,
            "truncated": answer.truncated,
            "confidence": answer.confidence,
            "confidence_label": answer.confidence_label,
            "cost": answer.cost,
            "input_tokens": answer.input_tokens,
            "output_tokens": answer.output_tokens,
            "cached": answer.cached,
            # fill in by hand after reading the answer: true / false
            "correct_manual": None,
        })
    # automatic verdict (the toolkit's "correct" is a manual check, recorded separately above)
    row["correct"] = row["stage"] == "ok"
    return row


# ---------------------------------------------------------------------------- summarising

def retrieval_metrics(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["expected_pages"]]
    if not scored:
        return {f"recall@{k}": None for k in KS} | {"mrr": None, "n": 0}
    ranks = [r["first_expected_rank"] for r in scored]
    out = {f"recall@{k}": mean(1.0 if rk and rk <= k else 0.0 for rk in ranks) for k in KS}
    out["mrr"] = mean(1 / rk if rk else 0.0 for rk in ranks)
    out["n"] = len(scored)
    return out


def avg(rows: list[dict], field: str):
    values = [r[field] for r in rows if r.get(field) is not None]
    return mean(values) if values else None


def group_metrics(rows: list[dict], field: str, full: bool) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(field) or "—"].append(r)
    out = {}
    for name, group in sorted(groups.items()):
        out[name] = {"n": len(group), **retrieval_metrics(group), "auto_correct": avg(group, "correct"),
                     "latency_s": avg(group, "latency_s")}
        if full:
            out[name]["answer_f1"] = avg(group, "answer_f1")
    return out


def threshold_report(rows: list[dict]) -> dict:
    """Refusal threshold on the top re-ranker score, from both groups the course names:
    questions that should be refused vs answerable ones that shouldn't."""
    refuse = [r for r in rows if r["category"] == "out-of-scope" or r["sub_type"] == "on_topic_unanswerable"]
    answerable = [r for r in rows if r["expected_pages"]]
    table = [{"threshold": t,
              "correct_refusals": sum(r["top_rerank_score"] < t for r in refuse),
              "wrong_refusals": sum(r["top_rerank_score"] < t for r in answerable)} for t in THRESHOLDS]
    # the threshold that refuses the most of what it should while refusing the fewest answerable
    best = max(table, key=lambda row: (row["correct_refusals"] - row["wrong_refusals"], -row["threshold"]))
    return {"should_refuse": len(refuse), "answerable": len(answerable), "table": table,
            "suggested": best["threshold"]}


def summarize(rows: list[dict], full: bool) -> dict:
    refuse = [r for r in rows if r["category"] == "out-of-scope" or r["sub_type"] == "on_topic_unanswerable"]
    answerable = [r for r in rows if r["expected_pages"]]
    summary = {
        "questions": len(rows),
        "retrieval": retrieval_metrics(rows),
        "context_recall": avg(answerable, "in_context"),
        "correct": avg(rows, "correct"),
        "refusal_accuracy": avg(refuse, "refused"),
        "false_refusal_rate": avg(answerable, "refused"),
        "latency_s": avg(rows, "latency_s"),
        "stages": dict(sorted({s: sum(r["stage"] == s for r in rows) for s in {r["stage"] for r in rows}}.items())),
        "by_category": group_metrics(rows, "category", full),
        "by_difficulty": group_metrics(rows, "difficulty", full),
        "by_sub_type": group_metrics([r for r in rows if r["sub_type"]], "sub_type", full),
        "threshold": threshold_report(rows),
    }
    if full:
        generated = [r for r in rows if r.get("input_tokens")]
        summary.update({
            "citation_hit": avg(answerable, "citation_hit"),
            "answer_f1": avg(answerable, "answer_f1"),
            "truncated": sum(bool(r.get("truncated")) for r in rows),
            "cost": sum(r.get("cost") or 0 for r in rows),
            "cost_per_question": (sum(r.get("cost") or 0 for r in rows) / len(rows)) if rows else None,
            "input_tokens": avg(generated, "input_tokens"),
            "output_tokens": avg(generated, "output_tokens"),
        })
        # is the confidence score right as often as it claims? (app/reflection.py)
        bands = defaultdict(list)
        for r in rows:
            if r.get("confidence_label"):
                bands[r["confidence_label"]].append(r)
        summary["calibration"] = {label: {"n": len(g), "mean_confidence": mean(r["confidence"] for r in g),
                                          "auto_correct": mean(r["correct"] for r in g)}
                                  for label, g in bands.items()}
    return summary


# ---------------------------------------------------------------------------- report

def num(v, kind="num") -> str:
    if v is None:
        return "—"
    return {"pct": f"{v:.0%}", "num": f"{v:.2f}", "s": f"{v:.2f} s", "usd": f"${v:.5f}"}[kind]


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def breakdown(groups: dict) -> str:
    return table(["group", "n", "R@1", "R@3", "R@5", "MRR", "auto ok", "latency"],
                 [[name, g["n"], num(g["recall@1"]), num(g["recall@3"]), num(g["recall@5"]), num(g["mrr"]),
                   num(g["auto_correct"], "pct"), num(g["latency_s"], "s")] for name, g in groups.items()])


def write_report(rows: list[dict], summary: dict, config: dict, path: Path) -> None:
    s, r, th = summary, summary["retrieval"], summary["threshold"]
    mode = "full" if config["full"] else "retrieval only"
    overall = [["recall@1 / @3 / @5", f"{num(r['recall@1'])} / {num(r['recall@3'])} / {num(r['recall@5'])}"],
               ["MRR", num(r["mrr"])],
               ["expected page sent to the model", num(s["context_recall"], "pct")],
               ["refused (out-of-scope + on-topic unanswerable)", num(s["refusal_accuracy"], "pct")],
               ["wrongly refused (answerable)", num(s["false_refusal_rate"], "pct")],
               ["average latency per question", num(s["latency_s"], "s")]]
    if config["full"]:
        overall += [["cited an expected page", num(s["citation_hit"], "pct")],
                    ["answer token F1 vs expected_answer", num(s["answer_f1"])],
                    ["answers cut off at max_tokens", s["truncated"]],
                    ["cost per question", num(s["cost_per_question"], "usd")]]
    calibration = ""
    if config["full"] and s.get("calibration"):
        order = ["high", "medium", "low"]
        rows_cal = [[b, c["n"], num(c["mean_confidence"], "pct"), num(c["auto_correct"], "pct")]
                    for b, c in sorted(s["calibration"].items(), key=lambda kv: order.index(kv[0]))]
        calibration = ("## Confidence calibration\n\nA trustworthy confidence score is right more often in "
                       "higher bands.\n\n" + table(["band", "n", "mean confidence", "actually ok"], rows_cal) + "\n\n")
    failures = [[x["id"], x["category"], x["sub_type"] or "—", x["stage"], x["question"][:70],
                 ", ".join(f"p.{p}" for _, p in x["expected_pages"]) or "—", x["first_expected_rank"] or "—"]
                for x in rows if not x["correct"] and x["stage"] != "ambiguous"]
    text = f"""# Eval run ({mode}, {config['backend']} mode)

{s['questions']} questions · switches: re-rank {config['rerank']}, BM25 {config['bm25']}, query rewriting {config['rewrite']}

## Overall

Recall and MRR on the {r['n']} questions with `expected_chunks`.

{table(["metric", "value"], overall)}

Stages: {", ".join(f"{k} {v}" for k, v in s['stages'].items())}

## By category

{breakdown(s['by_category'])}

## By difficulty

{breakdown(s['by_difficulty'])}

## By sub-type

{breakdown(s['by_sub_type'])}

## Confidence threshold (top re-ranker score)

A question is refused when its best passage scores below the threshold. Should refuse:
{th['should_refuse']} (out-of-scope + on-topic unanswerable) · answerable: {th['answerable']}.

{table(["threshold", "correct refusals", "wrong refusals"],
       [[t["threshold"], f"{t['correct_refusals']}/{th['should_refuse']}", f"{t['wrong_refusals']}/{th['answerable']}"]
        for t in th['table']])}

Best trade-off on this set: **{th['suggested']}** (most correct refusals minus wrong refusals).

{calibration}## Failed questions

retrieval = no expected page among the candidates · ranking = found but not in the top {max(KS)} ·
generation = in the top {max(KS)} but the answer didn't cite it (full mode) · answered unanswerable = should have refused.

{table(["id", "category", "sub-type", "stage", "question", "expected", "rank"], failures) if failures else "_None._"}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------- entry points

def build_rag(mode: str, cache_dir: Path | None):
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


def apply_switches(rag, rerank: bool, bm25: bool, rewrite: bool, cache_dir: Path) -> None:
    """Turn improvements on or off on a loaded pipeline. Each configuration gets its own search
    and re-rank caches, so nothing is reused from another row and latencies are comparable."""
    from app.hybrid import BM25Index
    from app.utils import DiskCache

    rag.use_rerank = rerank and rag.reranker is not None
    rag.bm25 = BM25Index(rag.retriever.chunks) if bm25 else None
    rag.use_rewrite = rewrite
    name = f"rr{int(rerank)}-bm{int(bm25)}-rw{int(rewrite)}"
    rag.retrieval_cache = DiskCache(cache_dir / f"retrieval-{name}.json", "eval")
    rag.rerank_cache = DiskCache(cache_dir / f"rerank-{name}.json", "eval")


def run_all(rag, questions: list[dict], full: bool, label: str = "") -> list[dict]:
    rows = []
    for i, question in enumerate(questions):
        row = run_question(rag, question, i, full)
        rows.append(row)
        print(f"{label}[{i + 1}/{len(questions)}] {row['stage']:<22} rank {str(row['first_expected_rank'] or '—'):<3} "
              f"{row['latency_s']:>6.2f}s  {question['question'][:58]}")
    return rows


ABLATION_ROWS = [
    ("Vector only", dict(rerank=False, bm25=False, rewrite=False)),
    ("+ re-ranking", dict(rerank=True, bm25=False, rewrite=False)),
    ("+ BM25 (hybrid search)", dict(rerank=True, bm25=True, rewrite=False)),
    ("+ query rewriting", dict(rerank=True, bm25=True, rewrite=True)),
]


def run_ablation(rag, questions: list[dict], cache_dir: Path, backend: str) -> str:
    lines = []
    follow_ups = [q for q in questions if q.get("history")]
    scored = sum(1 for q in questions if q.get("expected_chunks"))
    for name, switches in ABLATION_ROWS:
        apply_switches(rag, cache_dir=cache_dir, **switches)
        rows = run_all(rag, questions, full=False, label=f"{name}: ")
        m, f = retrieval_metrics(rows), retrieval_metrics([r for r in rows if r["history"]])
        lines.append([name, num(m["recall@1"]), num(m["recall@3"]), num(m["recall@5"]), num(m["mrr"]),
                      f"{num(f['recall@5'])} / {num(f['mrr'])}" if follow_ups else "—",
                      num(avg(rows, "latency_s"), "s")])
    text = f"""# Ablation ({backend} mode, retrieval only)

Each row adds one improvement on top of the row above. Recall@k and MRR on the
{scored} questions with `expected_chunks`;
the follow-up column covers the {len(follow_ups)} questions with a `history` field
{"" if follow_ups else "(none in questions.json yet, so query rewriting can't be measured)"}.

{table(["version", "R@1", "R@3", "R@5", "MRR", "follow-ups R@5 / MRR", "latency"], lines)}

What we kept, and why: _(fill in)_
"""
    ABLATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    ABLATION_PATH.write_text(text, encoding="utf-8")
    return text


def main():
    parser = argparse.ArgumentParser(description="Run data/questions.json through the RAG pipeline")
    parser.add_argument("--mode", choices=("api", "hybrid", "local"), default="api",
                        help="where the models run, as in app.main")
    parser.add_argument("--retrieval-only", action="store_true", help="search and re-rank only, no answers")
    parser.add_argument("--ablation", action="store_true", help="the ablation table (retrieval only)")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--fresh", action="store_true", help="ignore cached searches and answers")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-bm25", action="store_true")
    parser.add_argument("--no-rewrite", action="store_true")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH,
                        help="question file (default data/questions.json), e.g. a subset to re-run")
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    if not args.questions.exists():
        raise SystemExit(f"{args.questions} not found — get data/ from Teams.")
    all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    # agent-category questions test tool use, not retrieval -- scripts/agent_eval_runner.py
    # scores those; mixing them in here would pollute recall@k/refusal metrics with scenarios
    # like "calculator only, no search needed" that aren't retrieval questions at all
    questions = [q for q in all_questions if q.get("category") != "agent"][:args.limit]
    switched = args.no_rerank or args.no_bm25 or args.no_rewrite

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rag = build_rag(args.mode, tmp if args.fresh else None)
        if args.ablation:
            print(run_ablation(rag, questions, tmp, args.mode))
            print(f"written to {ABLATION_PATH}")
            return
        if switched:
            apply_switches(rag, rerank=not args.no_rerank, bm25=not args.no_bm25,
                           rewrite=not args.no_rewrite, cache_dir=tmp)
        full = not args.retrieval_only
        rows = run_all(rag, questions, full)

    config = {"backend": args.mode, "full": full, "rerank": rag.use_rerank, "bm25": rag.bm25 is not None,
              "rewrite": rag.use_rewrite}
    summary = summarize(rows, full)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": config, "summary": summary, "questions": rows}, indent=2,
                                   ensure_ascii=False), encoding="utf-8")
    # a custom --out gets its report next to it, so test runs don't overwrite the main report
    report_path = REPORT_PATH if args.out == RESULTS_PATH else args.out.with_suffix(".md")
    write_report(rows, summary, config, report_path)
    r = summary["retrieval"]
    print(f"\nR@1 {num(r['recall@1'])} · R@3 {num(r['recall@3'])} · R@5 {num(r['recall@5'])} · MRR {num(r['mrr'])} · "
          f"auto ok {num(summary['correct'], 'pct')} · latency {num(summary['latency_s'], 's')}")
    print(f"results: {args.out} · report: {report_path}")


if __name__ == "__main__":
    main()
