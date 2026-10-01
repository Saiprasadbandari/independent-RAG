"""Agent vs RAG (Agent day 2): the same questions through the plain RAG pipeline and the agent.

    uv run python -m scripts.agent_vs_rag --mode api      # or hybrid / local
    uv run python -m scripts.agent_vs_rag --mode local --limit 3

Questions: the agent eval questions ("category": "agent") plus the 5 priority questions from
data/questions.json. Each pipeline gets its own empty cache, so neither answers from the other's
work, and the agent runs with FakeSend, so nothing reaches Teams.

Per question and pipeline it records:
- model calls: chat-completion calls (query rewriting + answer for RAG, every loop step for the
  agent); embedding calls are counted separately
- latency in seconds, and cost from the token counts (0 for local models)
- correct_auto: the agent eval runner's number check (None when the expected answer has no number);
  correct_manual is left for a person to fill in and decides pass or fail

Writes data/eval/agent_vs_rag.json and data/eval/agent_vs_rag.md.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

from app.utils import DATA_DIR
from scripts.agent_eval_runner import QUESTIONS_PATH, grade_correctness

OUT_DIR = DATA_DIR / "eval"


def load_questions(path: Path) -> list[dict]:
    qs = json.loads(path.read_text(encoding="utf-8"))
    return [q for q in qs if q.get("category") == "agent"] + [q for q in qs if q.get("priority")]


class CallCounter:
    """Counts calls to one client method (e.g. client.chat.completions.create)."""

    def __init__(self, owner, name: str = "create"):
        self.count = 0
        original = getattr(owner, name)

        def counted(*args, **kwargs):
            self.count += 1
            return original(*args, **kwargs)

        setattr(owner, name, counted)


def measure(run, chat: CallCounter, embed: CallCounter, tracker) -> dict:
    chat_before, embed_before, cost_before = chat.count, embed.count, tracker.total
    start = time.perf_counter()
    text, extra = run()
    return {"answer": text, "model_calls": chat.count - chat_before,
            "embedding_calls": embed.count - embed_before,
            "latency_s": round(time.perf_counter() - start, 2),
            "cost": round(tracker.total - cost_before, 6), **extra}


def run(mode: str, questions: list[dict]) -> list[dict]:
    from app.agent import run_agent
    from app.embedding import Retriever, load_index
    from app.injection import FakeSend
    from app.local_models import make_client
    from app.rag import RAG
    from app.utils import CostTracker

    client, tracker = make_client(mode), CostTracker()
    chat, embed = CallCounter(client.chat.completions), CallCounter(client.embeddings)
    index, chunks = load_index()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        rag_pipeline = RAG(client, Retriever(client, index, chunks, tracker), tracker, cache_dir=Path(tmp) / "rag")
        agent_rag = RAG(client, Retriever(client, index, chunks, tracker), tracker, cache_dir=Path(tmp) / "agent")
        sends = Path(tmp) / "sends.jsonl"
        for q in questions:
            def ask_rag():
                answer = rag_pipeline.ask(q["question"])
                return answer.text, {"cited_pages": sorted({s["page"] for s in answer.sources})}

            def ask_agent():
                answer = run_agent(q["question"], agent_rag, client, tracker, send=FakeSend(log=sends))
                return answer.text, {"tools": [s.tool for s in answer.trajectory],
                                     "stopped_at_cap": answer.stopped_at_cap}

            row = {"question": q["question"], "category": q.get("category"),
                   "sub_type": q.get("sub_type"), "expected_answer": q.get("expected_answer"),
                   "rag": measure(ask_rag, chat, embed, tracker),
                   "agent": measure(ask_agent, chat, embed, tracker)}
            for side in ("rag", "agent"):
                row[side]["correct_auto"] = grade_correctness(row[side]["answer"], q)
                row[side]["correct_manual"] = None
            rows.append(row)
            print(f"RAG {row['rag']['model_calls']} calls {row['rag']['latency_s']:>6}s · "
                  f"agent {row['agent']['model_calls']} calls {row['agent']['latency_s']:>6}s "
                  f"{row['agent']['tools']} · {q['question'][:60]}", flush=True)
    return rows


def mark(value) -> str:
    return "—" if value is None else ("✅" if value else "❌")


def summary(rows: list[dict], side: str) -> dict:
    graded = [r[side]["correct_auto"] for r in rows if r[side]["correct_auto"] is not None]
    n = len(rows) or 1
    return {"correct_auto": f"{sum(graded)}/{len(graded)}",
            "avg_model_calls": round(sum(r[side]["model_calls"] for r in rows) / n, 1),
            "avg_latency_s": round(sum(r[side]["latency_s"] for r in rows) / n, 1),
            "total_cost": round(sum(r[side]["cost"] for r in rows), 5)}


def write_report(rows: list[dict], mode: str) -> None:
    lines = [f"# Agent vs RAG — {mode} mode", "",
             f"{len(rows)} questions: {sum(r['category'] == 'agent' for r in rows)} agent eval + "
             f"{sum(r['category'] != 'agent' for r in rows)} priority.", "",
             "| # | Question | RAG correct (auto) | Agent correct (auto) | RAG model calls | Agent model calls "
             "| Agent tools | RAG latency | Agent latency |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['question']} | {mark(r['rag']['correct_auto'])} | {mark(r['agent']['correct_auto'])} "
                     f"| {r['rag']['model_calls']} | {r['agent']['model_calls']} "
                     f"| {', '.join(r['agent']['tools']) or 'none'} "
                     f"| {r['rag']['latency_s']}s | {r['agent']['latency_s']}s |")
    rag, agent = summary(rows, "rag"), summary(rows, "agent")
    lines += ["", "| Pipeline | Correct (auto) | Avg model calls | Avg latency | Total cost |", "|---|---|---|---|---|",
              f"| RAG | {rag['correct_auto']} | {rag['avg_model_calls']} | {rag['avg_latency_s']}s | ${rag['total_cost']} |",
              f"| Agent | {agent['correct_auto']} | {agent['avg_model_calls']} | {agent['avg_latency_s']}s | ${agent['total_cost']} |",
              "", "## Answers", ""]
    for i, r in enumerate(rows, 1):
        lines += [f"**{i}. {r['question']}**", "", f"- expected: {r['expected_answer']}",
                  f"- RAG: {' '.join(r['rag']['answer'].split())[:400]}",
                  f"- agent: {' '.join(r['agent']['answer'].split())[:400]}", ""]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "agent_vs_rag.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "agent_vs_rag.json").write_text(json.dumps(
        {"mode": mode, "summary": {"rag": rag, "agent": agent}, "questions": rows},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines[4:lines.index("## Answers") - 1]))
    print(f"report: {OUT_DIR / 'agent_vs_rag.md'}")


def main():
    parser = argparse.ArgumentParser(description="Same questions through the RAG pipeline and the agent")
    parser.add_argument("--mode", choices=["api", "hybrid", "local"], default="api")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv()
    questions = load_questions(args.questions)[:args.limit]
    if not questions:
        raise SystemExit("No agent or priority questions found.")
    write_report(run(args.mode, questions), args.mode)


if __name__ == "__main__":
    main()
