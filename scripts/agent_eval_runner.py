"""Agent eval runner (course Evaluation toolkit, Agent day 2): runs the agent-category
questions in data/questions.json through run_agent() and writes a scorecard.

    uv run python -m scripts.agent_eval_runner              # api mode -- real tokens, estimate first
    uv run python -m scripts.agent_eval_runner --mode hybrid
    uv run python -m scripts.agent_eval_runner --mock       # zero-cost mechanics check

--mode local is refused: app/local_models.py's LocalChatCompletions doesn't implement
function/tool-calling (it just ignores the tools= argument and generates free text), so a
local-mode agent would never call a single tool. That isn't evaluating the agent, it's
evaluating whether the loop still returns a no-tool-calls answer on the first try. Use --mock
(scripts/mock_client.py, scripted responses) to check the harness mechanics for free instead.

Writes data/agent_eval_results.json and data/eval/agent_report.md.

Scoring:
- tool_choice_ok: every tool in the question's expected_tools was called at least once. A path
  that also calls something extra (e.g. a second search) still passes -- "used what it needed",
  not "used exactly and only what's listed".
- correct_auto: a heuristic, NOT authoritative. Checks whether every distinctive number in the
  question's expected_answer also appears in the agent's final answer text (normalized so
  "35,000,000", "35 000 000" and "35000000" all match). None when the expected answer has no
  number to check (e.g. the honest-refusal question) -- free-text grading is approximate the
  same way the RAG eval runner's "cited the expected page" metric is (see FAILURE_LOG.md).
  correct_manual is left for a human to fill in and should be trusted over the heuristic.
- steps: len(trajectory), the number of tool calls made.
- latency_s: wall-clock seconds for run_agent() to return.
"""

import argparse
import json
import re
import time
from pathlib import Path

from app.utils import DATA_DIR

QUESTIONS_PATH = DATA_DIR / "questions.json"
RESULTS_PATH = DATA_DIR / "agent_eval_results.json"
REPORT_PATH = DATA_DIR / "eval" / "agent_report.md"


def agent_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    qs = json.loads(path.read_text(encoding="utf-8"))
    return [q for q in qs if q.get("category") == "agent"]


def numbers_in(text: str) -> set[str]:
    """Distinctive numbers mentioned in text (>= 4 digits once separators are stripped), so
    '35,000,000', '35 000 000' and '35000000' all normalize to the same '35000000'."""
    return {re.sub(r"[,\s]", "", n) for n in re.findall(r"\d[\d,\s]{3,}\d|\d{4,}", text)}


def grade_correctness(answer_text: str, question: dict) -> bool | None:
    expected_numbers = numbers_in(question.get("expected_answer", ""))
    if not expected_numbers:
        return None
    return expected_numbers.issubset(numbers_in(answer_text))


def grade_tool_choice(trajectory: list, expected_tools: list[str] | None) -> dict:
    used = {step.tool for step in trajectory}
    expected = set(expected_tools or [])
    return {
        "used_tools": sorted(used),
        "expected_tools": sorted(expected),
        "missing": sorted(expected - used),
        "extra": sorted(used - expected),
        "tool_choice_ok": expected.issubset(used),
    }


def run_one(question: dict, rag, client, tracker) -> dict:
    from app.agent import run_agent

    # MockOpenAI's script is step-indexed (1st call -> search, 2nd -> calculator, 3rd -> answer)
    # and stateful on the instance, so it only looks realistic once per client -- give each
    # question its own client in --mock mode so the script restarts each time.
    if type(client).__name__ == "MockOpenAI":
        client = type(client)()

    start = time.perf_counter()
    answer = run_agent(question["question"], rag, client, tracker)
    latency_s = time.perf_counter() - start

    return {
        "question": question["question"],
        "scenario": question.get("scenario"),
        "answer": answer.text,
        "cost": answer.cost,
        "steps": len(answer.trajectory),
        "latency_s": round(latency_s, 2),
        "stopped_at_cap": answer.stopped_at_cap,
        "gave_up_retrying": answer.gave_up_retrying,
        **grade_tool_choice(answer.trajectory, question.get("expected_tools")),
        "correct_auto": grade_correctness(answer.text, question),
        "correct_manual": None,
        "trajectory": [{"tool": s.tool, "arguments": s.arguments, "result": s.result[:300]}
                       for s in answer.trajectory],
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    graded = [r for r in rows if r["correct_auto"] is not None]
    return {
        "n": n,
        "tool_choice_ok": sum(r["tool_choice_ok"] for r in rows) / n if n else 0.0,
        "correct_auto": sum(r["correct_auto"] for r in graded) / len(graded) if graded else None,
        "ungraded": n - len(graded),
        "avg_steps": sum(r["steps"] for r in rows) / n if n else 0.0,
        "avg_latency_s": sum(r["latency_s"] for r in rows) / n if n else 0.0,
        "stopped_at_cap": sum(r["stopped_at_cap"] for r in rows),
        "gave_up_retrying": sum(r["gave_up_retrying"] for r in rows),
        "total_cost": sum(r["cost"] for r in rows),
    }


def write_report(rows: list[dict], summary: dict, config: dict, path: Path = REPORT_PATH) -> str:
    correct_bit = (f" · auto-graded correct: {summary['correct_auto']:.0%} "
                   f"({summary['ungraded']} ungraded)" if summary["correct_auto"] is not None
                   else f" · {summary['ungraded']} ungraded (no number to check)")
    lines = [
        "# Agent eval report", "",
        f"Backend: `{config['backend']}` · {summary['n']} questions · "
        f"tool choice correct: {summary['tool_choice_ok']:.0%}{correct_bit}",
        f"Avg steps: {summary['avg_steps']:.1f} · avg latency: {summary['avg_latency_s']:.2f}s · "
        f"stopped at cap: {summary['stopped_at_cap']} · gave up retrying: {summary['gave_up_retrying']} · "
        f"total cost: ${summary['total_cost']:.5f}",
        "",
        "| # | Scenario | Tool choice | Steps | Latency | Correct (auto) | Correct (manual) |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        tool = "✅" if r["tool_choice_ok"] else f"❌ missing {r['missing']}"
        auto = "—" if r["correct_auto"] is None else ("✅" if r["correct_auto"] else "❌")
        lines.append(f"| {i} | {r['scenario']} | {tool} | {r['steps']} | {r['latency_s']}s | {auto} | |")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main():
    parser = argparse.ArgumentParser(description="Run the agent-category questions through run_agent()")
    parser.add_argument("--mode", choices=("api", "hybrid"), default="api",
                        help="local mode is refused -- it can't make tool calls, see module docstring")
    parser.add_argument("--mock", action="store_true",
                        help="scripts/mock_client.py instead of a real API key -- zero cost, "
                             "checks the harness mechanics only, not real tool-choice/correctness")
    parser.add_argument("--limit", type=int, help="only the first N agent questions")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    if not args.questions.exists():
        raise SystemExit(f"{args.questions} not found — get data/ from Teams.")
    questions = agent_questions(args.questions)[:args.limit]
    if not questions:
        raise SystemExit("No agent-category questions found.")

    from app.embedding import Retriever, load_index
    from app.rag import RAG
    from app.utils import CostTracker

    if args.mock:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from mock_client import MockOpenAI

        client = MockOpenAI()
        backend = "mock"
        tracker = CostTracker(path=Path("/tmp/mock_agent_eval_cost.json"))
    else:
        from dotenv import load_dotenv

        load_dotenv()
        from app.local_models import make_client

        client = make_client(args.mode)
        backend = args.mode
        tracker = CostTracker()

    index, chunks = load_index()
    rag = RAG(client, Retriever(client, index, chunks, tracker), tracker)

    rows = [run_one(q, rag, client, tracker) for q in questions]
    summary = summarize(rows)
    config = {"backend": backend}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": config, "summary": summary, "questions": rows}, indent=2,
                                   ensure_ascii=False), encoding="utf-8")
    write_report(rows, summary, config)
    print(f"tool choice {summary['tool_choice_ok']:.0%} · avg steps {summary['avg_steps']:.1f} · "
          f"avg latency {summary['avg_latency_s']:.2f}s · cost ${summary['total_cost']:.5f}")
    print(f"results: {args.out} · report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
