"""The Routing page: does the agent call the right tool for each kind of question?

One question per route (the rehearsal questions in scripts/DEMO_QUESTIONS.md), each with the
tools it should use. "Run the routing check" sends them through the real agent loop with the
fake send (nothing reaches Teams) and fills the table as each answer comes back; the last run is
saved to data/eval/routing.json and shown when the page opens.
"""

import json
import time
from datetime import datetime

import gradio as gr

from app.agent import run_agent
from app.injection import FakeSend
from app.utils import DATA_DIR

RESULTS_PATH = DATA_DIR / "eval" / "routing.json"
SENDS_LOG = DATA_DIR / "eval" / "routing_sends.jsonl"

# (route, question, tools it must use); an empty set means it should answer without a tool
PROBES = [
    ("One tool", "What obligations do providers of general-purpose AI models have?", {"search_corpus"}),
    ("Calculator", "What is 7% of 500000000?", {"calculator"}),
    ("Own tool: definition", "What does 'deployer' mean in the AI Act?", {"get_definition"}),
    ("Own tool: dates", "How many days are there between 2 February 2025 and 2 August 2026?", {"date_diff"}),
    ("Own tool: exact quote", "Give me the exact full text of page 123 of 'EU AI Regulations.pdf'.", {"quote_exact"}),
    ("Search, then calculate", "What is the maximum fine for a prohibited AI practice for a company with a "
                               "worldwide turnover of 1 billion euros?", {"search_corpus", "calculator"}),
    ("No tool", "Hello! What can you help me with?", set()),
    ("Out of scope", "What is the best pizza topping?", {"search_corpus"}),
    ("Act, with approval", "Send 'Team meeting at 3pm' to our team channel.", {"send_message"}),
]

ROUTING_CSS = """
#routing-table table { display: block; overflow-x: auto; font-size: 13px; }
#routing-table td, #routing-table th { word-break: keep-all !important; overflow-wrap: normal !important; }
#routing-table td:nth-child(6), #routing-table th:nth-child(6) { white-space: nowrap; }
"""


def cell(text: str) -> str:
    """Text safe inside a Markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def routed_ok(expected: set[str], used: list[str]) -> bool:
    return expected <= set(used) if expected else not used


def run_one(route: str, question: str, expected: set[str], rag) -> dict:
    start = time.perf_counter()
    try:
        answer = run_agent(question, rag, rag.client, rag.tracker, send=FakeSend(log=SENDS_LOG, label="routing"))
    except Exception as e:  # one failed question shouldn't stop the run
        return {"route": route, "question": question, "expected": sorted(expected), "calls": [],
                "ok": False, "answer": f"Error: {e}", "seconds": round(time.perf_counter() - start, 1)}
    calls = [f"{s.tool}({', '.join(f'{k}={v!r}' for k, v in s.arguments.items())})" for s in answer.trajectory]
    used = [s.tool for s in answer.trajectory]
    return {"route": route, "question": question, "expected": sorted(expected), "calls": calls,
            "ok": routed_ok(expected, used), "answer": answer.text, "seconds": round(time.perf_counter() - start, 1),
            "stopped_at_cap": answer.stopped_at_cap, "gave_up_retrying": answer.gave_up_retrying}


def render(run: dict | None, running: int | None = None) -> str:
    if not run:
        return ("_No routing run on this machine yet. Click **▶ Run the routing check**: the 9 questions go "
                "through the agent one by one and the table fills in as each answer comes back._")
    rows = run["rows"]
    ok = sum(r["ok"] for r in rows)
    status = (f"Running question {running} of {len(PROBES)}…" if running else
              f"**{ok} of {len(rows)} routed as expected** · run {run['when']} · answers by `{run['model']}`")
    lines = [status, "", "| # | Route | Question | Expected | Agent called | Time | |", "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        expected = ", ".join(f"`{t}`" for t in r["expected"]) or "_no tool_"
        called = "<br>".join(f"`{cell(c)}`" for c in r["calls"]) or "_no tool_"
        mark = "✅" if r["ok"] else "⚠️"
        lines.append(f"| {i} | {r['route']} | {cell(r['question'])} | {expected} | {called} | {r['seconds']} s | {mark} |")
    for i in range(len(rows) + 1, len(PROBES) + 1 if running else 0):
        route, question, _ = PROBES[i - 1]
        lines.append(f"| {i} | {route} | {cell(question)} | | {'⏳' if i == running else ''} | | |")
    answers = [f"**{i}. {r['route']}**: {' '.join(r['answer'].split())[:400]}" for i, r in enumerate(rows, 1)]
    return "\n".join(lines) + ("\n\n#### Answers\n\n" + "\n\n".join(answers) if answers else "")


def last_run() -> dict | None:
    try:
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def build_routing_page(rag) -> None:
    gr.Markdown("# Routing\nDoes the agent pick the right tool? One question per route, each with the tools it "
                "should call. The send question goes to a fake send, so nothing reaches Teams. Same questions as "
                "`scripts/DEMO_QUESTIONS.md`; results of earlier runs are in `scripts/TESTING_LOG.md` (tests 31 and 43).")
    run_btn = gr.Button("▶ Run the routing check", variant="primary", size="sm")
    table = gr.Markdown(lambda: render(last_run()), elem_id="routing-table")

    def run_all():
        model = getattr(rag.client, "chat_model", "gpt-4o-mini")
        run = {"when": datetime.now().strftime("%d %b %H:%M"), "model": model, "rows": []}
        for i, (route, question, expected) in enumerate(PROBES, 1):
            yield render(run, running=i)
            run["rows"].append(run_one(route, question, expected, rag))
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
        yield render(run)

    run_btn.click(run_all, None, table)
