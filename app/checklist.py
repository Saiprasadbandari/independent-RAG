"""Project checklist page: every task from the course instructions, ✅ when done, ⬜ when not.

The items live in app/checklist.json, grouped the way the course is: section (day) → group →
task. Items with an "auto" key tick themselves from what is in the repo and data/ (a file
exists, questions.json has follow-ups, the README has the ablation table...), so they stay right
as work lands. The rest are ticked by hand by setting "done" to true in the JSON file.
"""

import html
import json
import re
from pathlib import Path

import gradio as gr

from app.utils import DATA_DIR, ROOT

CHECKLIST_PATH = Path(__file__).with_name("checklist.json")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def questions() -> list[dict]:
    path = DATA_DIR / "questions.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def base_questions() -> list[dict]:
    """The Assignment 1 set: follow-ups and agent questions don't count towards the 40 to 50."""
    return [q for q in questions() if not q.get("history") and q.get("category") != "agent"]


def sub_types(category: str) -> set[str]:
    return {q["sub_type"] for q in base_questions() if q.get("category") == category and q.get("sub_type")}


def readme() -> str:
    return read(ROOT / "README.md")


def app_sources() -> str:
    # this file mentions what the checks look for, so it's left out of the search
    return "\n".join(read(p) for p in sorted((ROOT / "app").glob("*.py")) if p.name != "checklist.py")


def tools_source() -> str:
    return read(ROOT / "app" / "tools.py")


def minimums() -> bool:
    qs = base_questions()
    count = {c: sum(q.get("category") == c for q in qs) for c in
             ("factual", "cross-reference", "stress-test", "out-of-scope", "ambiguous")}
    return (40 <= len(qs) <= 50 and count["factual"] >= 10 and count["cross-reference"] >= 8
            and count["stress-test"] >= 10 and count["out-of-scope"] >= 3 and count["ambiguous"] >= 2)


def eval_results_from_api() -> bool:
    path = DATA_DIR / "eval_results.json"
    return path.exists() and json.loads(path.read_text(encoding="utf-8")).get("config", {}).get("backend") == "api"


CHECKS = {
    "gitignore": lambda: all(entry in read(ROOT / ".gitignore") for entry in (".venv", ".env", "data")),
    "uv_python312": lambda: (ROOT / "pyproject.toml").exists() and read(ROOT / ".python-version").startswith("3.12"),
    "corpus_files": lambda: (DATA_DIR / "corpus.json").exists() and (DATA_DIR / "sample.json").exists(),
    "questions_minimums": minimums,
    "questions_priority": lambda: sum(bool(q.get("priority")) for q in base_questions()) == 5,
    "questions_cross_subtypes": lambda: len(sub_types("cross-reference")) >= 5,
    "questions_stress_subtypes": lambda: len(sub_types("stress-test")) >= 8,
    "questions_follow_ups": lambda: sum(bool(q.get("history")) for q in questions()) >= 3,
    "questions_agent": lambda: sum(q.get("category") == "agent" and bool(q.get("expected_tools")) for q in questions()) >= 9,
    "conversation_memory": lambda: bool(re.search(r"def answer_messages\([^)]*history", read(ROOT / "app" / "rag.py"))),
    "readme_threshold": lambda: bool(re.search(r"\|\s*threshold\s*\|.*refusal", readme(), re.IGNORECASE)),
    "readme_ablation": lambda: bool(re.search(r"\|\s*vector only\s*\|\s*0?\.\d", readme(), re.IGNORECASE)),
    "readme_install": lambda: "uv sync" in readme() and "uv run python -m app.main" in readme(),
    "readme_rag_demo": lambda: bool(re.search(r"^#+ .*demo", readme(), re.IGNORECASE | re.MULTILINE)),
    "readme_own_tool": lambda: "quote_exact" in readme(),
    "readme_agent_vs_rag": lambda: bool(re.search(r"agent vs\.? rag", readme(), re.IGNORECASE)),
    "injection_flag": lambda: "INJECTION_TEST" in app_sources() and "<context>" in app_sources(),
    "eval_results_api": eval_results_from_api,
    "failure_log_regressions": lambda: "## Regression checks" in read(ROOT / "scripts" / "FAILURE_LOG.md"),
    "ui_agent_mode": lambda: "run_agent" in read(ROOT / "app" / "ui.py"),
    "five_tools": lambda: len(re.findall(r'"name":\s*"', tools_source())) >= 5,
    "send_message_tool": lambda: "send_message" in tools_source(),
    "agent_eval_runner": lambda: any(p.name for p in (ROOT / "scripts").glob("*agent*eval*.py"))
                                 or any(p.name for p in (ROOT / "scripts").glob("*eval*agent*.py")),
}


def is_done(item: dict) -> bool:
    check = item.get("auto")
    if not check:
        return bool(item.get("done"))
    try:
        if check.startswith("file:"):
            return (ROOT / check.removeprefix("file:")).exists()
        if check.startswith("tool:"):
            return f'"{check.removeprefix("tool:")}"' in tools_source()
        return bool(CHECKS[check]())
    except Exception:
        # a check that can't run (missing or malformed file) counts as not done
        return False


CSS = """
<style>
.checklist { font-size: 15px; line-height: 1.5; }
.checklist .overall { margin: 4px 0 18px; }
.checklist .bar { height: 8px; border-radius: 4px; background: var(--border-color-primary); overflow: hidden; }
.checklist .bar span { display: block; height: 100%; background: var(--color-accent); }
.checklist details { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; }
.checklist summary { cursor: pointer; font-weight: 600; font-size: 17px; display: flex; gap: 12px; align-items: center; }
.checklist summary .count { margin-left: auto; font-weight: 500; font-size: 14px; color: var(--body-text-color-subdued); }
.checklist .group { margin: 12px 0 4px 6px; }
.checklist .group h4 { margin: 0 0 6px; font-size: 15px; display: flex; gap: 10px; }
.checklist .group h4 .count { color: var(--body-text-color-subdued); font-weight: 400; }
.checklist ul { list-style: none; margin: 0; padding-left: 14px; }
.checklist li { display: grid; grid-template-columns: 1.6em 1fr; margin: 3px 0; }
.checklist .note { grid-column: 2; font-size: 13px; color: var(--body-text-color-subdued); }
.checklist .tag { font-size: 11px; padding: 0 6px; border-radius: 6px; border: 1px solid var(--border-color-primary);
                  color: var(--body-text-color-subdued); margin-left: 6px; }
</style>
"""


def bar(done: int, total: int) -> str:
    width = round(100 * done / total) if total else 0
    return f'<div class="bar" title="{done}/{total}"><span style="width:{width}%"></span></div>'


def render() -> str:
    data = json.loads(CHECKLIST_PATH.read_text(encoding="utf-8"))
    sections, all_done, all_total = [], 0, 0
    for section in data["sections"]:
        groups, s_done, s_total = [], 0, 0
        for group in section["groups"]:
            rows, g_done = [], 0
            for item in group["items"]:
                done = is_done(item)
                g_done += done
                tag = '<span class="tag">auto</span>' if item.get("auto") else ""
                note = f'<span class="note">{html.escape(item["note"])}</span>' if item.get("note") and not done else ""
                rows.append(f'<li><span>{"✅" if done else "⬜"}</span><span>{html.escape(item["label"])}{tag}</span>{note}</li>')
            total = len(group["items"])
            s_done, s_total = s_done + g_done, s_total + total
            groups.append(f'<div class="group"><h4>{"✅" if g_done == total else "⬜"} {html.escape(group["title"])}'
                          f'<span class="count">{g_done}/{total}</span></h4><ul>{"".join(rows)}</ul></div>')
        all_done, all_total = all_done + s_done, all_total + s_total
        icon = "✅" if s_done == s_total else "⬜"
        sections.append(f'<details open><summary>{icon} {html.escape(section["title"])}'
                        f'<span class="count">{s_done}/{s_total}</span></summary>{bar(s_done, s_total)}{"".join(groups)}</details>')
    overall = (f'<div class="overall"><b>{all_done} of {all_total} tasks done</b> '
               f'({round(100 * all_done / all_total) if all_total else 0}%){bar(all_done, all_total)}</div>')
    return CSS + f'<div class="checklist">{overall}{"".join(sections)}</div>'


def build_checklist_page() -> None:
    gr.Markdown("# Project checklist\nEvery task from the course instructions, grouped by day. "
                "Items marked **auto** tick themselves from the repo and `data/`; the others are ticked "
                "in `app/checklist.json`.")
    # a callable value is re-evaluated each time the page is opened, so ticks are always current
    page = gr.HTML(render)
    gr.Button("Refresh").click(render, None, page)
