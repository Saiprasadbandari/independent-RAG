"""Project checklist page: every task from the course instructions and whether it's delivered.

The items live in app/checklist.json, grouped the way the course is: section (day) → group →
task. Each item shows one of three states:

- ✅ delivered: on GitHub main (code, README), shared via data/ (Teams), or ticked by hand
- 🟡 done but not uploaded: the work exists on this machine or in an open PR, not on main yet
- ⬜ not done

Items with an "auto" key tick themselves. Repo checks (files, code, README) run twice: against
the files on disk and against origin/main, which is what tells "done" from "delivered". Data
checks (questions.json, corpus.json, eval results) only look at data/, which the course shares on
Teams rather than in git. Items without "auto" are ticked by hand in app/checklist.json.
"""

import fnmatch
import html
import json
import re
import subprocess
from pathlib import Path

import gradio as gr

from app.utils import DATA_DIR, ROOT

CHECKLIST_PATH = Path(__file__).with_name("checklist.json")
MAIN_REF = "origin/main"
GIT_TIMEOUT = 10


# ---------------------------------------------------------------------------- repo snapshots

class Snapshot:
    """The repo's files either as they are on disk (ref=None) or as they are on a git ref."""

    def __init__(self, ref: str | None = None):
        self.ref = ref
        self.cache: dict[str, str] = {}
        self.names: list[str] | None = None

    def _git(self, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=GIT_TIMEOUT, check=True).stdout

    def files(self) -> list[str]:
        if self.names is None:
            if self.ref:
                self.names = self._git("ls-tree", "-r", "--name-only", self.ref).splitlines()
            else:
                self.names = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
                              if p.is_file() and ".venv" not in p.parts and ".git" not in p.parts]
        return self.names

    def exists(self, name: str) -> bool:
        return (ROOT / name).exists() if self.ref is None else name in self.files()

    def read(self, name: str) -> str:
        if name not in self.cache:
            if self.ref is None:
                path = ROOT / name
                self.cache[name] = path.read_text(encoding="utf-8") if path.exists() else ""
            else:
                self.cache[name] = self._git("show", f"{self.ref}:{name}") if self.exists(name) else ""
        return self.cache[name]

    def glob(self, pattern: str) -> list[str]:
        return [n for n in self.files() if fnmatch.fnmatch(n, pattern)]


def main_snapshot() -> Snapshot | None:
    """origin/main, freshly fetched when possible. None outside a git checkout."""
    try:
        subprocess.run(["git", "fetch", "--quiet", "origin", "main"], cwd=ROOT, capture_output=True,
                       timeout=GIT_TIMEOUT)
        snapshot = Snapshot(MAIN_REF)
        snapshot.files()
        return snapshot
    except Exception:
        return None


# ---------------------------------------------------------------------------- checks

def app_sources(repo: Snapshot) -> str:
    # this file mentions what the checks look for, so it's left out of the search
    return "\n".join(repo.read(n) for n in repo.glob("app/*.py") if not n.endswith("checklist.py"))


REPO_CHECKS = {
    "gitignore": lambda r: all(entry in r.read(".gitignore") for entry in (".venv", ".env", "data")),
    "uv_python312": lambda r: r.exists("pyproject.toml") and r.read(".python-version").startswith("3.12"),
    "conversation_memory": lambda r: bool(re.search(r"def answer_messages\([^)]*history", r.read("app/rag.py"))),
    "readme_threshold": lambda r: bool(re.search(r"\|\s*threshold\s*\|.*refusal", r.read("README.md"), re.IGNORECASE)),
    "readme_ablation": lambda r: bool(re.search(r"\|\s*vector only\s*\|\s*0?\.\d", r.read("README.md"), re.IGNORECASE)),
    "readme_install": lambda r: "uv sync" in r.read("README.md") and "uv run python -m app.main" in r.read("README.md"),
    "readme_rag_demo": lambda r: bool(re.search(r"^#+ .*demo", r.read("README.md"), re.IGNORECASE | re.MULTILINE)),
    "readme_own_tool": lambda r: "quote_exact" in r.read("README.md"),
    "readme_agent_vs_rag": lambda r: bool(re.search(r"agent vs\.? rag", r.read("README.md"), re.IGNORECASE)),
    "injection_flag": lambda r: "INJECTION_TEST" in app_sources(r) and "<context>" in app_sources(r),
    "failure_log_regressions": lambda r: "## Regression checks" in r.read("scripts/FAILURE_LOG.md"),
    "ui_agent_mode": lambda r: "run_agent" in r.read("app/ui.py"),
    "five_tools": lambda r: len(re.findall(r'"name":\s*"', r.read("app/tools.py"))) >= 5,
    "send_message_tool": lambda r: "send_message" in r.read("app/tools.py"),
    "agent_eval_runner": lambda r: bool(r.glob("scripts/*agent*eval*.py") or r.glob("scripts/*eval*agent*.py")),
}


def questions() -> list[dict]:
    path = DATA_DIR / "questions.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def base_questions() -> list[dict]:
    """The Assignment 1 set: follow-ups and agent questions don't count towards the 40 to 50."""
    return [q for q in questions() if not q.get("history") and q.get("category") != "agent"]


def sub_types(category: str) -> set[str]:
    return {q["sub_type"] for q in base_questions() if q.get("category") == category and q.get("sub_type")}


def minimums() -> bool:
    qs = base_questions()
    count = {c: sum(q.get("category") == c for q in qs) for c in
             ("factual", "cross-reference", "stress-test", "out-of-scope", "ambiguous")}
    return (40 <= len(qs) <= 50 and count["factual"] >= 10 and count["cross-reference"] >= 8
            and count["stress-test"] >= 10 and count["out-of-scope"] >= 3 and count["ambiguous"] >= 2)


def eval_results_from_api() -> bool:
    path = DATA_DIR / "eval_results.json"
    return path.exists() and json.loads(path.read_text(encoding="utf-8")).get("config", {}).get("backend") == "api"


# data/ is shared on Teams, never in git, so these only look at the local data/ folder
DATA_CHECKS = {
    "corpus_files": lambda: (DATA_DIR / "corpus.json").exists() and (DATA_DIR / "sample.json").exists(),
    "questions_minimums": minimums,
    "questions_priority": lambda: sum(bool(q.get("priority")) for q in base_questions()) == 5,
    "questions_cross_subtypes": lambda: len(sub_types("cross-reference")) >= 5,
    "questions_stress_subtypes": lambda: len(sub_types("stress-test")) >= 8,
    "questions_follow_ups": lambda: sum(bool(q.get("history")) for q in questions()) >= 3,
    "questions_agent": lambda: sum(q.get("category") == "agent" and bool(q.get("expected_tools")) for q in questions()) >= 9,
    "eval_results_api": eval_results_from_api,
}


def repo_check(check: str, repo: Snapshot) -> bool:
    try:
        if check.startswith("file:"):
            return repo.exists(check.removeprefix("file:"))
        if check.startswith("tool:"):
            return f'"{check.removeprefix("tool:")}"' in repo.read("app/tools.py")
        return bool(REPO_CHECKS[check](repo))
    except Exception:
        # a check that can't run (missing or malformed file) counts as not done
        return False


def status(item: dict, local: Snapshot, main: Snapshot | None) -> str:
    """"done" (delivered), "local" (done but not on main yet) or "todo"."""
    check = item.get("auto")
    if not check:
        return "done" if item.get("done") else "todo"
    if check in DATA_CHECKS:
        try:
            return "done" if DATA_CHECKS[check]() else "todo"
        except Exception:
            return "todo"
    if not repo_check(check, local) and not (main and repo_check(check, main)):
        return "todo"
    if main is None or repo_check(check, main):
        return "done"
    return "local"


# ---------------------------------------------------------------------------- page

ICONS = {"done": "✅", "local": "🟡", "todo": "⬜"}
LOCAL_NOTE = "Done here or in an open PR, but not on GitHub main yet: merge it to deliver it."

CSS = """
<style>
.checklist { font-size: 15px; line-height: 1.5; }
.checklist .overall { margin: 4px 0 6px; }
.checklist .legend { font-size: 13px; color: var(--body-text-color-subdued); margin-bottom: 16px; }
.checklist .bar { height: 8px; border-radius: 4px; background: var(--border-color-primary); overflow: hidden; display: flex; }
.checklist .bar .done { background: var(--color-accent); }
.checklist .bar .local { background: #e3b341; }
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


def bar(done: int, local: int, total: int) -> str:
    def width(n: int) -> int:
        return round(100 * n / total) if total else 0
    return (f'<div class="bar" title="{done} delivered, {local} waiting to be merged, {total} in total">'
            f'<span class="done" style="width:{width(done)}%"></span><span class="local" style="width:{width(local)}%"></span></div>')


def count_label(done: int, local: int, total: int) -> str:
    return f"{done}/{total}" + (f" · {local} 🟡" if local else "")


def group_icon(states: list[str]) -> str:
    if all(s == "done" for s in states):
        return ICONS["done"]
    return ICONS["local"] if all(s != "todo" for s in states) else ICONS["todo"]


def render() -> str:
    data = json.loads(CHECKLIST_PATH.read_text(encoding="utf-8"))
    local, main = Snapshot(), main_snapshot()
    sections, everything = [], []
    for section in data["sections"]:
        groups, section_states = [], []
        for group in section["groups"]:
            rows, states = [], []
            for item in group["items"]:
                state = status(item, local, main)
                states.append(state)
                tag = '<span class="tag">auto</span>' if item.get("auto") else ""
                note = LOCAL_NOTE if state == "local" else (item.get("note") if state == "todo" else None)
                note_html = f'<span class="note">{html.escape(note)}</span>' if note else ""
                rows.append(f'<li><span>{ICONS[state]}</span><span>{html.escape(item["label"])}{tag}</span>{note_html}</li>')
            section_states += states
            groups.append(f'<div class="group"><h4>{group_icon(states)} {html.escape(group["title"])}'
                          f'<span class="count">{count_label(states.count("done"), states.count("local"), len(states))}'
                          f'</span></h4><ul>{"".join(rows)}</ul></div>')
        everything += section_states
        done, waiting, total = section_states.count("done"), section_states.count("local"), len(section_states)
        sections.append(f'<details open><summary>{group_icon(section_states)} {html.escape(section["title"])}'
                        f'<span class="count">{count_label(done, waiting, total)}</span></summary>'
                        f'{bar(done, waiting, total)}{"".join(groups)}</details>')
    done, waiting, total = everything.count("done"), everything.count("local"), len(everything)
    where = f"compared with GitHub <code>{MAIN_REF}</code>" if main else "GitHub main not reachable, local files only"
    overall = (f'<div class="overall"><b>{done} of {total} delivered</b>'
               f'{f" · {waiting} done but waiting to be merged" if waiting else ""} · {where}{bar(done, waiting, total)}</div>'
               f'<div class="legend">✅ delivered (on GitHub main, shared on Teams, or ticked by hand) · '
               f'🟡 done, not on main yet · ⬜ not done</div>')
    return CSS + f'<div class="checklist">{overall}{"".join(sections)}</div>'


def build_checklist_page() -> None:
    gr.Markdown("# Project checklist\nEvery task from the course instructions, grouped by day. Code and README "
                "items count as delivered once they're on GitHub `main`; items marked **auto** tick themselves, "
                "the others are ticked in `app/checklist.json`.")
    # a callable value is re-evaluated each time the page is opened, so ticks are always current
    page = gr.HTML(render)
    gr.Button("Refresh").click(render, None, page)
