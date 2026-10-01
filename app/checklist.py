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

An item's "files" say where the work is: a path, "path::function" (or "Class.method", or a
constant) for part of a Python file, or "README.md#Heading" for one section. Each becomes a link
to that place on GitHub, and git blame on main over exactly those lines says who wrote them.
"""

import ast
import fnmatch
import html
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import gradio as gr

from app.utils import DATA_DIR, ROOT

CHECKLIST_PATH = Path(__file__).with_name("checklist.json")
MAIN_REF = "origin/main"
GIT_TIMEOUT = 10
REPO_URL = "https://github.com/Saiprasadbandari/independent-RAG"
# git author e-mails (the part before @, after any "12345+" GitHub prefix) → team member
PEOPLE = {"xlucifer65": "Rayyan", "saiprasadbandari7": "Sai", "pro.paul108": "Prodipta"}
MIN_SHARE = 0.05  # contributors below this share of an item's lines aren't listed
# what the "Who did what" totals count: hand-written code and docs, not lock files or data
COUNTED_FILES = ("*.py", "*.md", "*.toml", ".gitignore")


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


# ---------------------------------------------------------------------------- links and authors

def person(email: str) -> str:
    handle = email.strip("<>").split("@")[0].lower().split("+")[-1]
    return PEOPLE.get(handle, handle)


class Blame:
    """Who wrote each line of each file on one commit, from git blame (cached per file)."""

    def __init__(self, ref: str):
        self.ref = ref
        self.lines: dict[str, list[str]] = {}

    def authors(self, path: str) -> list[str]:
        if path not in self.lines:
            try:
                out = subprocess.run(["git", "blame", "--line-porcelain", "-w", self.ref, "--", path], cwd=ROOT,
                                     capture_output=True, text=True, timeout=GIT_TIMEOUT * 3, check=True).stdout
                self.lines[path] = [person(line[len("author-mail "):]) for line in out.splitlines()
                                    if line.startswith("author-mail ")]
            except Exception:
                self.lines[path] = []
        return self.lines[path]


_BLAME: dict[str, Blame] = {}


def blame_for(ref: str) -> Blame:
    """One Blame per commit, so reopening the page doesn't blame every file again."""
    try:
        sha = subprocess.run(["git", "rev-parse", ref], cwd=ROOT, capture_output=True, text=True,
                             timeout=GIT_TIMEOUT, check=True).stdout.strip()
    except Exception:
        sha = ref
    return _BLAME.setdefault(sha, Blame(sha))


def slug(heading: str) -> str:
    """GitHub's anchor for a Markdown heading."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def markdown_span(text: str, heading: str) -> tuple[int, int] | None:
    """Lines (1-based, inclusive) of a heading and everything under it, until the next heading
    of the same or a higher level. Lines inside code fences are never headings."""
    lines, start, level, fenced = text.splitlines(), None, 0, False
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        m = None if fenced else re.match(r"^(#+)\s+(.*?)\s*$", line)
        if not m:
            continue
        if start is None and m.group(2).lower() == heading.lower():
            start, level = i, len(m.group(1))
        elif start is not None and len(m.group(1)) <= level:
            return start, i - 1
    return (start, len(lines)) if start else None


def python_span(text: str, name: str) -> tuple[int, int] | None:
    """Lines of a top-level function, class or constant, or of "Class.method"."""
    body = ast.parse(text).body
    *owners, last = name.split(".")
    for owner in owners:
        body = next((n.body for n in body if isinstance(n, ast.ClassDef) and n.name == owner), [])
    for node in body:
        names = ([node.name] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else
                 [t.id for t in getattr(node, "targets", [getattr(node, "target", None)]) if isinstance(t, ast.Name)])
        if last in names:
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            return first, node.end_lineno
    return None


def locate(spec: str, repo: Snapshot) -> tuple[str, str, tuple[int, int] | None, str] | None:
    """(path, label, line span or None for the whole file, GitHub URL); None when it isn't there."""
    sep = "::" if "::" in spec else ("#" if "#" in spec else "")
    path, part = spec.split(sep, 1) if sep else (spec, "")
    if not repo.exists(path):
        return None
    url = f"{REPO_URL}/blob/main/{path}"
    if not part:
        return path, path, None, url
    try:
        span = python_span(repo.read(path), part) if sep == "::" else markdown_span(repo.read(path), part)
    except SyntaxError:
        span = None
    if span is None:
        return None
    anchor = f"#{slug(part)}" if sep == "#" else f"#L{span[0]}-L{span[1]}"
    return path, f"{Path(path).name} › {part}", span, url + anchor


def contributors(spans: list[tuple[str, tuple[int, int] | None]], blame: Blame) -> Counter:
    counted, seen = Counter(), set()
    for path, span in spans:
        authors = blame.authors(path)
        first, last = span or (1, len(authors))
        for line in range(first, min(last, len(authors)) + 1):
            if (path, line) not in seen:
                seen.add((path, line))
                counted[authors[line - 1]] += 1
    return counted


def share_text(counted: Counter) -> str:
    total = sum(counted.values())
    shown = [(name, n / total) for name, n in counted.most_common() if total and n / total >= MIN_SHARE]
    return " · ".join(f"{name} {share:.0%}" for name, share in shown)


def item_files(item: dict) -> list[str]:
    if "files" in item:
        return item["files"]
    check = item.get("auto", "")
    return [check.removeprefix("file:")] if check.startswith("file:") else []


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
.checklist .meta { grid-column: 2; font-size: 12.5px; display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: center;
                   color: var(--body-text-color-subdued); }
.checklist .meta a { color: var(--link-text-color, var(--color-accent)); text-decoration: none; }
.checklist .meta a:hover { text-decoration: underline; }
.checklist .meta code { font-size: 12px; }
.checklist .who { font-weight: 600; color: var(--body-text-color); }
.checklist .people { display: flex; flex-wrap: wrap; gap: 10px; margin: 6px 0 18px; }
.checklist .person { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 8px 14px; min-width: 150px; }
.checklist .person b { font-size: 16px; }
.checklist .person div { font-size: 13px; color: var(--body-text-color-subdued); }
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


def item_meta(item: dict, local: Snapshot, main: Snapshot | None, blame: Blame, led: Counter) -> str:
    """Links to where the item is on GitHub, and who wrote those lines."""
    parts, spans = [], []
    for spec in item_files(item):
        found = locate(spec, main) if main else None
        if found:
            path, label, span, url = found
            spans.append((path, span))
            parts.append(f'<a href="{html.escape(url)}" target="_blank" rel="noopener">📄 {html.escape(label)}</a>')
        elif locate(spec, local):
            parts.append(f'<span>📄 <code>{html.escape(spec)}</code> (not on GitHub yet)</span>')
    if item.get("auto") in DATA_CHECKS:
        parts.append("<span>📦 data/, shared on Teams</span>")
    counted = contributors(spans, blame) if spans else Counter()
    if item.get("by"):
        # work git can't see (Teams posts, data/ files): named by hand in checklist.json
        parts.append(f'<span class="who">👤 {html.escape(item["by"])}</span>')
    elif counted:
        led[counted.most_common(1)[0][0]] += 1
        parts.append(f'<span class="who">👤 {html.escape(share_text(counted))}</span>')
    return f'<span class="meta">{"".join(parts)}</span>' if parts else ""


def people_summary(main: Snapshot | None, blame: Blame, led: Counter) -> str:
    if not main:
        return ""
    files = [n for n in main.files() if any(fnmatch.fnmatch(n, pattern) for pattern in COUNTED_FILES)]
    lines = contributors([(n, None) for n in files], blame)
    total = sum(lines.values()) or 1
    cards = "".join(f'<div class="person"><b>{html.escape(name)}</b>'
                    f'<div>{n:,} lines on main ({n / total:.0%})</div>'
                    f'<div>leads {led[name]} checklist item{"s" * (led[name] != 1)}</div></div>'
                    for name, n in lines.most_common())
    return (f'<h3>Who did what</h3><div class="legend">From git blame on GitHub main: who wrote the lines that are '
            f'there now (code and docs, not lock files or data). Each item below names who wrote the lines it '
            f'links to.</div><div class="people">{cards}</div>')


def render() -> str:
    data = json.loads(CHECKLIST_PATH.read_text(encoding="utf-8"))
    local, main = Snapshot(), main_snapshot()
    blame, led = blame_for(MAIN_REF if main else "HEAD"), Counter()
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
                meta = item_meta(item, local, main, blame, led)
                rows.append(f'<li><span>{ICONS[state]}</span><span>{html.escape(item["label"])}{tag}</span>'
                            f'{meta}{note_html}</li>')
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
    return CSS + f'<div class="checklist">{overall}{people_summary(main, blame, led)}{"".join(sections)}</div>'


def build_checklist_page() -> None:
    gr.Markdown("# Project checklist\nEvery task from the course instructions, grouped by day. Code and README "
                "items count as delivered once they're on GitHub `main`; items marked **auto** tick themselves, "
                "the others are ticked in `app/checklist.json`.")
    # a callable value is re-evaluated each time the page is opened, so ticks are always current
    page = gr.HTML(render)
    gr.Button("Refresh").click(render, None, page)
