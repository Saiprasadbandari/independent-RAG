"""The Evaluation page: the saved eval reports in data/eval/, read fresh each time the page opens.

Each tab shows one report as its script last wrote it, so re-running a script and pressing
Refresh is enough to update the page; nothing here calls a model.

    Retrieval            data/eval/report.md           scripts/eval_runner.py
    Ablation             data/eval/ablation.md         scripts/eval_runner.py --ablation
    Agent vs RAG         data/eval/agent_vs_rag.md     scripts/agent_vs_rag.py
    Prompt injection     data/eval/injection.md        scripts/injection_test.py
    Injection vs actions data/eval/action_injection.md scripts/action_injection_test.py
"""

import gradio as gr

from app.utils import DATA_DIR

EVAL_DIR = DATA_DIR / "eval"

# (tab label, report file, command that regenerates it)
REPORTS = [
    ("Retrieval", "report.md", "uv run python -m scripts.eval_runner --retrieval-only"),
    ("Ablation", "ablation.md", "uv run python -m scripts.eval_runner --retrieval-only --ablation"),
    ("Agent vs RAG", "agent_vs_rag.md", "uv run python -m scripts.agent_vs_rag --mode local"),
    ("Prompt injection", "injection.md", "uv run python -m scripts.injection_test --mode local"),
    ("Injection vs actions", "action_injection.md", "uv run python -m scripts.action_injection_test --mode local"),
]

EVALUATION_CSS = """
.eval-report table { display: block; overflow-x: auto; font-size: 13px; }
.eval-report td, .eval-report th { word-break: keep-all !important; overflow-wrap: normal !important;
                                   vertical-align: top; }
.eval-report td:first-child, .eval-report th:first-child { white-space: nowrap; }
"""


def render(filename: str, command: str) -> str:
    path = EVAL_DIR / filename
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return f"_`data/eval/{filename}` not found. Run `{command}` to create it._"
    return f"_Source: `data/eval/{filename}`. Regenerate with `{command}`._\n\n{text}"


def build_evaluation_page() -> None:
    gr.Markdown("How well retrieval, the agent and the safety checks work, from the last saved run of each "
                "eval script. Re-run a script, then press Refresh.")
    pages = []
    with gr.Tabs():
        for label, filename, command in REPORTS:
            with gr.Tab(label):
                pages.append(gr.Markdown(lambda f=filename, c=command: render(f, c), elem_classes="eval-report"))
    gr.Button("Refresh", size="sm").click(
        lambda: [render(f, c) for _, f, c in REPORTS], None, pages)
