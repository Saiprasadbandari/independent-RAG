"""The Testing log page: scripts/TESTING_LOG.md, read fresh each time the page opens."""

import re

import gradio as gr

from app.utils import ROOT

LOG_PATH = ROOT / "scripts" / "TESTING_LOG.md"
TESTING_LOG_CSS = """
#testing-log table { display: block; overflow-x: auto; font-size: 13px; }
#testing-log td, #testing-log th { word-break: keep-all !important; overflow-wrap: normal !important;
                                   vertical-align: top; }
#testing-log td:nth-child(-n+3), #testing-log th:nth-child(-n+3) { white-space: nowrap; }
"""


def render() -> str:
    try:
        text = LOG_PATH.read_text(encoding="utf-8")
    except OSError:
        return "_`scripts/TESTING_LOG.md` not found._"
    # the numbered contents list links to GitHub-style anchors, which don't exist inside the app
    return re.sub(r"\[([^\]]+)\]\(#[^)]+\)", r"\1", text)


def build_testing_log_page() -> None:
    gr.Markdown("Every test from the first commit on, what it found, what changed and how the numbers moved. "
                "Source: `scripts/TESTING_LOG.md` on this machine; add a row there and refresh.")
    page = gr.Markdown(render, elem_id="testing-log")
    gr.Button("Refresh", size="sm").click(render, None, page)
