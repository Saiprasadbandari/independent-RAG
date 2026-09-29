"""Export a conversation, or one answer from it, as plain text or PDF.

Turns are the dicts the UI keeps: {"question", "answer": asdict(Answer)} or
{"question", "error"}. PDFs are laid out with PyMuPDF (already a dependency).
"""

import html
import re
from datetime import datetime
from pathlib import Path

import pymupdf as fitz

from app.utils import format_citation

TITLE = "EU AI Act Assistant"
PDF_CSS = """
body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.4; }
h1 { font-size: 16pt; margin-bottom: 2pt; }
h2 { font-size: 12pt; margin-top: 14pt; margin-bottom: 4pt; }
.meta { color: #666666; font-size: 9pt; }
.label { font-weight: bold; }
"""


def turn_lines(number: int, turn: dict) -> list[str]:
    lines = [f"Q{number}: {turn['question']}"]
    if "error" in turn:
        return lines + [f"Error: {turn['error']}"]
    answer = turn["answer"]
    if answer["standalone_question"] and answer["standalone_question"] != turn["question"]:
        lines.append(f"(Interpreted as: {answer['standalone_question']})")
    lines.append(f"A{number}: {answer['text']}")
    if answer["confidence_label"]:
        subject = "Answer" if answer["in_scope"] else "Refusal"
        lines.append(f"{subject} confidence: {answer['confidence']:.0%} ({answer['confidence_label']})"
                     f" — {'; '.join(answer['confidence_notes'])}")
    if answer["sources"]:
        lines.append("Sources: " + "; ".join(f"[{s['ref']}] {format_citation(s)}" for s in answer["sources"]))
    return lines


def to_text(turns: list[tuple[int, dict]], heading: str) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    blocks = ["\n".join(turn_lines(n, t)) for n, t in turns]
    return f"{TITLE} — {heading}\nExported {stamp}\n\n" + "\n\n---\n\n".join(blocks) + "\n"


def paragraph(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    return escaped.replace("\n", "<br/>")


def to_html(turns: list[tuple[int, dict]], heading: str) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [f"<h1>{TITLE}</h1><p class='meta'>{html.escape(heading)} · exported {stamp}</p>"]
    for number, turn in turns:
        lines = turn_lines(number, turn)
        parts.append(f"<h2>{paragraph(lines[0])}</h2>")
        parts += [f"<p>{paragraph(line)}</p>" for line in lines[1:]]
    return "".join(parts)


def write_pdf(markup: str, path: Path) -> None:
    story = fitz.Story(html=markup, user_css=PDF_CSS)
    writer = fitz.DocumentWriter(str(path))
    page = fitz.paper_rect("a4")
    body = page + (50, 50, -50, -50)
    more = True
    while more:
        device = writer.begin_page(page)
        more, _ = story.place(body)
        story.draw(device)
        writer.end_page()
    writer.close()


def export(turns: list[tuple[int, dict]], heading: str, fmt: str, out_dir: Path, name: str) -> Path:
    """turns: (turn number, turn) pairs. fmt: "txt" or "pdf". Returns the written file."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{name}-{stamp}.{fmt}"
    if fmt == "pdf":
        write_pdf(to_html(turns, heading), path)
    else:
        path.write_text(to_text(turns, heading), encoding="utf-8")
    return path
