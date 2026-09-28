import json
import re
from collections import Counter
from pathlib import Path

import pymupdf as fitz
from bs4 import BeautifulSoup

SOURCES = Path("data/sources")
OUT_DIR = Path("data")
MIN_CHARS = 50


def collapse_whitespace(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def dehyphenate_and_join(text: str) -> str:
    # PDFs break lines mid-sentence and mid-word; rejoin lines that don't
    # end a sentence/paragraph so extracted text reads as prose.
    text = re.sub(r"-\n(\w)", r"\1", text)  # word split across a line break
    lines = text.split("\n")
    joined = []
    for ln in lines:
        if joined and not re.search(r"[.:;!?\"')\]]$", joined[-1]) and not ln[:1].isupper():
            joined[-1] = joined[-1] + " " + ln
        else:
            joined.append(ln)
    return "\n".join(joined)


PAGE_NUMBER_RE = re.compile(r"^\d+\s*/\s*\d+$")


def find_repeated_lines(pages_text, min_pages=3, threshold=0.4):
    """Lines that repeat across many pages (headers/footers) get flagged for removal."""
    counts = Counter()
    for text in pages_text:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        candidates = set(lines[:3] + lines[-4:])
        counts.update(candidates)
    if len(pages_text) < min_pages:
        return set()
    cutoff = max(2, int(len(pages_text) * threshold))
    return {
        line for line, n in counts.items()
        if n >= cutoff and (len(line) > 2 or line.isalpha())
    }


def strip_lines(text: str, drop: set) -> str:
    lines = [
        ln for ln in text.splitlines()
        if ln.strip() not in drop and not PAGE_NUMBER_RE.match(ln.strip())
    ]
    return "\n".join(lines)


def clean(text: str, drop_lines: set | None = None) -> str:
    text = collapse_whitespace(text)
    if drop_lines:
        text = strip_lines(text, drop_lines)
    text = dehyphenate_and_join(text)
    return text.strip()


def extract_pdf(path):
    doc = fitz.open(path)
    raw_pages = [page.get_text() for page in doc]
    drop_lines = find_repeated_lines(raw_pages)
    for i, raw in enumerate(raw_pages, start=1):
        yield i, clean(raw, drop_lines)


def extract_html(path):
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    yield 1, clean(text)


def extract_text_file(path):
    yield 1, clean(path.read_text(encoding="utf-8"))


def main():
    entries, near_empty = [], 0

    for path in sorted(SOURCES.iterdir()):
        ext = path.suffix.lower()
        if ext == ".pdf":
            pages = extract_pdf(path)
        elif ext in {".txt", ".md"}:
            pages = extract_text_file(path)
        elif ext in {".html", ".htm"}:
            pages = extract_html(path)
        else:
            continue

        for page, text in pages:
            char_count = len(text)
            if char_count < MIN_CHARS:
                near_empty += 1
            entries.append({
                "source": path.name,
                "page": page,
                "char_count": char_count,
                "text": text,
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "corpus.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "sample.json").write_text(
        json.dumps(entries[:10], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    chars = sum(e["char_count"] for e in entries)
    files = len({e["source"] for e in entries})
    print(f"source files: {files}")
    print(f"total entries: {len(entries)}")
    print(f"total characters: {chars}")
    print(f"avg characters/entry: {chars // max(len(entries), 1)}")
    print(f"near-empty entries (<{MIN_CHARS} chars): {near_empty}")


if __name__ == "__main__":
    main()
