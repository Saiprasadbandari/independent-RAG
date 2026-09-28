import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# USD per 1M tokens
PRICES = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}
BUDGET_USD = 5.0

SENTENCE_END_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END_RE.split(text) if s.strip()]


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Pack sentences into ~size-char chunks, carrying ~overlap chars into the next chunk."""
    chunks, current = [], []
    length = 0
    for sentence in split_sentences(text):
        # hard-split anything longer than a whole chunk so it can't blow past the limit
        pieces = [sentence[i:i + size] for i in range(0, len(sentence), size)] or [sentence]
        for piece in pieces:
            if current and length + len(piece) + 1 > size:
                chunks.append(" ".join(current))
                tail, tail_len = [], 0
                for prev in reversed(current):
                    if tail_len + len(prev) > overlap:
                        break
                    tail.insert(0, prev)
                    tail_len += len(prev) + 1
                current, length = tail, tail_len
            current.append(piece)
            length += len(piece) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


class CostTracker:
    """Accumulates API spend across runs in data/cost.json and enforces the budget."""

    def __init__(self, path: Path = DATA_DIR / "cost.json", budget: float = BUDGET_USD):
        self.path = path
        self.budget = budget
        self.total = 0.0
        self.calls = 0
        if path.exists():
            saved = json.loads(path.read_text())
            self.total = saved.get("total_usd", 0.0)
            self.calls = saved.get("calls", 0)

    @property
    def remaining(self) -> float:
        return self.budget - self.total

    def check(self) -> None:
        if self.total >= self.budget:
            raise RuntimeError(
                f"Budget of ${self.budget:.2f} reached (spent ${self.total:.4f}). "
                "Delete data/cost.json only if you've confirmed actual usage."
            )

    def add(self, model: str, input_tokens: int, output_tokens: int = 0) -> float:
        price = PRICES[model]
        cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
        self.total += cost
        self.calls += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"total_usd": self.total, "calls": self.calls}, indent=2))
        return cost

    def summary(self) -> str:
        return f"Spent ${self.total:.4f} of ${self.budget:.2f} ({self.calls} API calls)"


def format_citation(chunk: dict) -> str:
    return f"{chunk['source']}, p. {chunk['page']}"


def format_sources(chunks: list[dict]) -> str:
    lines = [f"[{i}] {format_citation(c)} (score {c['score']:.2f})" for i, c in enumerate(chunks, 1)]
    return "\n".join(lines)


def format_retrieved(chunks: list[dict], preview: int = 400) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        text = c["text"] if len(c["text"]) <= preview else c["text"][:preview].rstrip() + "…"
        blocks.append(f"**[{i}] {format_citation(c)}** — score {c['score']:.2f}\n\n> {text}")
    return "\n\n---\n\n".join(blocks)
