"""Reflection: how far to trust an answer, scored from signals the pipeline already has.

No extra API calls. Each signal is 0 to 1:
- relevance: how well the best passage matches the question (re-ranker score, or cosine
  similarity rescaled when re-ranking is off)
- support:   how many passages cleared the cut-offs (FULL_SUPPORT or more counts as full)
- citation:  did the answer cite the passages it was given, and only those
- grounding: how well the cited passages match the answer text itself, scored by the local
  re-ranker (skipped when it isn't installed)

Confidence is their weighted average. An uncited or truncated answer is capped below "high".
For a refusal, confidence is about the refusal itself: high when nothing relevant was
found, low when a strong passage existed and the model still declined.

The weights and bands are starting points; scripts/eval_runner.py (full mode) reports how often each
band is actually right so they can be tuned on the real question set.
"""

import re
from dataclasses import dataclass, field

WEIGHTS = {"relevance": 0.35, "grounding": 0.30, "citation": 0.20, "support": 0.15}
HIGH = 0.70
MEDIUM = 0.45
FULL_SUPPORT = 3
# cosine similarity range mapped onto 0..1 when there is no re-ranker score
COSINE_FLOOR = 0.30
COSINE_CEILING = 0.70
# bump when the scoring changes so cached answers get re-scored
REFLECTION_VERSION = 2

BRACKETS_RE = re.compile(r"\s*\[[^\]]*\]")


@dataclass
class Confidence:
    score: float
    label: str
    signals: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def label_for(score: float) -> str:
    if score >= HIGH:
        return "high"
    if score >= MEDIUM:
        return "medium"
    return "low"


def passage_relevance(chunk: dict) -> float:
    if "relevance" in chunk:
        return chunk["relevance"]
    scaled = (chunk["score"] - COSINE_FLOOR) / (COSINE_CEILING - COSINE_FLOOR)
    return min(max(scaled, 0.0), 1.0)


def assess_refusal(retrieved: list[dict]) -> Confidence:
    best = max((passage_relevance(c) for c in retrieved), default=0.0)
    score = 1.0 - best
    notes = [f"best passage relevance {best:.2f}"]
    if label_for(best) == "high":
        notes.append("a strongly relevant passage was found, so this refusal may be wrong")
    return Confidence(round(score, 3), label_for(score), {"relevance": round(best, 3)}, notes)


def assess(text: str, in_scope: bool, truncated: bool, context: list[dict], retrieved: list[dict],
           cited: list[dict], invalid_refs: list[int], reranker=None) -> Confidence:
    if not in_scope:
        return assess_refusal(retrieved)

    signals, notes = {}, []

    signals["relevance"] = passage_relevance(context[0]) if context else 0.0
    notes.append(f"best passage relevance {signals['relevance']:.2f}")

    signals["support"] = min(len(context) / FULL_SUPPORT, 1.0)
    if len(context) < 2:
        notes.append(f"only {len(context)} passage{'s' if len(context) != 1 else ''} passed the filters")

    if not cited:
        signals["citation"] = 0.0
        notes.append("answer cites no passages")
    elif invalid_refs:
        signals["citation"] = 0.5
        notes.append(f"cites passage(s) {invalid_refs} that weren't provided")
    else:
        signals["citation"] = 1.0
        notes.append(f"cites {len(cited)} passage{'s' if len(cited) != 1 else ''}")

    if reranker is not None and cited:
        claim = BRACKETS_RE.sub("", text).strip()
        signals["grounding"] = max(reranker.score(claim, cited))
        notes.append(f"answer–source match {signals['grounding']:.2f}")

    weights = {name: WEIGHTS[name] for name in signals}
    score = sum(signals[name] * w for name, w in weights.items()) / sum(weights.values())
    # an answer that can't be traced back to a passage, or was cut off, is never "high"
    if not cited:
        score = min(score, HIGH - 0.01)
    if truncated:
        score = min(score, HIGH - 0.01)
        notes.append("answer was cut off at the length limit")

    return Confidence(round(score, 3), label_for(score),
                      {name: round(value, 3) for name, value in signals.items()}, notes)
