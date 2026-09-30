"""Compare two evaluation runs and flag regressions.

    python -m scripts.regression data/eval/baseline.json data/eval/results.json

Both files come from scripts/evaluate.py. Questions are matched by their text. Prints the
headline metrics side by side, then every question whose result changed: "fixed" (wrong
before, right now), "regressed" (right before, wrong now) and "moved" (still wrong, but
failing at a different stage). Writes data/eval/regression.md and exits with status 1 when
anything regressed, so it can gate a merge.
"""

import argparse
import json
import sys
from pathlib import Path

from app.utils import DATA_DIR

OUT_PATH = DATA_DIR / "eval" / "regression.md"
# (label, path into the summary, higher is better, format)
METRICS = [
    ("correct", ("correct",), True, "pct"),
    ("context recall", ("context_recall",), True, "pct"),
    ("cited an expected page", ("citation_hit",), True, "pct"),
    ("citation precision", ("citation_precision",), True, "pct"),
    ("refusal accuracy", ("refusal_accuracy",), True, "pct"),
    ("false refusals", ("false_refusal_rate",), False, "pct"),
    ("answer F1", ("answer_f1",), True, "num"),
    ("MRR (final order)", ("retrieval", "final", "mrr"), True, "num"),
    ("hit@1 (final order)", ("retrieval", "final", "hit@1"), True, "pct"),
    ("hit@8 (final order)", ("retrieval", "final", "hit@8"), True, "pct"),
    ("avg input tokens", ("input_tokens",), False, "int"),
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def lookup(summary: dict, keys: tuple):
    for key in keys:
        summary = summary.get(key) if isinstance(summary, dict) else None
    return summary


def fmt(value, kind: str) -> str:
    if value is None:
        return "—"
    return {"pct": f"{value:.0%}", "num": f"{value:.2f}", "int": f"{value:.0f}"}[kind]


def delta(before, after, higher_is_better: bool, kind: str) -> str:
    if before is None or after is None:
        return ""
    diff = after - before
    if abs(diff) < 1e-9:
        return "="
    shown = {"pct": f"{diff * 100:+.0f} pts", "num": f"{diff:+.2f}", "int": f"{diff:+.0f}"}[kind]
    return shown + (" ✅" if (diff > 0) == higher_is_better else " ⚠️")


def compare(baseline: dict, candidate: dict) -> tuple[list, list, list, list, list]:
    before = {q["question"]: q for q in baseline["questions"]}
    after = {q["question"]: q for q in candidate["questions"]}
    fixed, regressed, moved = [], [], []
    for question, new in after.items():
        old = before.get(question)
        if old is None:
            continue
        if not old["correct"] and new["correct"]:
            fixed.append((old, new))
        elif old["correct"] and not new["correct"]:
            regressed.append((old, new))
        elif not new["correct"] and old["stage"] != new["stage"]:
            moved.append((old, new))
    only_before = [q for q in before if q not in after]
    only_after = [q for q in after if q not in before]
    return fixed, regressed, moved, only_before, only_after


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def change_rows(pairs: list) -> list[list]:
    return [[new["id"], new["category"], new["question"][:70], old["stage"], new["stage"]] for old, new in pairs]


def report(baseline: dict, candidate: dict, names: tuple[str, str]) -> tuple[str, int]:
    b, c = baseline["summary"], candidate["summary"]
    metric_rows = [[label, fmt(lookup(b, keys), kind), fmt(lookup(c, keys), kind),
                    delta(lookup(b, keys), lookup(c, keys), better, kind)]
                   for label, keys, better, kind in METRICS]
    fixed, regressed, moved, only_before, only_after = compare(baseline, candidate)
    headers = ["id", "category", "question", "before", "after"]
    verdict = (f"**{len(regressed)} regression(s)** — review before merging." if regressed
               else "**No regressions.**")
    text = f"""# Regression check

Baseline: `{names[0]}` · Candidate: `{names[1]}`

{verdict} Fixed {len(fixed)}, regressed {len(regressed)}, still failing at a different stage {len(moved)}.

## Metrics

{table(["metric", "baseline", "candidate", "change"], metric_rows)}

## Regressed (right before, wrong now)

{table(headers, change_rows(regressed)) if regressed else "_None._"}

## Fixed (wrong before, right now)

{table(headers, change_rows(fixed)) if fixed else "_None._"}

## Still failing, different stage

{table(headers, change_rows(moved)) if moved else "_None._"}
"""
    if only_before or only_after:
        text += f"\n_Question sets differ: {len(only_before)} only in baseline, {len(only_after)} only in candidate._\n"
    return text, len(regressed)


def main():
    parser = argparse.ArgumentParser(description="Compare two scripts/evaluate.py result files")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    text, regressions = report(load(args.baseline), load(args.candidate), (args.baseline.name, args.candidate.name))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"written to {args.out}")
    sys.exit(1 if regressions else 0)


if __name__ == "__main__":
    main()
