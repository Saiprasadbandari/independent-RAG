"""
Checks data/questions.json against the Assignment 1 rubric.
Run after writing your questions: python scripts/validate_questions.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

QUESTIONS_PATH = Path("data/questions.json")
CORPUS_PATH = Path("data/corpus.json")

CATEGORIES = {"factual", "cross-reference", "stress-test", "out-of-scope", "ambiguous"}
CROSS_REF_SUBTYPES = {
    "adjacent_chunks", "same_document", "different_documents",
    "repeated", "contradictory", "summary_vs_detail",
}
STRESS_SUBTYPES = {
    "paraphrased", "exact_term", "yes_no", "temporal", "negation",
    "multi_hop", "table_list", "edge_of_chunk", "language_variation",
    "on_topic_unanswerable", "multi_part",
}
NEEDS_ANSWER_FIELDS = {"factual", "cross-reference", "stress-test"}
MIN_COUNTS = {"factual": 10, "cross-reference": 8, "stress-test": 10, "out-of-scope": 3, "ambiguous": 2}

errors = []
warnings = []


def err(msg):
    errors.append(msg)


def warn(msg):
    warnings.append(msg)


def pages_as_list(page):
    if page is None:
        return []
    if isinstance(page, list):
        return page
    return [page]


def main():
    if not QUESTIONS_PATH.exists():
        err(f"{QUESTIONS_PATH} does not exist")
        report()
        return

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    corpus_pages = set()
    if CORPUS_PATH.exists():
        corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        corpus_pages = {(e["source"], e["page"]) for e in corpus}
    else:
        warn(f"{CORPUS_PATH} not found, skipping source/page cross-check")

    n = len(questions)
    if not (40 <= n <= 50):
        err(f"total questions is {n}, expected 40-50")

    cat_counts = Counter()
    priority_count = 0
    cross_ref_subtypes_seen = set()
    stress_subtypes_seen = set()
    factual_sources = Counter()

    for i, q in enumerate(questions):
        loc = f"question {i} ({q.get('question', '')[:40]!r})"

        category = q.get("category")
        if category not in CATEGORIES:
            err(f"{loc}: invalid category {category!r}")
            continue
        cat_counts[category] += 1

        if q.get("priority") is True:
            priority_count += 1

        if "difficulty" not in q or q["difficulty"] not in {"easy", "medium", "hard"}:
            err(f"{loc}: missing/invalid difficulty")

        if category in NEEDS_ANSWER_FIELDS:
            if not q.get("expected_answer"):
                err(f"{loc}: missing expected_answer")
            if not q.get("source"):
                err(f"{loc}: missing source")
            if q.get("page") is None:
                err(f"{loc}: missing page")
            if not q.get("expected_chunks"):
                err(f"{loc}: missing expected_chunks")
            else:
                for chunk in q["expected_chunks"]:
                    key = (chunk.get("source"), chunk.get("page"))
                    if corpus_pages and key not in corpus_pages:
                        err(f"{loc}: expected_chunks entry {key} not found in corpus.json")

        if category == "cross-reference":
            sub = q.get("sub_type")
            if sub not in CROSS_REF_SUBTYPES:
                err(f"{loc}: missing/invalid sub_type {sub!r}")
            else:
                cross_ref_subtypes_seen.add(sub)
            if len(pages_as_list(q.get("page"))) < 2:
                warn(f"{loc}: cross-reference question has fewer than 2 pages listed")

        if category == "stress-test":
            sub = q.get("sub_type")
            if sub not in STRESS_SUBTYPES:
                err(f"{loc}: missing/invalid sub_type {sub!r}")
            else:
                stress_subtypes_seen.add(sub)

        if category in {"out-of-scope", "ambiguous"}:
            if q.get("expected_chunks") is not None:
                warn(f"{loc}: expected_chunks should be null for {category}")

        if category == "factual" and q.get("source"):
            factual_sources[q["source"]] += 1

    for cat, minimum in MIN_COUNTS.items():
        if cat_counts.get(cat, 0) < minimum:
            err(f"category {cat!r} has {cat_counts.get(cat, 0)}, needs at least {minimum}")

    if len(cross_ref_subtypes_seen) < 5:
        err(f"only {len(cross_ref_subtypes_seen)}/6 cross-reference sub_types covered: {sorted(cross_ref_subtypes_seen)}")

    if len(stress_subtypes_seen) < 8:
        err(f"only {len(stress_subtypes_seen)}/11 stress-test sub_types covered: {sorted(stress_subtypes_seen)}")

    if priority_count != 5:
        err(f"exactly 5 questions must be priority:true, found {priority_count}")

    if factual_sources:
        top_source, top_count = factual_sources.most_common(1)[0]
        if top_count / max(sum(factual_sources.values()), 1) > 0.8 and len(factual_sources) > 1:
            warn(f"factual questions heavily concentrated on {top_source!r} ({top_count}/{n})")

    report()


def report():
    print(f"Errors: {len(errors)}")
    for e in errors:
        print(f"  [FAIL] {e}")
    print(f"Warnings: {len(warnings)}")
    for w in warnings:
        print(f"  [WARN] {w}")
    if not errors:
        print("\nAll rubric checks passed.")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
