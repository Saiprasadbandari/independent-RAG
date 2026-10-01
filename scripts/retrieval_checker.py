"""
Retrieval checker — Assignment 2, tool 1 (mandatory, RAG day 1).

Takes a question, runs retrieval, and shows exactly what was found: every
candidate's text, source, page, FAISS/BM25 score, and cross-encoder relevance
after re-ranking — plus whether the expected chunks (from data/questions.json)
were actually retrieved.

Usage:
    python -m scripts.retrieval_checker "What is the fine for prohibited AI practices?"
    python -m scripts.retrieval_checker --question-index 4    # pull a question from questions.json
    python -m scripts.retrieval_checker --mode local --question-index 4   # no API key needed
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from app.embedding import Retriever, load_index
from app.local_models import MODES, make_client
from app.rag import RAG
from app.utils import CostTracker, DATA_DIR, format_citation

QUESTIONS_PATH = DATA_DIR / "questions.json"


def expected_pages(question: dict) -> set[tuple[str, int]]:
    chunks = question.get("expected_chunks") or []
    return {(c["source"], c["page"]) for c in chunks}


def check(rag: RAG, question_text: str, expected: set[tuple[str, int]] | None = None) -> None:
    candidates = rag.retrieve(question_text)
    ranked = rag.rerank(question_text, candidates) if rag.reranker and candidates else candidates

    print(f"\nQUESTION: {question_text}")
    if expected:
        print(f"EXPECTED: {sorted(expected)}")
    print("-" * 100)

    found = set()
    for i, c in enumerate(ranked, 1):
        key = (c["source"], c["page"])
        hit_mark = ""
        if expected:
            hit_mark = " ✅" if key in expected else ""
            if key in expected:
                found.add(key)
        score = c.get("score")
        bm25 = c.get("bm25_score")
        relevance = c.get("relevance")
        score_bits = []
        if c.get("keyword_only"):
            score_bits.append("faiss=— (keyword only)")
        elif score is not None:
            score_bits.append(f"faiss={score:.3f}")
        if bm25 is not None:
            score_bits.append(f"bm25={bm25:.3f}")
        if relevance is not None:
            score_bits.append(f"rerank={relevance:.3f}")
        print(f"[{i}] {format_citation(c)}{hit_mark}  ({', '.join(score_bits)})")
        print(f"    {c['text'][:150].replace(chr(10), ' ')}...")

    if expected:
        missed = expected - found
        print("-" * 100)
        if missed:
            print(f"❌ MISSED: {sorted(missed)}")
        else:
            print("✅ all expected chunks were retrieved")


def build_rag(mode: str = "api") -> RAG:
    load_dotenv()
    client = make_client(mode)
    tracker = CostTracker()
    index, chunks = load_index()
    return RAG(client, Retriever(client, index, chunks, tracker), tracker)


def main():
    parser = argparse.ArgumentParser(description="Inspect retrieval for one question")
    parser.add_argument("question", nargs="?", help="a question to check")
    parser.add_argument("--question-index", type=int, help="pull question N from data/questions.json instead")
    parser.add_argument("--mode", choices=MODES, default="api", help="same run modes as app.main")
    args = parser.parse_args()

    rag = build_rag(args.mode)

    if args.question_index is not None:
        questions = json.loads(QUESTIONS_PATH.read_text())
        q = questions[args.question_index]
        check(rag, q["question"], expected_pages(q))
    elif args.question:
        check(rag, args.question)
    else:
        parser.error("give a question, or --question-index N")


if __name__ == "__main__":
    main()
