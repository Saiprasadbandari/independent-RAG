"""
Entry point for the RAG app.

Usage:
    python -m app.main                 # load the saved index and launch the UI
    python -m app.main --rebuild       # re-embed the corpus, rebuild the index, then launch
    python -m app.main --build-only    # just build/refresh the index, don't launch the UI
    python -m app.main --mode hybrid   # embeddings on the local GPU, answers from gpt-4o-mini
    python -m app.main --mode local    # everything on the local GPU, no API key needed

Note: this imports app.rag and app.ui, which live on the rag-app branch —
this file is only fully runnable once that branch is merged in alongside
this one.
"""

import argparse

from dotenv import load_dotenv

from app.embedding import Retriever, build_index, index_exists, load_index
from app.injection import INJECTION_TEST, add_test_chunk
from app.local_models import MODES, make_client
from app.rag import RAG
import gradio as gr

from app.ui import MINIMAL_CSS, build_ui
from app.utils import CostTracker


def parse_args():
    parser = argparse.ArgumentParser(description="EU AI Act RAG app")
    parser.add_argument("--rebuild", action="store_true",
                         help="re-embed data/corpus.json and rebuild the FAISS index")
    parser.add_argument("--build-only", action="store_true",
                         help="build the index and exit without starting the UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    parser.add_argument("--mode", choices=MODES, default="api",
                        help="api: all OpenAI; hybrid: local embeddings + gpt-4o-mini answers; "
                             "local: no API key (see app/local_models.py)")
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()
    client = make_client(args.mode)
    tracker = CostTracker()

    if args.rebuild or args.build_only or not index_exists():
        index, chunks = build_index(client, tracker)
    else:
        index, chunks = load_index()
        print(f"loaded index with {index.ntotal} chunks — {tracker.summary()}")

    if args.build_only:
        return

    if INJECTION_TEST:
        index, chunks = add_test_chunk(client, index, chunks, tracker)
        print("INJECTION_TEST is on: one poisoned passage added to the in-memory index")

    retriever = Retriever(client, index, chunks, tracker)
    rag = RAG(client, retriever, tracker)
    build_ui(rag).launch(server_port=args.port, share=args.share, css=MINIMAL_CSS, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
