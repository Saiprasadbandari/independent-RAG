import argparse

from dotenv import load_dotenv
from openai import OpenAI

from app.embeddings import Retriever, build_index, index_exists, load_index
from app.rag import RAG
from app.ui import build_ui
from app.utils import CostTracker


def parse_args():
    parser = argparse.ArgumentParser(description="EU AI Act RAG app")
    parser.add_argument("--rebuild", action="store_true", help="re-embed corpus.json and rebuild the FAISS index")
    parser.add_argument("--build-only", action="store_true", help="build the index and exit without starting the UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()
    client = OpenAI()
    tracker = CostTracker()

    if args.rebuild or args.build_only or not index_exists():
        index, chunks = build_index(client, tracker)
    else:
        index, chunks = load_index()
        print(f"loaded index with {index.ntotal} chunks — {tracker.summary()}")

    if args.build_only:
        return

    rag = RAG(client, Retriever(client, index, chunks, tracker), tracker)
    build_ui(rag).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
