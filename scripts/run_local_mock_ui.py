"""
Launches the REAL Gradio UI (app/ui.py, unmodified) wired to the mock client
instead of OpenAI. Zero API cost, no key needed. Lets you click through both
normal RAG mode and Agent mode in the browser to see the UI itself render
correctly — answers will be nonsense (fake embeddings, scripted chat replies),
this only proves the wiring, not quality. NOT part of the graded app.

Run: python scripts/run_local_mock_ui.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mock_client import MockOpenAI

from app.embedding import Retriever, load_index
from app.rag import RAG
from app.ui import build_ui
from app.utils import CostTracker

print("Loading real FAISS index...")
index, chunks = load_index()

client = MockOpenAI()
tracker = CostTracker(path=Path("/tmp/mock_cost.json"))

print("Building RAG (loads local re-ranker)...")
rag = RAG(client, Retriever(client, index, chunks, tracker), tracker)

print("\nLaunching UI at http://127.0.0.1:7861 (mock mode — answers are fake)")
build_ui(rag).launch(server_port=7861)
