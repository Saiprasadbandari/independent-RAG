"""
Local-only wiring check for the agent loop — uses scripts/mock_client.py instead
of a real API key. Zero cost, zero real intelligence: this only proves the code
runs end-to-end (tool dispatch, message threading, trajectory logging, step cap),
not that answers are good. NOT part of the graded app.

Run: python scripts/test_agent_local.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mock_client import MockOpenAI

from app.agent import run_agent
from app.embedding import Retriever, load_index
from app.rag import RAG
from app.utils import CostTracker

print("Loading real FAISS index from disk (data/my_index.faiss)...")
index, chunks = load_index()
print(f"  -> {index.ntotal} chunks loaded\n")

client = MockOpenAI()
tracker = CostTracker(path=Path("/tmp/mock_cost.json"))  # separate file, never touches real data/cost.json

print("Building RAG object (loads the local re-ranker — first run downloads "
      "BAAI/bge-reranker-base, ~1.1GB, from Hugging Face, no OpenAI cost)...")
rag = RAG(client, Retriever(client, index, chunks, tracker), tracker)

print("\nRunning the agent loop with the MOCK client (scripted: search -> calculate -> answer)...\n")
answer = run_agent("What is 35000000 doubled?", rag, client, tracker)

print("=" * 70)
print("QUESTION:", answer.question)
print("ANSWER:  ", answer.text)
print("STOPPED AT CAP:", answer.stopped_at_cap)
print("\nTRAJECTORY:")
for i, step in enumerate(answer.trajectory, 1):
    print(f"  {i}. {step.tool}({step.arguments}) -> {step.result[:100]}")
print("=" * 70)
print("\nThis confirms the LOOP MECHANICS work (tool selection, dispatch, message")
print("threading, trajectory logging). It says NOTHING about real answer quality —")
print("the model's 'decisions' here were scripted by mock_client.py, not gpt-4o-mini.")
