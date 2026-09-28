"""
Embedding + indexing + retrieval.

This is my part of the pipeline: turn data/corpus.json into a searchable
FAISS index (built once, saved to disk), and provide a Retriever that
embeds a user's question and finds the top-k closest chunks.

Follows the professor's code reference (Sections 5 & 6):
  - client.embeddings.create(model="text-embedding-3-small", ...)
  - faiss.IndexFlatIP + normalize_L2 (inner product on normalized
    vectors == cosine similarity)
"""

import json
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from app.utils import DATA_DIR, CostTracker, chunk_text

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

CORPUS_PATH = DATA_DIR / "corpus.json"
INDEX_PATH = DATA_DIR / "my_index.faiss"
CHUNKS_PATH = DATA_DIR / "chunks.json"


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    """Load the extracted corpus (Assignment 0's output)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/extract.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def build_chunks(corpus: list[dict]) -> list[dict]:
    """Split every corpus entry into overlapping chunks, keeping the
    source + page metadata attached to each piece."""
    chunks = []
    for entry in corpus:
        for text in chunk_text(entry["text"]):
            chunks.append({
                "id": len(chunks),
                "source": entry["source"],
                "page": entry["page"],
                "text": text,
            })
    return chunks


def get_embeddings(client: OpenAI, texts: list[str], tracker: CostTracker) -> np.ndarray:
    """Embed a list of texts in batches, normalize for cosine similarity,
    and track the cost of every call. Works for a single query (list of
    length 1) or the whole corpus."""
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        tracker.check()
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        tracker.add(EMBEDDING_MODEL, response.usage.prompt_tokens)
        vectors.extend(item.embedding for item in response.data)

    matrix = np.array(vectors, dtype="float32")
    faiss.normalize_L2(matrix)
    return matrix


def build_index(client: OpenAI, tracker: CostTracker) -> tuple[faiss.Index, list[dict]]:
    """Do the one-time prep: chunk the corpus, embed every chunk, build
    the FAISS index, and save both the index and the chunks to disk."""
    chunks = build_chunks(load_corpus())
    print(f"embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")

    vectors = get_embeddings(client, [c["text"] for c in chunks], tracker)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {INDEX_PATH.name} and {CHUNKS_PATH.name} — {tracker.summary()}")
    return index, chunks


def index_exists() -> bool:
    return INDEX_PATH.exists() and CHUNKS_PATH.exists()


def load_index() -> tuple[faiss.Index, list[dict]]:
    """Load a previously built index — everyone but the person who ran
    build_index() should use this instead of re-embedding."""
    index = faiss.read_index(str(INDEX_PATH))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if index.ntotal != len(chunks):
        raise ValueError("Index and chunks.json are out of sync — rebuild with --rebuild.")
    return index, chunks


class Retriever:
    """Embeds a question and searches the FAISS index for the top-k
    closest chunks. This is what app/rag.py calls on every question."""

    def __init__(self, client: OpenAI, index: faiss.Index, chunks: list[dict], tracker: CostTracker):
        self.client = client
        self.index = index
        self.chunks = chunks
        self.tracker = tracker

    def search(self, query: str, k: int = 5) -> list[dict]:
        query_vector = get_embeddings(self.client, [query], self.tracker)
        scores, ids = self.index.search(query_vector, k)
        return [
            {**self.chunks[i], "score": float(score)}
            for score, i in zip(scores[0], ids[0])
            if i != -1
        ]
