import json
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from app.utils import DATA_DIR, CostTracker, chunk_text

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 256

CORPUS_PATH = DATA_DIR / "corpus.json"
INDEX_PATH = DATA_DIR / "my_index.faiss"
CHUNKS_PATH = DATA_DIR / "chunks.json"


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/extract.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def build_chunks(corpus: list[dict]) -> list[dict]:
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


def embed(client: OpenAI, texts: list[str], tracker: CostTracker) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        tracker.check()
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[start:start + BATCH_SIZE])
        tracker.add(EMBED_MODEL, resp.usage.prompt_tokens)
        vectors.extend(d.embedding for d in resp.data)
    arr = np.asarray(vectors, dtype="float32")
    # normalized vectors + inner product index = cosine similarity
    faiss.normalize_L2(arr)
    return arr


def build_index(client: OpenAI, tracker: CostTracker) -> tuple[faiss.Index, list[dict]]:
    chunks = build_chunks(load_corpus())
    print(f"embedding {len(chunks)} chunks with {EMBED_MODEL}...")
    vectors = embed(client, [c["text"] for c in chunks], tracker)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {INDEX_PATH.name} and {CHUNKS_PATH.name} — {tracker.summary()}")
    return index, chunks


def load_index() -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(str(INDEX_PATH))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if index.ntotal != len(chunks):
        raise ValueError("Index and chunks.json are out of sync — rebuild with --rebuild.")
    return index, chunks


def index_exists() -> bool:
    return INDEX_PATH.exists() and CHUNKS_PATH.exists()


class Retriever:
    def __init__(self, client: OpenAI, index: faiss.Index, chunks: list[dict], tracker: CostTracker):
        self.client = client
        self.index = index
        self.chunks = chunks
        self.tracker = tracker

    def search(self, query: str, k: int = 5) -> list[dict]:
        vector = embed(self.client, [query], self.tracker)
        scores, ids = self.index.search(vector, k)
        return [
            {**self.chunks[i], "score": float(s)}
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]
