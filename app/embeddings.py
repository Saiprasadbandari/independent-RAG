import json
from pathlib import Path

import faiss
import numpy as np

from app.utils import DATA_DIR

CORPUS_PATH = DATA_DIR / "corpus.json"
INDEX_PATH = DATA_DIR / "my_index.faiss"
CHUNKS_PATH = DATA_DIR / "chunks.json"


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    vectors = np.ascontiguousarray(embeddings, dtype="float32")
    # normalized vectors + inner product index = cosine similarity
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index_and_chunks(index: faiss.Index, chunks: list[dict],
                          index_path: Path = INDEX_PATH, chunks_path: Path = CHUNKS_PATH) -> None:
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    Path(chunks_path).write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(index_path: Path = INDEX_PATH, chunks_path: Path = CHUNKS_PATH) -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(str(index_path))
    chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
    if index.ntotal != len(chunks):
        raise ValueError("Index and chunks.json are out of sync — rebuild with --rebuild.")
    return index, chunks


def index_exists(index_path: Path = INDEX_PATH, chunks_path: Path = CHUNKS_PATH) -> bool:
    return Path(index_path).exists() and Path(chunks_path).exists()


def search(index: faiss.Index, chunks: list[dict], query_vector: np.ndarray, k: int) -> list[dict]:
    """query_vector must already be normalized; results come back highest score first."""
    scores, ids = index.search(query_vector, k)
    return [
        {**chunks[i], "score": float(s)}
        for s, i in zip(scores[0], ids[0])
        if i != -1
    ]
