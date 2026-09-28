import json
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from app.embeddings import (CHUNKS_PATH, CORPUS_PATH, INDEX_PATH, build_faiss_index,
                            save_index_and_chunks, search)
from app.utils import CHUNK_OVERLAP, CHUNK_SIZE, CostTracker, chunk_text, format_citation

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
EMBED_BATCH_SIZE = 50
TOP_K = 5
MAX_TOKENS = 300
# cosine similarity below this means nothing in the corpus is really about the question
MIN_SCORE = 0.30
NO_INFO = "I don't have information about that in the provided documents."

SYSTEM_PROMPT = f"""You are a helpful assistant that answers questions about the EU AI Act.

Rules:
- Answer ONLY based on the numbered context passages provided. Do not use outside knowledge.
- If the context doesn't contain the answer, reply exactly: "{NO_INFO}"
- Always cite the passage your answer comes from with its number in square brackets, e.g. [1] or [2][3].
- Be concise and precise. Quote article/paragraph numbers when the passage gives them."""

tracker = CostTracker()
_client: OpenAI | None = None


def get_client() -> OpenAI:
    # created lazily so main.py can load .env before the key is read
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ============================================================================
# PART 1: INITIALIZATION & LOADING (offline — run once, saved to disk)
# ============================================================================

def load_corpus(corpus_path: Path = CORPUS_PATH) -> list[dict]:
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"{corpus_path} not found — run scripts/extract.py first.")
    return json.loads(corpus_path.read_text(encoding="utf-8"))


def chunk_corpus(corpus: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    chunks = []
    for entry in corpus:
        for text in chunk_text(entry["text"], chunk_size, overlap):
            chunks.append({"id": len(chunks), "source": entry["source"], "page": entry["page"], "text": text})
    return chunks


def embed_texts(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), batch_size):
        tracker.check()
        resp = get_client().embeddings.create(model=EMBED_MODEL, input=texts[start:start + batch_size])
        track_cost(resp, is_embedding=True)
        vectors.extend(d.embedding for d in resp.data)
    return np.asarray(vectors, dtype="float32")


def embed_chunks(chunks: list[dict], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    return embed_texts([c["text"] for c in chunks], batch_size)


# build_faiss_index and save_index_and_chunks live in app/embeddings.py


# ============================================================================
# PART 2: RETRIEVAL (online — every question)
# ============================================================================

def retrieve(query: str, index: faiss.Index, chunks: list[dict], k: int = TOP_K) -> list[dict]:
    vector = embed_texts([query])
    faiss.normalize_L2(vector)
    return search(index, chunks, vector, k)


# ============================================================================
# PART 3: GENERATION (online — after retrieval)
# ============================================================================

def relevant_chunks(retrieved_chunks: list[dict]) -> list[dict]:
    # FAISS always returns k results, even for off-topic questions
    return [c for c in retrieved_chunks if c["score"] >= MIN_SCORE]


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i}] [Source: {format_citation(c)}]\n{c['text']}" for i, c in enumerate(chunks, 1))


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    relevant = relevant_chunks(retrieved_chunks)
    if not relevant:
        # skip the LLM call entirely — saves budget on off-topic questions
        return NO_INFO

    tracker.check()
    resp = get_client().chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{build_context(relevant)}\n\nQuestion: {query}"},
        ],
    )
    track_cost(resp)
    return resp.choices[0].message.content.strip()


# ============================================================================
# PART 4: COMPLETE PIPELINE
# ============================================================================

def run_rag_pipeline_offline(corpus_path: Path = CORPUS_PATH, output_index_path: Path = INDEX_PATH,
                             output_chunks_path: Path = CHUNKS_PATH) -> tuple[faiss.Index, list[dict]]:
    corpus = load_corpus(corpus_path)
    chunks = chunk_corpus(corpus)
    print(f"embedding {len(chunks)} chunks with {EMBED_MODEL}...")
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)
    save_index_and_chunks(index, chunks, output_index_path, output_chunks_path)
    print(f"✅ Index built and saved! {tracker.summary()}")
    return index, chunks


def run_rag_query_online(query: str, index: faiss.Index, chunks: list[dict]) -> str:
    retrieved = retrieve(query, index, chunks, k=TOP_K)
    return generate_answer(query, retrieved)


@dataclass
class Answer:
    question: str
    text: str
    sources: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    in_scope: bool = True
    cost: float = 0.0


def cited_sources(text: str, chunks: list[dict]) -> list[dict]:
    """Only the passages the model actually cited, de-duplicated by document + page."""
    seen, sources = set(), []
    for i, c in enumerate(chunks, 1):
        key = (c["source"], c["page"])
        if f"[{i}]" in text and key not in seen:
            seen.add(key)
            sources.append({**c, "ref": i})
    return sources


def ask(query: str, index: faiss.Index, chunks: list[dict]) -> Answer:
    """Same as run_rag_query_online, but keeps sources and retrieved chunks for the UI."""
    query = query.strip()
    start = tracker.total
    retrieved = retrieve(query, index, chunks, k=TOP_K)
    text = generate_answer(query, retrieved)
    in_scope = NO_INFO not in text
    return Answer(
        question=query,
        text=text,
        sources=cited_sources(text, relevant_chunks(retrieved)) if in_scope else [],
        retrieved=retrieved,
        in_scope=in_scope,
        cost=tracker.total - start,
    )


# ============================================================================
# PART 5: COST TRACKING ($5 budget)
# ============================================================================

def track_cost(response, is_embedding: bool = False) -> float:
    if is_embedding:
        return tracker.add(EMBED_MODEL, response.usage.prompt_tokens)
    return tracker.add(CHAT_MODEL, response.usage.prompt_tokens, response.usage.completion_tokens)
