import json
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from app.embedding import (CHUNKS_PATH, CORPUS_PATH, EMBEDDING_MODEL, INDEX_PATH, Retriever,
                           get_embeddings, load_corpus)
from app.utils import CHUNK_OVERLAP, CHUNK_SIZE, CostTracker, chunk_text, format_citation

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

# Defaults for the standalone pipeline functions below. main.py passes its own
# client and tracker through the RAG class, so these are only created if needed.
_client: OpenAI | None = None
_tracker: CostTracker | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def get_tracker() -> CostTracker:
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker


# ============================================================================
# PART 1: INITIALIZATION & LOADING (offline — run once, saved to disk)
# ============================================================================

# load_corpus is imported from app.embedding


def chunk_corpus(corpus: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    chunks = []
    for entry in corpus:
        for text in chunk_text(entry["text"], chunk_size, overlap):
            chunks.append({"id": len(chunks), "source": entry["source"], "page": entry["page"], "text": text})
    return chunks


def embed_chunks(chunks: list[dict], batch_size: int = EMBED_BATCH_SIZE,
                 client: OpenAI | None = None, tracker: CostTracker | None = None) -> np.ndarray:
    client, tracker = client or get_client(), tracker or get_tracker()
    texts = [c["text"] for c in chunks]
    batches = [get_embeddings(client, texts[i:i + batch_size], tracker) for i in range(0, len(texts), batch_size)]
    return np.vstack(batches)


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


# ============================================================================
# PART 2: RETRIEVAL (online — every question)
# ============================================================================

def retrieve(query: str, index: faiss.Index, chunks: list[dict], k: int = TOP_K,
             client: OpenAI | None = None, tracker: CostTracker | None = None) -> list[dict]:
    # Retriever embeds + normalizes the query and returns results highest score first
    return Retriever(client or get_client(), index, chunks, tracker or get_tracker()).search(query, k)


# ============================================================================
# PART 3: GENERATION (online — after retrieval)
# ============================================================================

def relevant_chunks(retrieved_chunks: list[dict]) -> list[dict]:
    # FAISS always returns k results, even for off-topic questions
    return [c for c in retrieved_chunks if c["score"] >= MIN_SCORE]


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i}] [Source: {format_citation(c)}]\n{c['text']}" for i, c in enumerate(chunks, 1))


def generate_answer(query: str, retrieved_chunks: list[dict],
                    client: OpenAI | None = None, tracker: CostTracker | None = None) -> str:
    relevant = relevant_chunks(retrieved_chunks)
    if not relevant:
        # skip the LLM call entirely — saves budget on off-topic questions
        return NO_INFO

    client, tracker = client or get_client(), tracker or get_tracker()
    tracker.check()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{build_context(relevant)}\n\nQuestion: {query}"},
        ],
    )
    track_cost(resp, tracker=tracker)
    return resp.choices[0].message.content.strip()


# ============================================================================
# PART 4: COMPLETE PIPELINE
# ============================================================================

def run_rag_pipeline_offline(corpus_path: Path = CORPUS_PATH, output_index_path: Path = INDEX_PATH,
                             output_chunks_path: Path = CHUNKS_PATH) -> tuple[faiss.Index, list[dict]]:
    corpus = load_corpus(Path(corpus_path))
    chunks = chunk_corpus(corpus)
    print(f"embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)
    save_index_and_chunks(index, chunks, output_index_path, output_chunks_path)
    print(f"✅ Index built and saved! {get_tracker().summary()}")
    return index, chunks


def run_rag_query_online(query: str, index: faiss.Index, chunks: list[dict]) -> str:
    retrieved = retrieve(query, index, chunks, k=TOP_K)
    return generate_answer(query, retrieved)


# ============================================================================
# PART 5: COST TRACKING ($5 budget)
# ============================================================================

def track_cost(response, is_embedding: bool = False, tracker: CostTracker | None = None) -> float:
    tracker = tracker or get_tracker()
    if is_embedding:
        return tracker.add(EMBEDDING_MODEL, response.usage.prompt_tokens)
    return tracker.add(CHAT_MODEL, response.usage.prompt_tokens, response.usage.completion_tokens)


# ============================================================================
# App interface — used by main.py and ui.py
# ============================================================================

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


class RAG:
    def __init__(self, client: OpenAI, retriever: Retriever, tracker: CostTracker, top_k: int = TOP_K):
        self.client = client
        self.retriever = retriever
        self.tracker = tracker
        self.top_k = top_k

    def ask(self, question: str) -> Answer:
        """Same as run_rag_query_online, but keeps sources and retrieved chunks for the UI."""
        question = question.strip()
        start = self.tracker.total
        retrieved = self.retriever.search(question, k=self.top_k)
        text = generate_answer(question, retrieved, self.client, self.tracker)
        in_scope = NO_INFO not in text
        return Answer(
            question=question,
            text=text,
            sources=cited_sources(text, relevant_chunks(retrieved)) if in_scope else [],
            retrieved=retrieved,
            in_scope=in_scope,
            cost=self.tracker.total - start,
        )
