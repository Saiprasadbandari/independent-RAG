import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from app.embedding import (CHUNKS_PATH, CORPUS_PATH, EMBEDDING_MODEL, INDEX_PATH, Retriever,
                           get_embeddings, load_corpus)
from app.utils import (CACHE_DIR, CHUNK_OVERLAP, CHUNK_SIZE, CostTracker, DiskCache, chunk_text,
                       fingerprint, format_citation, normalize_query)
from app.reflection import REFLECTION_VERSION, assess
from app.rerank import load_reranker
from app.hybrid import BM25Index, reciprocal_rank_fusion

CHAT_MODEL = "gpt-4o-mini"
EMBED_BATCH_SIZE = 50
TOP_K = 8
MAX_TOKENS = 300
# cosine similarity below this means nothing in the corpus is really about the question
MIN_SCORE = 0.30
# FAISS pulls this many candidates; the re-ranker keeps the best TOP_K of them
# (relevance cut-off lives with each re-ranker in app/rerank.py)
RERANK_CANDIDATES = 15
# hybrid search: merge FAISS with BM25 keyword search via RRF before re-ranking.
# Our corpus is a legal text full of exact terms (article numbers, "Annex III", etc.)
# that pure vector search can miss — see the "exact_term" stress-test category.
USE_BM25 = True
BM25_CANDIDATES = 15
NO_INFO = "I don't have information about that in the provided documents."
TRUNCATED_NOTE = "\n\n*(Answer cut off at the length limit.)*"
# bump when answer parsing changes so cached answers get re-parsed
ANSWER_FORMAT_VERSION = 2

# "[2]", "[1, 3]", "[2-4]" — brackets holding only passage numbers
REF_BRACKET_RE = re.compile(r"\[\s*(\d+(?:\s*[-–,]\s*\d+)*)\s*\]")
# "[Source: EU AI Regulations.pdf, p. 9]" — the label from build_context, copied by the model
SOURCE_LABEL_RE = re.compile(r"\[Source:\s*(.+?),\s*p\.\s*(\d+)\s*\]")
# follow-ups are rewritten into a standalone question using this many previous turns
HISTORY_TURNS = 3
HISTORY_ANSWER_CHARS = 600
REWRITE_MAX_TOKENS = 100
CONDENSE_PROMPT = """Rewrite the user's latest question so it can be understood without the conversation.
Resolve references such as "it", "that", "the second requirement" using the conversation.
Keep everything else as the user wrote it. If the question is already standalone, return it unchanged.
Reply with the rewritten question only."""
# progress stages RAG.ask reports through on_stage, in order
STAGES = ("rewriting", "searching", "reranking", "generating", "checking")

REFUSAL_RE = re.compile(
    r"\b(?:don't|do not|doesn't|does not|cannot|can't)\s+(?:have|contain|provide|include|find)\b"
    r"[^.]*\binformation\b", re.IGNORECASE)

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
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        # the model declined outright; newer models put the reason in .refusal, not .content
        return getattr(choice.message, "refusal", None) or NO_INFO
    if choice.finish_reason == "length":
        text += TRUNCATED_NOTE
    return text


def history_answer(answer: str) -> str:
    """An earlier answer as context for rewriting: no citation markers, trimmed."""
    answer = re.sub(r"\s*\[[^\]]*\]", "", answer.replace(TRUNCATED_NOTE, ""))
    return answer[:HISTORY_ANSWER_CHARS]


def condense_question(question: str, history: list[tuple[str, str]],
                      client: OpenAI | None = None, tracker: CostTracker | None = None) -> str:
    """Rewrite a follow-up ("what about the second one?") into a standalone question.

    history: previous (question, answer) turns, oldest first.
    """
    if not history:
        return question
    client, tracker = client or get_client(), tracker or get_tracker()
    turns = "\n".join(f"User: {q}\nAssistant: {history_answer(a)}" for q, a in history[-HISTORY_TURNS:])
    tracker.check()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        max_tokens=REWRITE_MAX_TOKENS,
        messages=[
            {"role": "system", "content": CONDENSE_PROMPT},
            {"role": "user", "content": f"Conversation:\n{turns}\n\nLatest question: {question}"},
        ],
    )
    track_cost(resp, tracker=tracker)
    return (resp.choices[0].message.content or "").strip().strip('"') or question


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
    # what was actually searched for: the question itself, or a follow-up rewritten using history
    standalone_question: str = ""
    sources: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    in_scope: bool = True
    cost: float = 0.0
    cached: bool = False
    truncated: bool = False
    # the first context_size entries of `retrieved` are the passages the model was given
    context_size: int = 0
    confidence: float = 0.0
    confidence_label: str = ""
    confidence_signals: dict = field(default_factory=dict)
    confidence_notes: list[str] = field(default_factory=list)


def cited_refs(text: str, chunks: list[dict]) -> tuple[list[int], list[int]]:
    """Passage numbers the answer cites, split into (valid, out of range)."""
    refs = set()
    for match in REF_BRACKET_RE.finditer(text):
        for part in match.group(1).split(","):
            ends = [int(n) for n in re.split(r"[-–]", part)]
            refs.update(range(ends[0], ends[-1] + 1) if len(ends) == 2 else ends)
    for source, page in SOURCE_LABEL_RE.findall(text):
        refs.update(i for i, c in enumerate(chunks, 1)
                    if c["source"] == source.strip() and c["page"] == int(page))
    valid = sorted(r for r in refs if 1 <= r <= len(chunks))
    return valid, sorted(refs - set(valid))


def cited_sources(text: str, chunks: list[dict]) -> list[dict]:
    """Only the passages the model actually cited, de-duplicated by document + page."""
    seen, sources = set(), []
    for i in cited_refs(text, chunks)[0]:
        c = chunks[i - 1]
        key = (c["source"], c["page"])
        if key not in seen:
            seen.add(key)
            sources.append({**c, "ref": i})
    return sources


def is_refusal(text: str) -> bool:
    """A reply that declines without citing anything. A partial answer that cites passages
    and then says part of the question isn't covered still counts as an answer."""
    if REF_BRACKET_RE.search(text) or SOURCE_LABEL_RE.search(text):
        return False
    return normalize_query(NO_INFO) in normalize_query(text) or bool(REFUSAL_RE.search(text))


class RAG:
    """Online pipeline with caches in front of every API call:

    1. query cache     — question seen before? return the stored answer, zero API calls
    2. retrieval cache — question retrieved before? reuse the candidates, skip the embedding call
    3. rerank cache    — passage already judged for this question? reuse its relevance
    4. re-rank: relevance probability per candidate (app/rerank.py, local GPU), best TOP_K kept
    5. generate with gpt-4o-mini from the re-ranked passages
    """

    def __init__(self, client: OpenAI, retriever: Retriever, tracker: CostTracker, top_k: int = TOP_K,
                 cache_dir: Path = CACHE_DIR):
        self.client = client
        self.retriever = retriever
        self.tracker = tracker
        self.top_k = top_k
        self.reranker = load_reranker()
        reranker_id = (self.reranker.name, self.reranker.min_relevance) if self.reranker else None
        self.bm25 = BM25Index(retriever.chunks) if USE_BM25 else None

        # a rebuilt index or a changed model/prompt/threshold invalidates the caches
        index_id = fingerprint([c["text"] for c in retriever.chunks])
        self.retrieval_cache = DiskCache(cache_dir / "retrieval.json",
                                         fingerprint(index_id, EMBEDDING_MODEL, RERANK_CANDIDATES,
                                                     USE_BM25, BM25_CANDIDATES))
        self.rerank_cache = DiskCache(cache_dir / "rerank.json", fingerprint(index_id, reranker_id))
        self.query_cache = DiskCache(cache_dir / "answers.json", fingerprint(
            index_id, EMBEDDING_MODEL, CHAT_MODEL, SYSTEM_PROMPT, top_k, MIN_SCORE, MAX_TOKENS,
            RERANK_CANDIDATES, reranker_id, ANSWER_FORMAT_VERSION, REFLECTION_VERSION,
        ))
        self.rewrite_cache = DiskCache(cache_dir / "rewrites.json", fingerprint(
            CHAT_MODEL, CONDENSE_PROMPT, HISTORY_TURNS, HISTORY_ANSWER_CHARS, REWRITE_MAX_TOKENS,
        ))
        print(f"cache: {len(self.query_cache)} answers, {len(self.retrieval_cache)} retrievals, "
              f"{len(self.rerank_cache)} re-rankings")

    def retrieve(self, question: str) -> list[dict]:
        key = normalize_query(question)
        hit = self.retrieval_cache.get(key)
        if hit is not None:
            return [{**self.retriever.chunks[i], "score": s} for i, s in hit]

        vector_hits = self.retriever.search(question, k=RERANK_CANDIDATES)
        if self.bm25:
            keyword_hits = self.bm25.search(question, k=BM25_CANDIDATES)
            retrieved = reciprocal_rank_fusion(vector_hits, keyword_hits)
            # a BM25-only hit has no cosine "score" — a keyword match is itself a real
            # relevance signal, so let it clear the pre-rerank MIN_SCORE filter too
            for c in retrieved:
                c.setdefault("score", MIN_SCORE)
        else:
            retrieved = vector_hits

        self.retrieval_cache.set(key, [[c["id"], c["score"]] for c in retrieved])
        return retrieved

    def rerank(self, question: str, candidates: list[dict]) -> list[dict]:
        """Candidates with a "relevance" score attached, most relevant first."""
        key = normalize_query(question)
        judged = dict(self.rerank_cache.get(key) or {})
        unjudged = [c for c in candidates if str(c["id"]) not in judged]
        if unjudged:
            relevance = self.reranker.score(question, unjudged)
            judged.update({str(c["id"]): p for c, p in zip(unjudged, relevance)})
            self.rerank_cache.set(key, judged)
        ranked = [{**c, "relevance": judged[str(c["id"])]} for c in candidates]
        return sorted(ranked, key=lambda c: c["relevance"], reverse=True)

    def condense(self, question: str, history: list[tuple[str, str]]) -> str:
        key = fingerprint(history[-HISTORY_TURNS:], normalize_query(question))
        hit = self.rewrite_cache.get(key)
        if hit is not None:
            return hit
        standalone = condense_question(question, history, self.client, self.tracker)
        self.rewrite_cache.set(key, standalone)
        return standalone

    def ask(self, question: str, history: list[tuple[str, str]] | None = None,
            on_stage=None) -> Answer:
        """Same as run_rag_query_online, but keeps sources and retrieved chunks for the UI.

        history: previous (question, answer) turns, so follow-ups like "what about the second
        requirement?" work. on_stage: called with each name in STAGES as the pipeline reaches it.
        """
        stage = on_stage or (lambda name: None)
        question = question.strip()
        start = self.tracker.total

        standalone = question
        if history:
            stage("rewriting")
            standalone = self.condense(question, history)

        key = normalize_query(standalone)
        hit = self.query_cache.get(key)
        if hit is not None:
            return Answer(**{**hit, "question": question, "standalone_question": standalone,
                             "cost": self.tracker.total - start, "cached": True})

        stage("searching")
        candidates = self.retrieve(standalone)
        # only re-rank passages that are at least loosely similar; off-topic stops here
        plausible = relevant_chunks(candidates)
        if self.reranker and plausible:
            stage("reranking")
            ranked = self.rerank(standalone, plausible)
            context = [c for c in ranked if c["relevance"] >= self.reranker.min_relevance][:self.top_k]
        else:
            ranked = plausible
            context = plausible[:self.top_k]

        stage("generating")
        text = generate_answer(standalone, context, self.client, self.tracker)
        stage("checking")
        truncated = text.endswith(TRUNCATED_NOTE)
        in_scope = not is_refusal(text)
        sources = cited_sources(text, context) if in_scope else []
        retrieved = ranked + [c for c in candidates if c not in plausible]
        confidence = assess(text, in_scope, truncated, context, retrieved, sources,
                            cited_refs(text, context)[1], self.reranker)
        answer = Answer(
            question=question,
            text=text,
            standalone_question=standalone,
            sources=sources,
            retrieved=retrieved,
            in_scope=in_scope,
            cost=self.tracker.total - start,
            truncated=truncated,
            context_size=len(context),
            confidence=confidence.score,
            confidence_label=confidence.label,
            confidence_signals=confidence.signals,
            confidence_notes=confidence.notes,
        )
        # a cut-off answer isn't worth keeping; asking again might fit (e.g. after a prompt tweak)
        if not truncated:
            self.query_cache.set(key, asdict(answer))
        return answer
