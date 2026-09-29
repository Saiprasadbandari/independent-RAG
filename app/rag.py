import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from app.embedding import (CHUNKS_PATH, CORPUS_PATH, EMBEDDING_MODEL, INDEX_PATH, Retriever,
                           get_embeddings, load_corpus)
from app.utils import (CACHE_DIR, CHUNK_OVERLAP, CHUNK_SIZE, CostTracker, DiskCache, chunk_text,
                       fingerprint, format_citation, normalize_query, split_sentences)
from app.reflection import REFLECTION_VERSION, assess
from app.rerank import load_reranker

CHAT_MODEL = "gpt-4o-mini"
EMBED_BATCH_SIZE = 50
TOP_K = 8
MAX_TOKENS = 300
# cosine similarity below this means nothing in the corpus is really about the question
MIN_SCORE = 0.30
# FAISS pulls this many candidates; the re-ranker keeps the best TOP_K of them
# (relevance cut-off lives with each re-ranker in app/rerank.py)
RERANK_CANDIDATES = 15
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
# only questions that look like follow-ups are rewritten, so a fresh question is never
# bent towards the previous topic (and most questions skip the rewrite call entirely)
FOLLOW_UP_RE = re.compile(
    r"\b(?:it|its|that|this|these|those|they|them|their|such|same|above|previous|former|latter|"
    r"first|second|third|last|other|another|else|also|what about|how about|and if)\b", re.IGNORECASE)

# How the context is assembled when the local re-ranker is available. Everything here runs
# on the local GPU and exists to send the API fewer, better tokens.
CONTEXT_PASSAGES = 5          # at most this many ranked passages...
RELATIVE_RELEVANCE = 0.4      # ...each scoring at least this fraction of the best one
NEIGHBOUR_TOP = 2             # the best passages also pull in the chunks that follow them,
NEIGHBOUR_STEPS = 6           # up to this far ahead: lists and articles cross chunk boundaries
TRIM_MIN_CHARS = 400          # ranked passages longer than this are trimmed...
SENTENCE_MIN_RELEVANCE = 0.1  # ...to the sentences scoring at least this

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


def answer_messages(query: str, chunks: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{build_context(chunks)}\n\nQuestion: {query}"},
    ]


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
        messages=answer_messages(query, relevant),
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


def stream_answer(query: str, retrieved_chunks: list[dict],
                  client: OpenAI | None = None, tracker: CostTracker | None = None):
    """generate_answer, streamed: yields ("delta", text piece) as the model writes, then
    ("done", text, truncated, input_tokens, output_tokens)."""
    relevant = relevant_chunks(retrieved_chunks)
    if not relevant:
        yield "done", NO_INFO, False, 0, 0
        return

    client, tracker = client or get_client(), tracker or get_tracker()
    tracker.check()
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        max_tokens=MAX_TOKENS,
        messages=answer_messages(query, relevant),
        stream=True,
        stream_options={"include_usage": True},
    )
    pieces, refusal, finish, usage = [], None, None, None
    for chunk in stream:
        usage = chunk.usage or usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.delta.content:
            pieces.append(choice.delta.content)
            yield "delta", choice.delta.content
        refusal = getattr(choice.delta, "refusal", None) or refusal
        finish = choice.finish_reason or finish

    input_tokens, output_tokens = (usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0)
    tracker.add(CHAT_MODEL, input_tokens, output_tokens)
    text = "".join(pieces).strip() or refusal or NO_INFO
    truncated = finish == "length"
    yield "done", text + (TRUNCATED_NOTE if truncated else ""), truncated, input_tokens, output_tokens


def looks_like_follow_up(question: str) -> bool:
    return len(question.split()) <= 3 or bool(FOLLOW_UP_RE.search(question))


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
    # tokens sent to / received from the answer model (0 for cached or local answers)
    input_tokens: int = 0
    output_tokens: int = 0


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
        # local/hybrid clients (app/local_models.py) say which models they really run
        embedding_id = getattr(client, "embedding_model", EMBEDDING_MODEL)
        chat_id = getattr(client, "chat_model", CHAT_MODEL)
        context_id = (CONTEXT_PASSAGES, RELATIVE_RELEVANCE, NEIGHBOUR_TOP, NEIGHBOUR_STEPS,
                      TRIM_MIN_CHARS, SENTENCE_MIN_RELEVANCE)

        # a rebuilt index or a changed model/prompt/threshold invalidates the caches
        index_id = fingerprint([c["text"] for c in retriever.chunks])
        self.retrieval_cache = DiskCache(cache_dir / "retrieval.json",
                                         fingerprint(index_id, embedding_id, RERANK_CANDIDATES))
        self.rerank_cache = DiskCache(cache_dir / "rerank.json", fingerprint(index_id, reranker_id))
        self.query_cache = DiskCache(cache_dir / "answers.json", fingerprint(
            index_id, embedding_id, chat_id, SYSTEM_PROMPT, top_k, MIN_SCORE, MAX_TOKENS,
            RERANK_CANDIDATES, reranker_id, context_id, ANSWER_FORMAT_VERSION, REFLECTION_VERSION,
        ))
        self.rewrite_cache = DiskCache(cache_dir / "rewrites.json", fingerprint(
            chat_id, CONDENSE_PROMPT, HISTORY_TURNS, HISTORY_ANSWER_CHARS, REWRITE_MAX_TOKENS,
        ))
        print(f"cache: {len(self.query_cache)} answers, {len(self.retrieval_cache)} retrievals, "
              f"{len(self.rerank_cache)} re-rankings")

    def retrieve(self, question: str) -> list[dict]:
        key = normalize_query(question)
        hit = self.retrieval_cache.get(key)
        if hit is not None:
            return [{**self.retriever.chunks[i], "score": s} for i, s in hit]
        retrieved = self.retriever.search(question, k=RERANK_CANDIDATES)
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

    def add_neighbours(self, question: str, context: list[dict]) -> list[dict]:
        """After each of the best passages, add the chunks that follow it while they stay
        relevant. A low-scoring chunk between two relevant ones is kept too (it's usually the
        middle of the same list); two low ones in a row end the walk."""
        chunks = self.retriever.chunks
        have = {c["id"] for c in context}
        ahead = {c["id"]: [i for i in range(c["id"] + 1, min(c["id"] + 1 + NEIGHBOUR_STEPS, len(chunks)))
                           if chunks[i]["source"] == c["source"]]
                 for c in context[:NEIGHBOUR_TOP]}
        todo = sorted({i for ids in ahead.values() for i in ids} - have)
        scores = dict(zip(todo, self.reranker.score(question, [chunks[i] for i in todo]))) if todo else {}

        out = []
        for c in context:
            out.append(c)
            pending, misses = [], 0
            for i in ahead.get(c["id"], []):
                if i in have:
                    break
                pending.append(i)
                if scores[i] < self.reranker.min_relevance:
                    misses += 1
                    if misses == 2:
                        break
                    continue
                # neighbours weren't FAISS results; they borrow the similarity of the passage they follow
                out += [{**chunks[j], "score": c["score"], "relevance": scores[j], "neighbour_of": c["id"]}
                        for j in pending]
                have.update(pending)
                pending, misses = [], 0
        return out

    def trim(self, question: str, context: list[dict]) -> list[dict]:
        """Cut each long ranked passage down to its relevant sentences. Neighbours stay whole:
        they're usually the rest of a list, and list items rarely repeat the question's words."""
        sentences = {k: split_sentences(c["text"]) for k, c in enumerate(context)
                     if "neighbour_of" not in c and len(c["text"]) > TRIM_MIN_CHARS}
        sentences = {k: s for k, s in sentences.items() if len(s) > 2}
        flat = [(k, j) for k, s in sentences.items() for j in range(len(s))]
        if not flat:
            return context
        scores = defaultdict(dict)
        for (k, j), score in zip(flat, self.reranker.score(question, [{"text": sentences[k][j]} for k, j in flat])):
            scores[k][j] = score

        out = list(context)
        for k, sents in sentences.items():
            keep = [j for j in range(len(sents)) if scores[k][j] >= SENTENCE_MIN_RELEVANCE]
            keep = keep or sorted(sorted(range(len(sents)), key=lambda j: -scores[k][j])[:2])
            if len(keep) == len(sents):
                continue
            text = sents[keep[0]]
            for prev, j in zip(keep, keep[1:]):
                text += (" " if j == prev + 1 else " … ") + sents[j]
            out[k] = {**context[k], "text": text, "trimmed": True}
        return out

    def select_context(self, question: str, ranked: list[dict]) -> list[dict]:
        if not ranked:
            return []
        floor = max(self.reranker.min_relevance, RELATIVE_RELEVANCE * ranked[0]["relevance"])
        best = [c for c in ranked if c["relevance"] >= floor][:CONTEXT_PASSAGES]
        return self.trim(question, self.add_neighbours(question, best))

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
        for kind, value in self.ask_stream(question, history):
            if kind == "stage":
                stage(value)
            elif kind == "answer":
                return value

    def ask_stream(self, question: str, history: list[tuple[str, str]] | None = None):
        """ask(), as events: ("stage", name), ("delta", answer text so far), and finally
        ("answer", Answer)."""
        question = question.strip()
        start = self.tracker.total

        standalone = question
        if history and looks_like_follow_up(question):
            yield "stage", "rewriting"
            standalone = self.condense(question, history)

        key = normalize_query(standalone)
        hit = self.query_cache.get(key)
        if hit is not None:
            yield "answer", Answer(**{**hit, "question": question, "standalone_question": standalone,
                                      "cost": self.tracker.total - start, "cached": True})
            return

        yield "stage", "searching"
        candidates = self.retrieve(standalone)
        # only re-rank passages that are at least loosely similar; off-topic stops here
        plausible = relevant_chunks(candidates)
        if self.reranker and plausible:
            yield "stage", "reranking"
            ranked = self.rerank(standalone, plausible)
            context = self.select_context(standalone, ranked)
        else:
            ranked = plausible
            context = plausible[:self.top_k]

        yield "stage", "generating"
        so_far = ""
        for event in stream_answer(standalone, context, self.client, self.tracker):
            if event[0] == "delta":
                so_far += event[1]
                yield "delta", so_far
            else:
                _, text, truncated, input_tokens, output_tokens = event
        yield "stage", "checking"
        in_scope = not is_refusal(text)
        sources = cited_sources(text, context) if in_scope else []
        # the passages the model saw first, then every other candidate once
        retrieved, seen = list(context), {c["id"] for c in context}
        for c in ranked + candidates:
            if c["id"] not in seen:
                seen.add(c["id"])
                retrieved.append(c)
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
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        # a cut-off answer isn't worth keeping; asking again might fit (e.g. after a prompt tweak)
        if not truncated:
            self.query_cache.set(key, asdict(answer))
        yield "answer", answer
