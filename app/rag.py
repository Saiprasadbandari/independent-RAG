from dataclasses import dataclass, field

from openai import OpenAI

from app.embedding import Retriever
from app.utils import CostTracker, format_citation

CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
# cosine similarity below this means nothing in the corpus is really about the question
MIN_SCORE = 0.30
NO_INFO = "I don't have information about that in the provided documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered context passages provided.

Rules:
- Cite every claim with the passage number in square brackets, e.g. [1] or [2][3].
- Do not use outside knowledge. If the passages do not contain the answer, reply exactly:
  "{NO_INFO}"
- Be concise and precise. Quote article/paragraph numbers when the passage gives them."""


@dataclass
class Answer:
    question: str
    text: str
    sources: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    in_scope: bool = True
    cost: float = 0.0


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i}] ({format_citation(c)})\n{c['text']}" for i, c in enumerate(chunks, 1))


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
    def __init__(self, client: OpenAI, retriever: Retriever, tracker: CostTracker,
                 top_k: int = TOP_K, min_score: float = MIN_SCORE):
        self.client = client
        self.retriever = retriever
        self.tracker = tracker
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, question: str) -> list[dict]:
        return self.retriever.search(question, k=self.top_k)

    def generate(self, question: str, chunks: list[dict]) -> tuple[str, float]:
        self.tracker.check()
        resp = self.client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{build_context(chunks)}\n\nQuestion: {question}"},
            ],
        )
        cost = self.tracker.add(CHAT_MODEL, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return resp.choices[0].message.content.strip(), cost

    def ask(self, question: str) -> Answer:
        question = question.strip()
        start = self.tracker.total
        retrieved = self.retrieve(question)

        relevant = [c for c in retrieved if c["score"] >= self.min_score]
        if not relevant:
            # skip the LLM call entirely — saves budget on off-topic questions
            return Answer(question, NO_INFO, retrieved=retrieved, in_scope=False,
                          cost=self.tracker.total - start)

        text, _ = self.generate(question, relevant)
        in_scope = NO_INFO not in text
        return Answer(
            question=question,
            text=text,
            sources=cited_sources(text, relevant) if in_scope else [],
            retrieved=retrieved,
            in_scope=in_scope,
            cost=self.tracker.total - start,
        )
