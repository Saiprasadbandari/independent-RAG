"""Jev-style decision layer, used here to re-rank retrieved passages.

Jev (TypeSafe AI) answers a yes/no question with a probability instead of generated
text. Here the question is "does this passage help answer the question?", asked of
every FAISS candidate, and the answer is always a number from 0 to 1.

FAISS ranks by embedding similarity, which rewards passages that *sound* like the
question. A relevance decision per passage catches the ones that actually answer it,
and pushes near-misses out of the context.

Two backends, same output:
- LocalReranker (default): a cross-encoder running on this machine's GPU (Apple MPS
  or CUDA, CPU otherwise). Free — no API calls.
- LLMReranker (fallback when sentence-transformers isn't installed): gpt-4o-mini
  limited to one token, P(yes) read from the logprobs. Costs ~$0.0007 per question.
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from app.utils import CostTracker, fingerprint

# silences the fork warning from HF tokenizers when Gradio runs requests in threads
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RERANK_MODEL = "BAAI/bge-reranker-base"
# the cross-encoder reads question + passage together; our ~1000-char chunks fit well under this
RERANK_MAX_LENGTH = 512
RERANK_BATCH_SIZE = 16
# passages need at least this relevance to reach the answer prompt. The cross-encoder's
# scores aren't on the same scale as the LLM's P(yes): relevant passages land anywhere
# from ~0.4 to 1.0, on-topic-but-useless ones under ~0.05.
LOCAL_MIN_RELEVANCE = 0.1
LLM_MIN_RELEVANCE = 0.5

DECISION_MODEL = "gpt-4o-mini"
DECISION_SYSTEM = "You are a classifier. Reply with exactly one word: yes or no."
RELEVANCE_QUESTION = "Does the PASSAGE contain information that helps answer the QUESTION?"
MAX_WORKERS = 8


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalReranker:
    """Cross-encoder on the local GPU; sigmoid output = P(relevant)."""

    def __init__(self, model_name: str = RERANK_MODEL):
        import torch
        from sentence_transformers import CrossEncoder

        self.device = pick_device()
        self.name = f"local:{model_name}"
        self.min_relevance = LOCAL_MIN_RELEVANCE
        self.model = CrossEncoder(model_name, device=self.device, max_length=RERANK_MAX_LENGTH,
                                  activation_fn=torch.nn.Sigmoid())
        print(f"re-ranker: {model_name} on {self.device}")

    def score(self, question: str, chunks: list[dict]) -> list[float]:
        pairs = [(question, c["text"]) for c in chunks]
        return [float(p) for p in self.model.predict(pairs, batch_size=RERANK_BATCH_SIZE)]


def noul(client: OpenAI, state: str, question: str) -> tuple[float, int, int]:
    """P(yes) for a yes/no question about `state`, plus (input, output) tokens used.

    Named after Jev's yes/no primitive. Probabilities are renormalized over the
    "yes" and "no" tokens so stray tokens don't drag the score down.
    """
    resp = client.chat.completions.create(
        model=DECISION_MODEL,
        temperature=0,
        max_tokens=1,
        logprobs=True,
        top_logprobs=10,
        messages=[
            {"role": "system", "content": DECISION_SYSTEM},
            {"role": "user", "content": f"{state}\n\n{question}"},
        ],
    )
    yes = no = 0.0
    for candidate in resp.choices[0].logprobs.content[0].top_logprobs:
        word = candidate.token.strip().lower()
        if word == "yes":
            yes += math.exp(candidate.logprob)
        elif word == "no":
            no += math.exp(candidate.logprob)
    p_yes = yes / (yes + no) if yes + no else 0.0
    return p_yes, resp.usage.prompt_tokens, resp.usage.completion_tokens


class LLMReranker:
    """One gpt-4o-mini yes/no decision per passage, run in parallel."""

    def __init__(self, client: OpenAI, tracker: CostTracker):
        self.client = client
        self.tracker = tracker
        self.name = f"llm:{fingerprint(DECISION_MODEL, DECISION_SYSTEM, RELEVANCE_QUESTION)}"
        self.min_relevance = LLM_MIN_RELEVANCE
        print(f"re-ranker: {DECISION_MODEL} via API (install sentence-transformers to re-rank locally for free)")

    def score(self, question: str, chunks: list[dict]) -> list[float]:
        self.tracker.check()
        states = [f"QUESTION:\n{question}\n\nPASSAGE:\n{c['text']}" for c in chunks]
        with ThreadPoolExecutor(MAX_WORKERS) as pool:
            results = list(pool.map(lambda s: noul(self.client, s, RELEVANCE_QUESTION), states))
        # cost is recorded here, not in the worker threads, so cost.json writes don't race
        for _, input_tokens, output_tokens in results:
            self.tracker.add(DECISION_MODEL, input_tokens, output_tokens)
        return [p for p, _, _ in results]


def load_reranker(client: OpenAI, tracker: CostTracker) -> LocalReranker | LLMReranker:
    try:
        return LocalReranker()
    except ImportError:
        return LLMReranker(client, tracker)
