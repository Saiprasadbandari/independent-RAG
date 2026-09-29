"""Jev-style decision layer, used here to re-rank retrieved passages.

Jev (TypeSafe AI) answers yes/no questions with a probability instead of generated
text. We get the same shape from gpt-4o-mini: ask a yes/no question, let it produce
exactly one token, and read P(yes) from the token logprobs. The output is always a
number between 0 and 1 — nothing to parse, nothing to hallucinate.

FAISS ranks by embedding similarity, which rewards passages that *sound* like the
question. Asking "does this passage help answer the question?" per passage catches
the ones that are actually useful, and pushes near-misses out of the context.
"""

import math
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from app.utils import CostTracker

DECISION_MODEL = "gpt-4o-mini"
DECISION_SYSTEM = "You are a classifier. Reply with exactly one word: yes or no."
RELEVANCE_QUESTION = "Does the PASSAGE contain information that helps answer the QUESTION?"
MAX_WORKERS = 8


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


def score_relevance(client: OpenAI, tracker: CostTracker, question: str, chunks: list[dict]) -> list[float]:
    """One yes/no decision per passage, run in parallel."""
    tracker.check()
    states = [f"QUESTION:\n{question}\n\nPASSAGE:\n{c['text']}" for c in chunks]
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        results = list(pool.map(lambda s: noul(client, s, RELEVANCE_QUESTION), states))
    # cost is recorded here, not in the worker threads, so cost.json writes don't race
    for _, input_tokens, output_tokens in results:
        tracker.add(DECISION_MODEL, input_tokens, output_tokens)
    return [p for p, _, _ in results]
