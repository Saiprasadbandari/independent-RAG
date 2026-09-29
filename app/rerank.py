"""Re-ranks retrieved passages with a cross-encoder running on the local GPU.

FAISS ranks by embedding similarity, which rewards passages that *sound* like the
question. A cross-encoder reads the question and passage together and scores how
well the passage answers it (0 to 1), which catches the ones that are actually
useful and pushes near-misses out of the context.

Runs on Apple Silicon (MPS), NVIDIA (CUDA) or CPU. Free — no API calls. If
sentence-transformers isn't installed, re-ranking is skipped and FAISS order is used.
"""

import os

# silences the fork warning from HF tokenizers when Gradio runs requests in threads
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RERANK_MODEL = "BAAI/bge-reranker-base"
# the cross-encoder reads question + passage together; our ~1000-char chunks fit well under this
RERANK_MAX_LENGTH = 512
RERANK_BATCH_SIZE = 16
# passages need at least this relevance to reach the answer prompt. Relevant passages land
# anywhere from ~0.4 to 1.0, on-topic-but-useless ones under ~0.05.
MIN_RELEVANCE = 0.1


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
        self.min_relevance = MIN_RELEVANCE
        self.model = CrossEncoder(model_name, device=self.device, max_length=RERANK_MAX_LENGTH,
                                  activation_fn=torch.nn.Sigmoid())
        print(f"re-ranker: {model_name} on {self.device}")

    def score(self, question: str, chunks: list[dict]) -> list[float]:
        pairs = [(question, c["text"]) for c in chunks]
        return [float(p) for p in self.model.predict(pairs, batch_size=RERANK_BATCH_SIZE)]


def load_reranker() -> LocalReranker | None:
    try:
        return LocalReranker()
    except ImportError:
        print("sentence-transformers not installed — re-ranking off, using FAISS order.")
        return None
