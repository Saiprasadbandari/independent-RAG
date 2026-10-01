"""Clients for the three run modes, so the heavy work can stay on this machine's GPU.

- api:    everything through OpenAI (text-embedding-3-small + gpt-4o-mini)
- hybrid: embeddings on the local GPU (free), answers from gpt-4o-mini (fast, streamed)
- local:  embeddings and answers on the local GPU, no API key needed (slower, weaker answers)

The local pieces mimic the two OpenAI calls the app makes, embeddings.create and
chat.completions.create (including stream=True), so the rest of the app doesn't care
which mode it runs in. Local calls report 0 tokens, so they cost $0 in the tracker.
"""

from threading import Thread
from types import SimpleNamespace as NS

import app.embedding as embedding
from app.rerank import pick_device
from app.utils import DATA_DIR

LOCAL_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
LOCAL_CHAT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MODES = ("api", "hybrid", "local")


class LocalEmbeddings:
    def __init__(self, model_name: str = LOCAL_EMBED_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=pick_device())

    def create(self, model: str, input: list[str]):
        vectors = self.model.encode(list(input), batch_size=32, normalize_embeddings=True)
        return NS(data=[NS(embedding=v.tolist()) for v in vectors], usage=NS(prompt_tokens=0))


class LocalChatCompletions:
    def __init__(self, model_name: str = LOCAL_CHAT_MODEL):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(self.device).eval()

    def create(self, model: str, messages: list[dict], max_tokens: int = 300, temperature: float = 0,
               stream: bool = False, **_):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        options = dict(**inputs, max_new_tokens=max_tokens, do_sample=False,
                       pad_token_id=self.tokenizer.eos_token_id)
        if stream:
            return self._stream(options, max_tokens)

        import torch

        with torch.no_grad():
            out = self.model.generate(**options)
        new = out[0][inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(new, skip_special_tokens=True).strip()
        finish = "length" if len(new) >= max_tokens else "stop"
        return NS(choices=[NS(message=NS(content=text, refusal=None), finish_reason=finish)],
                  usage=NS(prompt_tokens=0, completion_tokens=0))

    def _stream(self, options: dict, max_tokens: int):
        """Yields chunks shaped like OpenAI's streaming chunks."""
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        Thread(target=self.model.generate, kwargs={**options, "streamer": streamer}, daemon=True).start()
        pieces = 0
        for piece in streamer:
            pieces += 1
            yield NS(choices=[NS(delta=NS(content=piece, refusal=None), finish_reason=None)], usage=None)
        # the streamer yields roughly a word at a time, so this is an estimate of hitting the cap
        finish = "length" if pieces >= max_tokens * 0.9 else "stop"
        yield NS(choices=[NS(delta=NS(content=None, refusal=None), finish_reason=finish)], usage=None)
        yield NS(choices=[], usage=NS(prompt_tokens=0, completion_tokens=0))


def make_client(mode: str):
    """The client for a run mode. Local embeddings get their own index files, because their
    vectors (768-d) don't fit an index built with OpenAI embeddings (1536-d)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode == "api":
        from openai import OpenAI

        return OpenAI()

    embedding.INDEX_PATH = DATA_DIR / "my_index-local.faiss"
    embedding.CHUNKS_PATH = DATA_DIR / "chunks-local.json"
    if mode == "hybrid":
        from openai import OpenAI

        chat, chat_id = OpenAI().chat, "gpt-4o-mini"
    else:
        chat, chat_id = NS(completions=LocalChatCompletions()), LOCAL_CHAT_MODEL
    # read by RAG so caches never mix answers from different embedding or chat models
    return NS(embeddings=LocalEmbeddings(), chat=chat,
              embedding_model=LOCAL_EMBED_MODEL, chat_model=chat_id)
