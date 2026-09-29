# independent-RAG

Team: ParisAI

Members:
- Rayyan Ahemad Mohd
- Sai Prasad Bandari
- Prodipta Paul

## What it does

A RAG app over the EU AI Act (Regulation (EU) 2024/1689). Questions are answered by
`gpt-4o-mini` using passages retrieved from a FAISS index of `text-embedding-3-small`
embeddings, with document + page citations. Questions outside the corpus get
"I don't have information about that in the provided documents."

## Getting started

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
```

Get the `data/` folder from Teams (it's gitignored). You need at least `data/corpus.json`;
to regenerate it from `data/sources/`, run `python scripts/extract.py`.

Build the index and start the app (run from the repo root):

```bash
python -m app.main --build-only   # embeds corpus.json -> data/my_index.faiss + data/chunks.json
python -m app.main                # starts Gradio on http://localhost:7860
```

The index is built automatically on first launch if it's missing. Use `--rebuild` after the
corpus changes, `--share` for a public link, `--port` to change the port.

## Project layout

```
scripts/extract.py     text extraction from data/sources -> data/corpus.json
app/main.py            entry point: load/build index, start the UI
app/embeddings.py      chunking corpus, embedding, FAISS build/load, retrieval
app/rag.py             retrieval + gpt-4o-mini answer generation with citations
app/ui.py              Gradio interface
app/utils.py           chunking, cost tracking, formatting helpers
```

## Budget

Every API call is priced and logged to `data/cost.json`. Calls are refused once total spend
reaches $5. Embedding the full corpus costs well under $0.01; a new question costs roughly
$0.0004 (re-ranking runs locally for free). Off-topic questions (no passage above the
similarity threshold) skip re-ranking and the LLM call.

## Re-ranking and caching

- **Re-ranking** (`app/rerank.py`): FAISS pulls 15 candidates, then each one gets a relevance
  probability from 0 to 1 for "does this passage help answer the question?" (the same idea as
  TypeSafe's Jev: a decision, not generated text). Passages above the cut-off are kept, best 8
  go to the answer prompt.
  - **Local (default, free):** the `BAAI/bge-reranker-base` cross-encoder runs on your GPU
    (Apple Silicon via MPS, NVIDIA via CUDA, CPU otherwise). The model (~1.1 GB) downloads from
    Hugging Face on first start; after that it loads in ~10 s and scores a question in under a
    second. Cut-off: 0.1.
  - **API fallback:** if `sentence-transformers` isn't installed, gpt-4o-mini makes a one-token
    yes/no decision per passage and P(yes) is read from its logprobs. Costs ~$0.0007 per
    question. Cut-off: 0.5.
- **Caches** (`data/cache/`): repeated questions are answered from the query cache with no API
  calls; the retrieval and re-rank caches skip the embedding and relevance calls when only the
  answer needs regenerating. All caches reset automatically when the index, models, prompts or
  thresholds change. Delete `data/cache/` to clear them by hand.
