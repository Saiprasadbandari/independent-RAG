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
reaches $5. Embedding the full corpus costs well under $0.01; each question is roughly
$0.0005. Off-topic questions (no passage above the similarity threshold) skip the LLM call.
