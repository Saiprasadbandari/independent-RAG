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

## Hybrid search, re-ranking and caching

- **Hybrid search** (`app/hybrid.py`): our corpus is a legal text full of exact terms (article
  numbers, "Annex III", defined terms) that pure vector search can miss — the `exact_term`
  stress-test category in `data/questions.json` targets exactly this. FAISS (semantic) and BM25
  (keyword) each retrieve their own top-15 candidates, merged by Reciprocal Rank Fusion (rank
  position, not raw score — FAISS cosine and BM25 scores live on different scales). Runs entirely
  locally, no API calls. Toggle with `USE_BM25` in `app/rag.py`.
- **Re-ranking** (`app/rerank.py`): FAISS pulls 15 candidates, then the `BAAI/bge-reranker-base`
  cross-encoder scores each one from 0 to 1 for how well it answers the question. Passages
  scoring at least 0.1 are kept, best 8 go to the answer prompt. It runs on your GPU (Apple
  Silicon via MPS, NVIDIA via CUDA, CPU otherwise) with no API calls. The model (~1.1 GB)
  downloads from Hugging Face on first start; after that it loads in ~10 s and scores a question
  in under a second. Without `sentence-transformers` installed, re-ranking is skipped and FAISS
  order is used.
- **Caches** (`data/cache/`): repeated questions are answered from the query cache with no API
  calls; the retrieval and re-rank caches skip the embedding and relevance calls when only the
  answer needs regenerating. All caches reset automatically when the index, models, prompts or
  thresholds change. Delete `data/cache/` to clear them by hand.

## Reflection and evaluation

- **Confidence per answer** (`app/reflection.py`): every answer gets a confidence score shown under
  it in the UI, built from signals the pipeline already has, so it costs nothing extra: how
  relevant the best passage is, how many passages passed the filters, whether the answer cited
  the passages it was given, and how well the cited passages match the answer text (scored by the
  local re-ranker). Truncated answers are capped below "high". For a refusal, the score is
  confidence in the refusal: high when nothing relevant was found, low when a strong passage
  existed.
- **Evaluation** (`scripts/evaluate.py`): runs every question in `data/questions.json` and writes
  `data/eval/report.md` and `data/eval/results.json`.

  ```bash
  python -m scripts.evaluate               # all questions
  python -m scripts.evaluate --limit 5     # quick run
  python -m scripts.evaluate --fresh       # ignore cached answers (costs API calls)
  ```

  The report covers retrieval quality (Hit@k, Recall@k, MRR, FAISS order vs re-ranked),
  context recall, citation accuracy, refusal accuracy, token F1 against `expected_answer`,
  results by category and sub-type, the priority questions, and which stage each failure came
  from (retrieval, ranking or generation). Its calibration table shows whether each confidence
  band is right as often as it claims; use it to tune the weights in `app/reflection.py`.

## RAG day 2 status (confidence threshold, ablation, eval runner)

The team was asked to reduce API spend, so the numbers below are measured **only on
questions already cached from real prior use** (`data/cache/`) — zero new API calls were
made to produce them. Run `python -m scripts.cache_only_eval` to reproduce; it skips any
question that isn't already cached rather than making a live call.

- **Retrieval quality (7 of 48 questions happened to be cached):** recall@1 = recall@5 =
  71%, MRR = 0.738. Real numbers, small sample — not the full 48-question picture yet.
- **Confidence threshold:** not yet computable. None of the currently-cached questions that
  *should* be refused (out-of-scope/ambiguous) made it far enough into the pipeline to get a
  re-rank score — the one cached example ("what's the weather today") was filtered out
  before re-ranking, which is correct behaviour but leaves nothing to set a threshold
  against. Needs a handful of targeted live calls on refuse-group questions to complete.
- **Ablation table (vector-only / +reranking / +BM25):** not yet run for real. The cache
  only reflects the one pipeline configuration actually used in production (hybrid +
  re-ranking) — isolating "vector only" and "+reranking only" rows needs fresh retrieval per
  config, which can't be done from cache alone. `scripts/eval_runner.py --ablation` is built
  and mechanically verified (see its local mock test), just not run live yet.
- **Prompt injection test, regression check:** built and run (`scripts/injection_test.py`,
  `scripts/regression.py`) — results pending from whoever ran them last.

**Next step, once the team's spend situation is resolved:** a small, targeted live run
(the missing refuse-group questions for the threshold, plus the two ablation configs) would
complete this — estimated at well under 200 total API calls, all cheap embedding-only
requests.

## Using the app

- **Follow-up questions**: the chat remembers the conversation. A follow-up such as "what about
  the second requirement?" is rewritten into a standalone question from the last 3 turns (one
  small gpt-4o-mini call, cached), and the answer shows how it was interpreted. The 🗑 icon or
  "Clear conversation" starts over.
- **Sources**: the panel next to the chat shows the full text of every passage the model was
  given, with the cited ones marked ✅, plus their relevance and similarity scores. Click any
  earlier answer to see its sources. Candidates that didn't make it into the context are under
  "Other retrieved candidates".
- **Feedback**: 👍/👎 under each answer is logged to `data/feedback.jsonl` with the question, the
  answer, its confidence, and the cited and context pages.
- **Progress**: while an answer is being prepared the chat shows the current stage (understanding
  the follow-up, embedding and searching, re-ranking, generating, checking confidence).
- **Export**: download the whole conversation or one answer (click it first; the latest answer
  otherwise) as .txt or .pdf.
