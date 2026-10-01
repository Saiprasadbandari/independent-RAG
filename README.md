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

Python 3.12, managed with [uv](https://docs.astral.sh/uv/). Install uv once
(`curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS/Linux), then from the repo root:

```bash
uv sync                                   # creates .venv with Python 3.12 and the locked dependencies
echo "OPENAI_API_KEY=sk-..." > .env
```

`uv sync` installs exactly the versions in `uv.lock`, so every teammate gets the same environment.
To add a dependency use `uv add <package>` (it updates `pyproject.toml` and `uv.lock`; commit both).

Get the `data/` folder from Teams (it's gitignored). You need at least `data/corpus.json`;
to regenerate it from `data/sources/`, run `uv run python scripts/extract.py`.

Build the index and start the app (run from the repo root):

```bash
uv run python -m app.main --build-only   # embeds corpus.json -> data/my_index.faiss + data/chunks.json
uv run python -m app.main                # starts Gradio on http://localhost:7860
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
  locally, no API calls. Toggle with `USE_BM25` in `app/rag.py`. Keyword-only hits are passed to
  the re-ranker like any other candidate (and labelled "keyword match" in the sources panel), but
  they don't count towards the off-topic check: BM25 matches common words in almost any question,
  so a question is only treated as on-topic when a vector match clears the similarity threshold.
- **Re-ranking** (`app/rerank.py`): the merged FAISS + BM25 candidates (up to 30) are scored by the
  `BAAI/bge-reranker-base` cross-encoder from 0 to 1 for how well they answer the question.
  Passages scoring at least 0.1 are kept, and the context is built from the best of them (see
  "Keeping the answer prompt small" below). It runs on your GPU (Apple
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
- **Eval runner** (`scripts/eval_runner.py`): runs every question in `data/questions.json` and
  writes `data/eval_results.json` (summary plus one row per question) and `data/eval/report.md`.

  ```bash
  uv run python -m scripts.eval_runner --retrieval-only            # search + re-rank, no answers (nearly free)
  uv run python -m scripts.eval_runner                             # full: answers generated too
  uv run python -m scripts.eval_runner --retrieval-only --ablation # ablation table -> data/eval/ablation.md
  uv run python -m scripts.eval_runner --limit 5 --fresh           # quick run, ignoring caches
  ```

  A retrieved chunk is a hit when its `(source, page)` is in the question's `expected_chunks`.
  The report gives recall@1/3/5 and MRR (on questions with `expected_chunks`), refusals on
  out-of-scope and on-topic-unanswerable questions, average latency, breakdowns by category,
  difficulty and sub-type, a confidence-threshold table (correct vs wrong refusals at each top
  re-ranker score) and the failed questions with their stage (retrieval, ranking, generation). Each
  row logs the retrieved chunks with scores, the rank of the first expected chunk, the rewritten
  query for follow-ups, latency, and in full mode the answer, citations, cost, and a
  `correct_manual` field to fill in by hand. Full mode also has a calibration table (is each
  confidence band right as often as it claims). `--no-rerank`, `--no-bm25` and `--no-rewrite`
  switch one improvement off; `--ablation` measures them cumulatively.

### Confidence threshold

`MIN_RELEVANCE` in `app/rerank.py` is the top re-ranker score a question's best passage needs to
reach the answer prompt; below it, the question is refused without calling the model. Moved from
0.1 to 0.05 after 0.1 wrongly refused a real answerable question (local mode, best passage 0.097):

| Threshold | Correct refusals | Wrong refusals |
|---|---|---|
| local (0.1, old) | 2/4 should-refuse | 1/38 answerable |
| local (0.05) | 2/4 should-refuse | 0/38 answerable |
| api (0.05) | 2/6 should-refuse | 0/12 answerable |

A few should-refuse questions still score high regardless of threshold (0.999/0.985 in api mode,
0.999/0.22 in local mode) — that's the model citing a plausible-but-wrong passage, not a threshold
problem, so no cutoff fixes it without also refusing answerable questions. It's caught separately
by the uncited-answer confidence cap, not by this constant. Full details in `scripts/FAILURE_LOG.md`.

### Ablation

Each retrieval stage added cumulatively, measured with `scripts/eval_runner.py --ablation`:

| Version | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| Vector only | 0.38 | 0.67 | 0.56 |
| + re-ranking | 0.79 | 0.83 | 0.84 |
| + BM25 (hybrid) | 0.79 | 0.83 | 0.85 |

Re-ranking is the big jump (recall@1 more than doubles); BM25 on top gives a smaller but real MRR
gain by fixing keyword-heavy questions vector search misses on its own. Kept both: the hybrid
(vector + BM25 + re-ranking) pipeline is what `app/rag.py` runs by default (`USE_BM25` /
`USE_RERANK` in `app/rag.py` toggle either off). Numbers above are api mode, `data/day2_final.json`
(12-question sample); local mode numbers on the full 38-question set are in `scripts/FAILURE_LOG.md`.

## Run modes

`uv run python -m app.main --mode <mode>` (also `scripts.eval_runner --mode`) picks where each step runs:

| mode | embeddings | answers | needs API key | cost per new question |
|---|---|---|---|---|
| `api` (default) | text-embedding-3-small | gpt-4o-mini | yes | ~$0.0002 |
| `hybrid` | `BAAI/bge-base-en-v1.5` on your GPU | gpt-4o-mini | yes | ~$0.0002 |
| `local` | `BAAI/bge-base-en-v1.5` on your GPU | `Qwen/Qwen2.5-3B-Instruct` on your GPU | no | $0 |

Re-ranking and passage trimming run on your GPU in every mode, and answers stream in every
mode. `api` and `hybrid` cost about the same per question (embedding a question is ~$0.0000004;
the answer is the real cost); `hybrid` makes building the index free and keeps retrieval
working without the API. The per-question saving comes from the prompt trimming below. `local` works offline but is slower (~10 s before the answer starts on an M4)
and the small model follows the citation instructions less reliably. The local models download
from Hugging Face on first use (~0.4 GB embeddings, ~6 GB chat model). Local embeddings get their
own index (`data/my_index-local.faiss`), so switching modes never mixes vectors of different sizes.

### Keeping the answer prompt small

Input tokens are most of the cost, so the context is assembled on the GPU before anything is sent:

- at most 5 re-ranked passages, each scoring at least 40% of the best one
- the chunks following the 2 best passages are added while they stay relevant, so lists and
  articles that cross a chunk boundary arrive complete
- other long passages are trimmed to their relevant sentences (✂️ in the sources panel)

On 8 test questions this sent 29% fewer tokens than sending the 8 best passages whole, with the
answer still in context every time; list questions get more complete (Article 5's list of
prohibited practices went from 1 item in context to most of them).

Follow-up questions are only rewritten when they look like follow-ups ("it", "that",
"what about…", very short questions), which skips that call for most questions and stops a new
question from being bent towards the previous topic.

## Safety and testing

Questions are cleaned before anything else runs (`sanitize_question` in `app/rag.py`):
sentences that give the assistant orders ("ignore your rules", "end your answer with…",
"write a poem", "reveal your prompt", "cite page…") are dropped, and text passed off as context
(`Context:` blocks, `[1]`, `[Source: …]` labels) is stripped, so the question's own claims can't
pose as retrieved passages. A message that is nothing but instructions is refused without a
search. The chat shows the cleaned question as "Interpreted as" whenever it differs. The system prompt is
deliberately unchanged: adding anti-injection rules to it made the local 3B model refuse easy
questions it answered before, while cleaning the question alone stops all 10 test attacks.

Follow-up rewrites are checked: the rewritten question must keep at least half of the topic words
the user typed, otherwise the user's own wording is used. This stops a rewrite from swapping an
unrelated message ("boom boom") for the previous question.

The off-topic gate uses a similarity threshold per embedding model (`MIN_SCORES`): 0.30 for
text-embedding-3-small, 0.60 for the local bge-base, whose scores run higher.

```bash
uv run python -m scripts.injection_test --mode local   # 10 attacks, writes data/eval/injection.md
uv run python -m scripts.eval_runner --mode local --fresh # full question set
uv run python -m scripts.regression data/eval/baseline.json data/eval_results.json   # exits 1 on regressions
```

The injection test covers instruction override, persona jailbreaks, system-prompt extraction,
fake context in the question, forged citations, a follow-up hijack, and a poisoned passage planted
in an in-memory copy of the index. Content inside a poisoned passage can still end up in an
answer; the only defence against that is controlling what gets indexed.

## Using the app

- **Conversation memory**: the answer model receives the last 2 exchanges (earlier answers
  trimmed to 600 characters, citation markers removed) along with each new question, so it can
  follow up on what was said ("explain that more simply"). Retrieval can't read history, so a
  question that looks like a follow-up ("what about the second one?", "and the fines?") is also
  rewritten into a standalone question for the search (one small gpt-4o-mini call, cached; skipped
  on the first turn). The answer gets the user's own question, the history and the rewrite as a
  hint; the chat shows the rewrite as "Interpreted as". A rewrite that drops the user's topic words
  is discarded. Answers given inside a conversation are cached per conversation, so the same words
  in another conversation get their own answer. The 🗑 icon or "Clear conversation" starts over.
  In `questions.json`, follow-up questions list the earlier questions in a `history` field and the
  evaluation replays them first.
- **Sources**: the panel next to the chat shows the full text of every passage the model was
  given, with the cited ones marked ✅, plus their relevance and similarity scores. Click any
  earlier answer to see its sources. Candidates that didn't make it into the context are under
  "Other retrieved candidates".
- **Feedback**: 👍/👎 under each answer is logged to `data/feedback.jsonl` with the question, the
  answer, its confidence, and the cited and context pages.
- **Progress and streaming**: the reply shows the current stage (understanding the follow-up,
  searching, re-ranking and trimming, writing), then the answer streams in as it's written.
- **Export**: download the whole conversation or one answer (click it first; the latest answer
  otherwise) as .txt or .pdf.

## RAG demo questions

Six questions covering the main capabilities, with the expected answer and the page(s) the
answer should cite. All are in `data/questions.json` (the first five are the `priority` set).

1. **Factual** — "What is the maximum administrative fine for non-compliance with the prohibited
   AI practices in Article 5?"
   Expected: up to EUR 35,000,000 or 7% of worldwide annual turnover, whichever is higher.
   Page: 115.

2. **Cross-reference (adjacent chunks)** — "What must a deployer's fundamental rights impact
   assessment for a high-risk AI system cover, and when relative to first use must it be carried
   out?"
   Expected: the deployer's processes for using the system, period/frequency of use, affected
   persons, specific risks of harm, human oversight measures and a complaint/redress plan — carried
   out before the system is first used.
   Pages: 69, 70.

3. **Cross-reference (same document)** — "If an AI system is classified as high-risk under
   Article 6, what is the maximum fine its provider risks for failing to meet the resulting
   obligations?"
   Expected: up to EUR 15,000,000 or 3% of worldwide annual turnover, whichever is higher (Article
   6 classification triggers the Section 2/3 obligations, whose breach falls under Article 99(4)).
   Pages: 53, 115.

4. **Stress-test (yes/no)** — "Does the AI Act apply to AI systems developed and used exclusively
   for military or national security purposes?"
   Expected: no — the Regulation explicitly excludes AI systems used exclusively for military,
   defence or national security purposes.
   Page: 45.

5. **Stress-test (table/list)** — "What are the 8 areas of high-risk AI system use cases listed in
   Annex III?"
   Expected: biometrics; critical infrastructure; education and vocational training; employment,
   workers' management and access to self-employment; access to essential private and public
   services; law enforcement; migration, asylum and border control management; administration of
   justice and democratic processes.
   Pages: 127, 128, 129.

6. **Out-of-scope (refusal)** — "What are the specific data breach notification deadlines under
   the GDPR?"
   Expected: refused. GDPR is a separate regulation from the EU AI Act and isn't in this corpus —
   a good test of refusal since the question sounds on-topic (EU tech regulation) but isn't
   answerable from the indexed document.
   Page: none (correct behaviour is no citation, no answer).
