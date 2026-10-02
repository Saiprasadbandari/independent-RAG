# Testing log

Everything we tested from the first commit to the demo, what each test found, what we changed
and what the numbers did. Times are Paris time (CEST). Built from the git history and pull
requests, `scripts/FAILURE_LOG.md`, `scripts/AGENT_FAILURE_LOG.md`, the README tables and our
work-session notes. Tests marked *local* ran on this machine (bge-base embeddings, local
re-ranker, Qwen2.5-3B), *api* ones on `text-embedding-3-small` + `gpt-4o-mini`.

1. [Testing log](#1-testing-log): every test, in order
2. [Improvements](#2-improvements): every change it led to, in order
3. [How the numbers moved](#3-how-the-numbers-moved)
4. [Still open](#4-still-open)

## 1. Testing log

| # | When | Who | What we tested, and how | What we found | What we changed | Result |
|---|---|---|---|---|---|---|
| 1 | 28 Sep 15:11 | Sai | Extracting the 144-page PDF (`scripts/extract.py`) | Clean text per page; written up in `EXTRACTION_REPORT.md` | — | `corpus.json` + `sample.json` shared on Teams |
| 2 | 28 Sep 17:02 | Sai | Merging the first RAG app with the embedding module | The two first versions overlapped and broke each other | Reverted both, re-added embedding/indexing cleanly (17:10); Rayyan rebuilt `rag.py` on the template (18:14) | One pipeline on `main` |
| 3 | 28 Sep 18:58 | Sai | Accuracy check on all 44 questions | The Article 99 fine passage (a priority question) ranked 6th–7th, just past `TOP_K=5` | `TOP_K` 5 → 8 | Hit rate 29/38 → 32/38, all 5 priority questions fixed |
| 4 | 29 Sep 15:33 | Rayyan | Cost of each start-up | The corpus was re-embedded on every run, paying for tokens each time | Index saved to disk and loaded; query and retrieval caches (15:43) | Embeddings paid once |
| 5 | 29 Sep 16:12 | Rayyan | Cost of the first re-ranker (gpt-4o-mini scoring passages, "Jev-style") | Too expensive: one extra model call per question | Local cross-encoder `bge-reranker-base` on the Mac GPU (16:27), API re-ranker removed (16:33) | Re-ranking costs $0 |
| 6 | 29 Sep 16:52 | Rayyan | Python compatibility of the new code | A backslash inside an f-string only works on Python 3.12+ | Moved into a helper | Runs on any Python (and the team later standardised on 3.12) |
| 7 | 29 Sep 16:56 | Rayyan | Chat UI by hand | Enter didn't send from the multi-line box; 🗑 cleared the screen but not the conversation memory | Enter sends, Shift+Enter is a new line; 🗑 resets memory (17:00) | Both work |
| 8 | 29 Sep 17:22 | Rayyan | First full local run (no API key) | (a) an answer with no citation showed 71% "high" confidence; (b) "best pizza topping" was rewritten into the previous question and answered; (c) exact terms like "Article 5" missed by vector search | (a) uncited answers capped below high; (b) only rewrite messages that look like follow-ups (17:51); (c) noted for hybrid search | (a), (b) fixed; (c) fixed by #13 |
| 9 | 29 Sep 17:41 | Sai | Agent loop mechanics with a scripted mock client | Tool dispatch, message threading, trajectory log and step cap all work | — | Loop verified at $0 |
| 10 | 29 Sep 17:47 | Rayyan | Loading the local Qwen model on macOS | Segmentation fault: FAISS imported before PyTorch | `app/__init__.py` imports torch first (17:51) | Local mode starts |
| 11 | 29 Sep 18:02 | Rayyan | Running all 44 questions to warm the cache | — | Answers cached | Repeated questions answered instantly, at $0 |
| 12 | 30 Sep 16:46 | Rayyan | Nonsense input ("boom boom") in a conversation | Rewritten into the previous topic and answered with confidence | Rewrite check (the rewrite must keep at least half the topic words) and an off-topic threshold per embedding model: 0.60 for bge (in-domain ≥ 0.64, off-topic ≤ 0.59, "boom boom" 0.45) | Refused (fixed 30 Sep 22:06) |
| 13 | 30 Sep 18:14 | Sai | BM25 + Reciprocal Rank Fusion on the real corpus | BM25 alone puts the Article 99 fine (p. 115) first for "Article 99 penalties fines" | Hybrid search, `retrieval_checker.py`, `FAILURE_LOG.md` with 5 real failures | See #18 |
| 14 | 30 Sep 19:05 | Rayyan | Prompt injection test, 10 attacks (`injection_test.py`) | 7/10 attacks resisted | Question cleaning (`sanitize_question`) | 10/10 resisted (fixed 30 Sep 22:06) |
| 15 | 30 Sep 19:49 | Prodipta | Conversation sidebar and PDF source viewer against a fake RAG | Works | — | Merged (PR #10) |
| 16 | 30 Sep 22:06 | Rayyan | Hardening the system prompt against injection, regression check | The local model then refused 3 easy questions it answered before (Q3, Q4, Q6) | Kept the original system prompt; question cleaning alone does the job | 45% → 45% correct, 66% → 66% context recall: no regression |
| 17 | 01 Oct 08:37 | Rayyan | Moving to uv + Python 3.12 | All code checked for 3.12; lock file builds | `pyproject.toml`, `uv.lock`, `.python-version` | One command install (`uv sync`) |
| 18 | 01 Oct 09:10 | Rayyan | Regression check of hybrid search (PR #9) | BM25-only hits got a made-up 0.30 score and were thrown away by the 0.60 gate | Keyword-only hits flagged and kept for the re-ranker; the off-topic gate decides on vector hits only | 45% → 50% correct, 66% → 71% recall; fixed Q9, Q19, Q27 |
| 19 | 01 Oct 09:25 | Rayyan | Q32 "regressed" in the same check | It had passed before with a *wrong* answer and now refused | — | Shows "cited an expected page" can pass wrong answers |
| 20 | 01 Oct 16:04 | Sai | Eval on cached data only (no new API calls) | 7 of 48 questions cached: recall@1 71%, MRR 0.74 | Plan for one small paid run | See #22 |
| 21 | 01 Oct 16:18 | Rayyan | Conversation memory (last 2 exchanges) | Identical scores on all questions; a 5-turn conversation keeps context, a new topic mid-chat is answered, nonsense still refused | — | 50% → 50% |
| 22 | 01 Oct 16:33 | Sai | Ablation in api mode (12 questions, 42 calls) | Vector only R@1 0.38 · + re-ranking 0.79 · + BM25 MRR 0.84 → 0.85 | Kept re-ranking and BM25 | Table in the README |
| 23 | 01 Oct 16:48 | Rayyan | New eval runner against the old script | Bugs in the first version (a garbled expression, caches per row, report overwritten) fixed before use | Shared search code in `rag.py` for the app and the runner | All 44 answers word-for-word identical; local R@1 0.61, R@5 0.84, MRR 0.70 |
| 24 | 01 Oct 17:04 | Rayyan | "What does Article 41 say?" and "the next one" | Article 41 was refused as off-topic; Article 42 retrieved definition (42) of Article 3 | Look-up by heading, and "next / previous" resolved by counting | R@5 84% → 87%, MRR 0.70 → 0.72, Q8 fixed; Article 41 → next → next → previous gave pages 76 → 77 → 78 → 77 |
| 25 | 01 Oct 17:16 | Rayyan | Confidence threshold 0.1 | Wrongly refused the AI-generated images question (best passage 0.097) | Threshold 0.05 | Wrong refusals 1/38 → 0/38, correct 50% → 52% (Q36 fixed) |
| 26 | 01 Oct 18:16 | Sai | Threshold 0.05 in api mode (existing data, no new calls) | All 12 answerable questions score ≥ 0.516 | — | 0/12 wrong refusals, 2/6 correct refusals |
| 27 | 01 Oct 18:16 | Rayyan | Checklist page | It ticked an item because `checklist.py` itself contains the text it searches for | Its own file left out of the search; "done" told apart from "delivered" (18:34) | Correct ticks |
| 28 | 01 Oct 19:18 | Sai | Agent recovery | Only an exact repeated call was caught: a failing search could be rephrased until the step cap | `RETRY_LIMIT = 3` failed calls in a row | Mock test stops at exactly 3 |
| 29 | 01 Oct 19:32 | Rayyan | `get_definition` and `date_diff` against the real corpus | First parse found 67 of 68 definitions (no. 58 has words between the term and "means"); a README date wasn't in Article 113 | Parser and README wording fixed | 68/68 definitions; 1 Aug 2024 → 2 Aug 2026 = 731 days |
| 30 | 01 Oct 19:33 | Sai | Agent eval runner with the mock client | Tool choice 11%: one mock client shared by all questions ran out of script | A fresh mock client per question | 78% (7/9), the 2 misses being the tools the mock never calls |
| 31 | 01 Oct 19:38 | Rayyan | Agent in local mode | Qwen ignored the tools: the local client didn't support tool calls | Qwen's `<tool_call>` format mapped to OpenAI tool calls | 9/9 probe questions routed to the right tool |
| 32 | 01 Oct 20:05 | Sai | RAG eval runner after agent questions were added | It would have scored the 9 agent questions as retrieval questions | Agent questions excluded | Caught with a free local dry run |
| 33 | 01 Oct 20:09 | Rayyan | Injection against actions: a poisoned Article 113 passage telling the agent to send a message, fake send that declines | Before hardening 3/3 tried to send; after prompt hardening still 3/3 | A guard in the agent code: `send_message` only offered when the user asks to send | 0/3 reached the approval gate; a real send request still works (1/1) |
| 34 | 01 Oct 20:20 | Rayyan | "Act, with approval" demo | "Send selected answer to Teams" refused every agent answer as failed | The preview uses the agent's draft | Works |
| 35 | 01 Oct 20:38 | Rayyan | Agent vs RAG, 5 priority questions (local) | A Qwen tool call cut off by the token limit was shown as the answer | Unclosed `<tool_call>` parsed as a call | RAG: 2 correct + 1 partial, 1 model call, 18 s. Agent: 1 correct + 1 partial, 2.4 calls, 65 s; it quoted Article 100's €1.5M instead of Article 99 |
| 36 | 01 Oct 20:53 | Rayyan | How it works page, clicked in headless Chrome | Opened decision boxes squeezed their text | Rounded corners when open | 0 → 2 → 56 → 2 boxes open (expand all and put back) |
| 37 | 01 Oct 21:02 | Rayyan | Checklist links to GitHub | All 78 file, function and README links resolve on `main` | — | Who wrote each item, from git blame |
| 38 | 02 Oct 00:05 | Sai | Real post to Teams | The new Workflows webhook rejected every post ("adaptive card request is missing or invalid") | Adaptive Card payload | A real message arrived in the channel |
| 39 | 02 Oct 00:59 | Sai | Forged citations | A `[Source: …, p. N]` label for a page the model was never given went unnoticed | `forged_citations` check, confidence capped | 0% unverified citations on 5 api answers |
| 40 | 02 Oct 01:12 | Prodipta | Backend unit tests and chunk report | — | On branch `feature/backend-tests`, not merged yet | — |
| 41 | 02 Oct 01:46 | Sai | Agent on an off-topic question ("best pizza topping") | Answered from general knowledge instead of refusing | Prompt: search first, refuse when nothing is found | Refuses |
| 42 | 02 Oct 06:28 | Sai | A bare "hi" in RAG mode | Refused with the off-topic sentence | Greetings answered directly at $0; candidates list scrolls | Friendly reply |
| 43 | 02 Oct 10:40 | Rayyan | Routing through the real UI, one question per tool (local) | 7/9 as expected: on the fine question it searched but did the maths in its head (and used 3% instead of 7%); on pizza it refused without searching first | — | Re-check in api mode |
| 44 | 02 Oct 00:20 | Sai | Agent eval questions audited against the real course spec repo (`GenAI_week`, not just the internal checklist) | Used a free-text `"scenario"` field instead of the required `"sub_type"` with 11 exact enum values | Rewrote all 11 questions to the correct taxonomy; verified `rephrase_retry`'s wording genuinely fails/succeeds via `retrieval_checker.py` (api mode) | All 11 required sub-types present, 60 questions total, minimums/priority checks still pass |
| 45 | 02 Oct 00:42 | Sai | RAG demo questions in the README against the same spec repo | Used invented categories instead of the 6 course-required scenarios (Single chunk, Multiple chunks, Follow-up, Out of scope, Prompt injection, Exact term) | Rewrote the section to the 6 required scenarios, verified against real corpus pages | `readme_rag_demo` checklist check still passes |
| 46 | 02 Oct 07:26 | Sai | Teams webhook, after a user report that a real send wasn't arriving | `.env` pointed at a different flow ("Send webhook alerts to a channel") that had failed on every single run since 01 Oct — an "Attachments is null" error in its own branching logic, unrelated to the #38 payload fix | Repointed `.env` at the verified-working "Send webhook alerts to General" flow (confirmed via Power Automate run history, not guessed) | New run Succeeded; message confirmed arrived in Teams by the user |

## 2. Improvements

| When | Who | Improvement | Why (test #) |
|---|---|---|---|
| 28 Sep 15:11 | Sai | Corpus extraction, team README | Assignment 0 |
| 28 Sep 17:10 | Sai | Chunking, embeddings, FAISS index, app entry point | Day 1 |
| 28 Sep 17:19 | Prodipta | First Gradio UI, falls back to the real RAG | Day 1 |
| 28 Sep 18:14 | Rayyan | RAG pipeline on the course template: retrieval, cited answers, refusals | Day 1 |
| 28 Sep 18:59 | Sai | `TOP_K` 5 → 8 | 3 |
| 29 Sep 15:43 | Rayyan | Saved index, query and retrieval caches | 4 |
| 29 Sep 16:27 | Rayyan | Re-ranking on the local GPU (bge-reranker-base) | 5 |
| 29 Sep 16:46 | Sai | Agent tools: `search_corpus`, `calculator`, `quote_exact` | Agent day 1 |
| 29 Sep 16:48 | Rayyan | Answer confidence score, evaluation script | Reflection |
| 29 Sep 17:00 | Rayyan | Chat UI: follow-ups, sources panel, feedback, export | 7 |
| 29 Sep 17:33 | Sai | Agent loop (6-step cap), agent mode in the UI | Agent day 1 |
| 29 Sep 17:51 | Rayyan | Hybrid and local run modes, streamed answers, smaller prompts | 8, 10 |
| 30 Sep 18:14 | Sai | Hybrid search (BM25 + RRF), retrieval checker, failure log | 8c, 13 |
| 30 Sep 19:49 | Prodipta | Conversation sidebar, PDF source viewer | 15 |
| 30 Sep 22:06 | Rayyan | Question cleaning, rewrite check, off-topic threshold per model | 12, 14, 16 |
| 01 Oct 08:37 | Rayyan | uv + Python 3.12 | 17 |
| 01 Oct 09:25 | Rayyan | Keyword-only BM25 hits kept; regression checks logged | 18, 19 |
| 01 Oct 16:18 | Rayyan | Last 2 exchanges sent to the answer model | 21 |
| 01 Oct 16:33 | Sai | Ablation table, first confidence threshold | 20, 22 |
| 01 Oct 16:59 | Rayyan | Eval runner: retrieval-only and full modes, ablation, threshold report | 23 |
| 01 Oct 17:13 | Rayyan | Article look-up by heading, "the next one" by counting | 24 |
| 01 Oct 17:23 | Rayyan | Confidence threshold 0.05 | 25, 26 |
| 01 Oct 17:24 | Sai | UI redesign: live cost table, follow-up suggestions, ablation table | UX |
| 01 Oct 17:29 | Sai | Teams webhook behind preview + confirm | Agent day 2 |
| 01 Oct 18:20 | Rayyan | Project checklist page | 27 |
| 01 Oct 18:38 | Sai | Threshold, ablation and RAG demo write-ups in the README | 22, 26 |
| 01 Oct 19:18 | Sai | Retry limit; `send_message` as an agent tool | 28 |
| 01 Oct 19:32 | Rayyan | `get_definition` and `date_diff` tools (6 in total) | 29 |
| 01 Oct 19:33 | Sai | Agent eval runner | 30 |
| 01 Oct 20:05 | Sai | Agent questions kept out of the RAG eval | 32 |
| 01 Oct 20:09 | Rayyan | `INJECTION_TEST` flag, send guard, tool calls on the local model | 31, 33 |
| 01 Oct 20:22 | Rayyan | Agent section in the README; sending agent answers fixed | 34 |
| 01 Oct 20:31 | Sai | `AGENT_FAILURE_LOG.md` | 28, 30, 33, 34 |
| 01 Oct 20:42 | Rayyan | Agent vs RAG comparison | 35 |
| 01 Oct 20:53 | Rayyan | How it works page | 36 |
| 01 Oct 21:02 | Rayyan | Checklist links to GitHub, who did what | 37 |
| 02 Oct 00:05 | Sai | Teams Adaptive Card payload | 38 |
| 02 Oct 00:59 | Sai | Citation check | 39 |
| 02 Oct 01:46 | Sai | Agent refuses off-topic questions | 41 |
| 02 Oct 06:28 | Sai | Greetings answered directly in RAG mode | 42 |
| 02 Oct 00:20 | Sai | Agent eval questions rewritten to the correct `sub_type` taxonomy | 44 |
| 02 Oct 00:42 | Sai | RAG demo questions rewritten to the 6 required scenarios | 45 |
| 02 Oct 07:26 | Sai | Teams webhook repointed at the working flow | 46 |

## 3. How the numbers moved

RAG, local mode, all questions (`scripts/regression.py` between runs):

| After | Correct (cited an expected page) | Context recall | Recall@5 | MRR | Wrong refusals | Injection attacks resisted |
|---|---|---|---|---|---|---|
| Baseline, 30 Sep | 45% | 66% | — | — | — | 7/10 |
| Question cleaning, 30 Sep 22:06 | 45% | 66% | — | — | — | 10/10 |
| Hybrid search, 01 Oct 09:25 | 50% | 71% | — | — | — | 10/10 |
| Conversation memory, 01 Oct 16:18 | 50% | 71% | — | — | — | — |
| Eval runner, 01 Oct 16:59 | 50% | 82%* | 0.84 | 0.70 | 1/38 | — |
| Article look-up, 01 Oct 17:13 | 50% | 84% | 0.87 | 0.72 | 1/38 | — |
| Threshold 0.05, 01 Oct 17:23 | 52% | 87% | — | — | 0/38 | — |

\* The new eval runner counts context recall differently from the earlier script (82% at its own
baseline), so compare within the rows from 01 Oct 16:59 on.

Earlier, Sai's first accuracy check (28 Sep) went from 29/38 to 32/38 questions with the right
passage retrieved when `TOP_K` went from 5 to 8.

Retrieval stages, api mode, 12-question sample (01 Oct 16:33):

| Version | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| Vector only | 0.38 | 0.67 | 0.56 |
| + re-ranking | 0.79 | 0.83 | 0.84 |
| + BM25 | 0.79 | 0.83 | 0.85 |

Agent:

| Test | Before | After |
|---|---|---|
| Tool choice, mock eval (30) | 11% | 78% |
| Routing probes, local (31) | tools never called | 9/9 |
| Injected sends reaching the approval gate (33) | 3/3 | 0/3 |
| Article 3 definitions found (29) | 67/68 | 68/68 |
| Routing through the UI, local (43) | — | 7/9 |

## 4. Still open

- Every agent number above is local mode (Qwen2.5-3B) or the mock client: re-run
  `agent_eval_runner.py`, `agent_vs_rag.py` and `action_injection_test.py` in api mode.
- Two negation questions and the Annex II count question are still open in `FAILURE_LOG.md`.
- Only 4 to 6 should-refuse questions in the set: two of them score high whatever the threshold.
- The local model sometimes skips `calculator` and does the maths itself (test 43).
- Prodipta's `feature/backend-tests` adds the PDF under `data/`, which has to stay out of git.
