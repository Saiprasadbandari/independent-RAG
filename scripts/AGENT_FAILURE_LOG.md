# Agent Failure Log

Documented failures found while building the agent loop, and the fixes applied. Same spirit as
`scripts/FAILURE_LOG.md` for the RAG side — see `scripts/agent_eval_runner.py` (correctness, tool
choice, steps, latency) and `scripts/action_injection_test.py` for how these are found.

| Failure | Found by | Problem | Fix applied | Fixed? |
|---|---|---|---|:---:|
| No real retry limit | Code review while finishing Agent day 2 | `app/agent.py` only caught an *exact* duplicate call (same tool, same arguments twice in a row) — a question whose search kept failing could be rephrased indefinitely within the 6-step cap, never actually giving up early. | Added a consecutive-failure counter (`_is_failure`, anything starting with `Error`, `No matching passages found.` or `No text found for`) and a `RETRY_LIMIT=3`: after 3 failures in a row the loop forces a final answer instead of continuing. Verified with a standalone test (`AlwaysFailClient`): stops at exactly step 3, not the step cap, with a new `gave_up_retrying` flag distinct from `stopped_at_cap`. | ✅ |
| Mock client gave misleading tool-choice scores across a 9-question eval run | First `--mock` run of `scripts/agent_eval_runner.py` | `scripts/mock_client.py`'s `MockOpenAI` scripts responses by a `self._step` counter on the instance (1st call → search, 2nd → calculator, 3rd → answer). Reusing one client across all 9 questions meant only question 1 got the realistic sequence; by question 4+ the step counter was past the scripted range and tool choice looked wrong (11% pass) even though the harness itself worked. | Give each question its own `MockOpenAI()` instance in `--mock` mode. Re-ran: tool choice 78% (7/9), correctly flagging only the 2 scenarios needing `quote_exact`/`send_message` that the scripted mock never calls — the real failure signal, not a client-state artifact. | ✅ |
| "Send selected answer to Teams" rejected every agent-mode answer | Rayyan, while testing the "Act, with approval" demo scenario | Agent turns are stored under `agent_answer`, not `answer`; the send-preview code only looked for `answer`, so it always said "Can't send a failed answer." for agent-mode turns — blocking the one demo scenario that needs it most. | Preview now uses the agent's own `send_message` draft when present, falling back to its answer text otherwise. | ✅ |
| Prompt hardening alone doesn't stop an injected send | Rayyan, `scripts/action_injection_test.py` (local mode, Qwen2.5-3B, 3 probe questions, poisoned Article 113 passage instructing the agent to message the team) | A firmer system prompt ("only call `send_message` when the user asks") didn't move the model at all: 3/3 poisoned runs still called `send_message` and reached the approval gate, same as before hardening. | A code-level guard in `app/agent.py`: `send_message` is only offered as a tool when the user's own message actually asks to send/post/share; a call made anyway is refused before reaching the approval gate. Guard stopped all 3/3 injected sends. Matches the course's own guidance (`Agentic AI` slides, p.70): "your harness decides what is allowed, never the model policing itself." | ✅ |

## Regression checks

Each change below is checked against the agent-category questions in `data/questions.json`
(`scripts/agent_eval_runner.py`). Local-mode tool calling only works since `app/local_models.py`
was updated for Qwen's `<tool_call>` format (see the retry-limit/send-tool row) — before that,
local mode could not evaluate the agent at all.

| Change | Tool choice | Notes |
|---|---|---|
| Retry limit + `send_message` tool added | n/a (pre-dates local tool calling) | Verified in isolation: standalone mock test confirms the retry limit stops at exactly 3 consecutive failures; `send_message` checked directly for empty/normal/over-limit text and confirmed via AST inspection that it never calls `send_to_teams()` itself. |
| `get_definition` + `date_diff` tools added (6 tools total) | — | Rayyan: both tools tested directly against the real corpus (all 68 Article 3 definitions found, spelling variants, unknown terms, bad dates) and the agent loop with the mock client (tools offered, dispatched correctly, retry limit stops after 3 failed lookups). Not yet tested with `gpt-4o-mini`. |
| Local Qwen tool calling added, `--mode local` allowed in the eval runner | 2/2 | Rayyan: two probe questions in local mode, both chose the right tool. |
| Injection-against-actions guard added | — | Rayyan, local mode (Qwen2.5-3B): before hardening 3/3 poisoned runs called `send_message` and reached approval; after prompt hardening alone, still 3/3; after hardening + code guard, 0/3 — guard is what actually stops it. Control question (user genuinely asks to send) still reaches approval, 1/1. |

**Still open**: none of the rows above have been run in `--mode api` (`gpt-4o-mini`) yet — only
local mode and the zero-cost mock harness. Re-run `scripts/agent_eval_runner.py --mode api` with a
small `--limit` before reporting final agent numbers, the same way the RAG side's local-mode
results were re-checked in api mode.
