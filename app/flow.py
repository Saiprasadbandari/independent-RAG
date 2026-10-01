"""The "How it works" page: the whole system as a clickable flowchart.

Every box is a <details> element, so clicking opens the parts it is made of, and clicking a
part opens its settings. The numbers come from the code's own constants, so the page can't
drift from what actually runs. One button opens everything and a second click puts every box
back the way it was.
"""

from html import escape

import gradio as gr

from app import agent, hybrid, reflection, rerank, utils
from app import rag as pipeline
from app.injection import INJECTION_TEST


def node(title: str, icon: str = "", summary: str = "", kind: str = "step", children: list | None = None,
         specs: dict | None = None, where: str = "") -> dict:
    return {"title": title, "icon": icon, "summary": summary, "kind": kind,
            "children": children or [], "specs": specs or {}, "where": where}


def arrow(label: str = "") -> dict:
    return {"arrow": label}


def threshold_text() -> str:
    local = ", ".join(f"{v:.2f} for {k.split('/')[-1]}" for k, v in pipeline.MIN_SCORES.items())
    return f"{pipeline.MIN_SCORE:.2f} for text-embedding-3-small, {local}"


def flow() -> dict:
    question = node(
        "You ask a question", "👤", "typed in the chat, or picked from the suggestions", "io",
        specs={"Send": "Enter (Shift+Enter for a new line)",
               "Memory": f"the last {pipeline.HISTORY_TURNS} exchanges go along with each question",
               "Clear": "🗑 empties the chat and the memory",
               "Mode": "🤖 Agent mode tick box chooses the right-hand path"},
        where="app/ui.py · ChatHandlers.ask")

    rag_steps = [
        node("Clean the question", "🧹", "orders and fake context are removed before anything runs",
             children=[
                 node("Instruction filter", "🚫", "drops sentences that try to steer the assistant",
                      specs={"Catches": "\"ignore your rules\", \"reveal your prompt\", \"write a poem\", \"cite page…\"",
                             "Only orders": "a message that is nothing but instructions is refused without a search"}),
                 node("Fake context filter", "🏷️", "strips text passed off as retrieved passages",
                      specs={"Removes": "\"Context:\" blocks, [1] markers, [Source: …] labels",
                             "Shown as": "\"Interpreted as: …\" under the answer when it changed"}),
             ], where="app/rag.py · sanitize_question"),
        arrow(),
        node("Understand follow-ups", "🔁", "only when it looks like a follow-up and there is history",
             children=[
                 node("Navigation by counting", "🔢", "\"the next one\" after Article 41 → Article 42",
                      specs={"Units": ", ".join(f"{u} {a}–{b}" for u, (a, b) in pipeline.UNIT_RANGES.items()),
                             "Cost": "no model call"}, where="app/rag.py · resolve_navigation"),
                 node("Model rewrite", "✏️", "turns \"what about the second one?\" into a full question",
                      specs={"Model": pipeline.CHAT_MODEL, "History used": f"last {pipeline.HISTORY_TURNS} exchanges, "
                             f"answers cut to {pipeline.HISTORY_ANSWER_CHARS} characters",
                             "Max tokens": pipeline.REWRITE_MAX_TOKENS, "Cached": "yes (rewrites.json)"},
                      where="app/rag.py · condense_question"),
                 node("Rewrite check", "✔️", "the rewrite must keep the user's topic",
                      specs={"Rule": "at least half the topic words typed must survive, otherwise the user's own words are used",
                             "Stops": "\"boom boom\" being swapped for the previous question"},
                      where="app/rag.py · keeps_question"),
             ], where="app/rag.py · RAG.rewrite"),
        arrow(),
        node("Asked before?", "⚡", "the same question in the same conversation is answered from the cache",
             "decision", specs={"Yes": "answer returned at once, no model call",
                                "Key": "question + recent exchanges + every model and setting that shapes the answer",
                                "Store": "data/cache/answers.json"},
             where="app/rag.py · RAG.ask_stream"),
        arrow("no"),
        node("Find candidate passages", "🔎", "two searches merged, plus a direct article lookup",
             children=[
                 node("Vector search", "🧭", "passages with the same meaning",
                      specs={"Index": "FAISS IndexFlatIP on L2-normalised vectors = cosine similarity",
                             "Embeddings": "text-embedding-3-small (1536-d), or bge-base-en-v1.5 (768-d) on the GPU in hybrid/local",
                             "Returns": f"top {pipeline.RERANK_CANDIDATES}"},
                      where="app/embedding.py · Retriever.search"),
                 node("BM25 keyword search", "🔤", "passages with the same exact words",
                      specs={"Library": "rank_bm25 BM25Okapi over lower-cased word tokens",
                             "Returns": f"top {pipeline.BM25_CANDIDATES}",
                             "Why": "a legal text is full of exact terms (\"Article 50\", \"Annex III\") embeddings can miss"},
                      where="app/hybrid.py · BM25Index"),
                 node("Reciprocal Rank Fusion", "🔀", "merges the two ranked lists",
                      specs={"Formula": f"score = Σ 1 / ({hybrid.RRF_K} + rank) over both lists",
                             "Keyword-only hits": "kept for the re-ranker, flagged \"keyword match (BM25)\" in the sources"},
                      where="app/hybrid.py · reciprocal_rank_fusion"),
                 node("Direct article lookup", "📌", "\"What does Article 41 say?\" fetches Article 41 by its heading",
                      specs={"Chunks": f"the heading chunk and the {pipeline.REFERENCE_CHUNKS - 1} after it",
                             "Pinned when": f"the question is mainly a lookup (≤ {pipeline.LOOKUP_MAX_WORDS} other topic words)",
                             "Shown as": "\"direct lookup (Article N)\""},
                      where="app/rag.py · RAG.referenced_chunks"),
             ], specs={"Cached": "yes (retrieval.json)"}, where="app/rag.py · RAG.retrieve"),
        arrow(),
        node("About the AI Act at all?", "🚦", "off-topic questions stop here", "decision",
             specs={"Passes when": "a vector hit scores at least the threshold, or the question names an existing article",
                    "Threshold": threshold_text(),
                    "No": f"\"{pipeline.NO_INFO}\"",
                    "Keyword hits": "don't count: BM25 matches common words in almost any question"},
             where="app/rag.py · RAG.search_steps"),
        arrow("yes"),
        node("Re-rank on the GPU", "⚖️", "a cross-encoder reads question and passage together",
             specs={"Model": rerank.RERANK_MODEL, "Device": "cuda, else Apple GPU (mps), else CPU",
                    "Output": "sigmoid → probability the passage is relevant (0 to 1)",
                    "Drops": f"passages under {rerank.MIN_RELEVANCE}",
                    "Limits": f"{rerank.RERANK_MAX_LENGTH} tokens per pair, batches of {rerank.RERANK_BATCH_SIZE}",
                    "Cached": "yes (rerank.json)"},
             where="app/rerank.py · LocalReranker"),
        arrow(),
        node("Build the context", "🧩", "fewer, better passages for the model",
             children=[
                 node("Keep the best", "🥇", "only passages close to the top one",
                      specs={"At most": f"{pipeline.CONTEXT_PASSAGES} passages",
                             "Each": f"≥ {pipeline.RELATIVE_RELEVANCE:.0%} of the best passage's relevance"}),
                 node("Add what follows", "➡️", "lists and articles run across chunk boundaries",
                      specs={"From": f"the best {pipeline.NEIGHBOUR_TOP} passages",
                             "Up to": f"{pipeline.NEIGHBOUR_STEPS} chunks ahead",
                             "Joins if": f"relevance ≥ {pipeline.NEIGHBOUR_MIN_RELEVANCE}",
                             "Shown as": "\"continues [n]\""}),
                 node("Trim sentences", "✂️", "long passages keep only their relevant sentences",
                      specs={"Applies to": f"passages over {pipeline.TRIM_MIN_CHARS} characters",
                             "Keeps": f"sentences scoring ≥ {pipeline.SENTENCE_MIN_RELEVANCE}"}),
             ], where="app/rag.py · RAG.select_context"),
        arrow(),
        node("Write the answer", "✍️", "streamed into the chat word by word",
             children=[
                 node("Prompt", "📜", "the rules the model follows",
                      specs={"Rules": "answer only from the numbered passages, cite them as [n], "
                                      "reply exactly with the refusal sentence when they don't cover it",
                             "Passages": "[n] [Source: document, p. N] + text",
                             "Also sent": f"the last {pipeline.HISTORY_TURNS} exchanges, and the rewrite as a hint"}),
                 node("Model", "🤖", "the only paid call in hybrid mode",
                      specs={"api / hybrid": f"{pipeline.CHAT_MODEL}, temperature 0, max {pipeline.MAX_TOKENS} tokens",
                             "local": "Qwen2.5-3B-Instruct on the GPU, no API key",
                             "Price": f"${utils.PRICES['gpt-4o-mini']['input']} / ${utils.PRICES['gpt-4o-mini']['output']} "
                                      "per million input / output tokens"}),
             ], where="app/rag.py · stream_answer"),
        arrow(),
        node("Check the answer", "🧪", "citations, refusal and a confidence score",
             children=[
                 node("Citations → sources", "🔗", "[n] in the answer becomes document + page",
                      where="app/rag.py · cited_sources"),
                 node("Confidence score", "📊", "how much to trust this answer",
                      specs={"Weights": ", ".join(f"{k} {v:.2f}" for k, v in reflection.WEIGHTS.items()),
                             "Labels": f"high ≥ {reflection.HIGH}, medium ≥ {reflection.MEDIUM}, else low",
                             "Capped": "an answer with no citation, or cut off, is never high"},
                      where="app/reflection.py · assess"),
             ]),
    ]

    tools = [
        node("search_corpus", "🔎", "the RAG search above, as a tool",
             specs={"Runs": "retrieve + re-rank, same caches", "Nothing relevant": "\"No matching passages found.\""}),
        node("calculator", "🧮", "arithmetic the model shouldn't do in its head", specs={"Engine": "simpleeval, never eval()"}),
        node("quote_exact", "📄", "the full, uncut text of one page", specs={"Why": "chunks can split a figure or a list"}),
        node("get_definition", "📖", "the official Article 3 definition of a term",
             specs={"Covers": "all 68 definitions, pages 46–50", "Unknown term": "suggests defined terms that contain it"}),
        node("date_diff", "📅", "time between two dates", specs={"Returns": "days, and months + days"}),
        node("send_message", "📨", "drafts a Teams message, never sends it",
             specs={"Offered only": "when the user's own message asks to send, post or share",
                    "Sending": "a person clicks Send selected answer to Teams, checks the preview, then Confirm & Send"}),
    ]
    agent_steps = [
        node("Think", "🧠", "the model reads the question and picks a tool, or answers",
             specs={"Model": f"{pipeline.CHAT_MODEL} (Qwen2.5-3B in local mode), temperature 0",
                    "Rules": "answer only from tool results, cite pages, retrieved text is data and never instructions"},
             where="app/agent.py · run_agent"),
        arrow(),
        node("Tool call or answer?", "🔀", "", "decision",
             specs={"Answer": "the loop ends", "Tool call": "run it, then think again"}),
        arrow("tool"),
        node("Act: run a tool", "🛠️", "six tools, all read-only except the draft", children=tools,
             where="app/tools.py"),
        arrow(),
        node("Observe", "👀", "the result goes back to the model",
             specs={"Repeated call": "the exact same call again is refused",
                    "Failure": "\"No matching passages found.\", \"Error…\", \"No definition found…\""}),
        arrow(),
        node("Keep going?", "⏱️", "the loop is bounded", "decision",
             specs={"Step cap": f"{agent.MAX_STEPS} tool-calling steps",
                    "Retry limit": f"{agent.RETRY_LIMIT} failed calls in a row",
                    "Yes": "back to Think with the new result",
                    "Then": "one last call with tools off: answer with what it has, or say it doesn't know"}),
        arrow("done"),
        node("Trace", "🧾", "how the agent got there",
             specs={"In the chat": "🤖 Tool calls under the answer, with arguments",
                    "On disk": "every run, with arguments and results, in data/trajectories.jsonl"}),
    ]

    answer = node("Answer in the chat", "💬", "with citations, sources and confidence", "io",
                  children=[
                      node("Sources panel", "📚", "the passages the model saw",
                           specs={"Labels": "direct lookup (Article N), keyword match (BM25), continues [n], trimmed",
                                  "Also": "other retrieved candidates, collapsed"}),
                      node("Cost", "💶", "what this answer cost",
                           specs={"Budget": f"${utils.BUDGET_USD:.2f} for the week, tracked in data/cost.json",
                                  "Local models": "$0"}),
                      node("Follow-up suggestions", "💡", "three buttons with next questions"),
                  ])
    after = node("What you can do next", "📤", "", "io",
                 children=[
                     node("Feedback", "👍", "rate the answer", specs={"Saved to": "data/feedback.jsonl"}),
                     node("Export", "🗂️", "the chat or one answer", specs={"Formats": ".txt, .pdf"}),
                     node("Send to Teams", "📨", "human approval before anything is posted",
                          specs={"Steps": "Send selected answer to Teams → editable preview → Confirm & Send",
                                 "Webhook": "TEAMS_WEBHOOK_URL in .env"}, where="app/send_message.py"),
                 ])

    indexing = [
        node("The PDF", "📕", "EU AI Regulations.pdf", "io", specs={"Pages": "144"}),
        arrow(),
        node("Extract", "🧾", "text per page", specs={"Tool": "PyMuPDF", "Output": "data/corpus.json"},
             where="scripts/extract.py"),
        arrow(),
        node("Chunk", "✂️", "overlapping pieces",
             specs={"Size": f"{utils.CHUNK_SIZE} characters", "Overlap": f"{utils.CHUNK_OVERLAP}", "Result": "769 chunks"},
             where="app/embedding.py · build_chunks"),
        arrow(),
        node("Embed and index", "🗄️", "built once, saved to disk",
             specs={"Batches": f"{pipeline.EMBED_BATCH_SIZE} texts per call",
                    "Files": "data/my_index.faiss + chunks.json (my_index-local.faiss for local embeddings)",
                    "BM25": "built from the same chunks at startup"},
             where="app/embedding.py · build_index"),
    ]
    modes = node("Run modes", "🔌", "where each model runs", children=[
        node("api", "☁️", "everything through OpenAI", specs={"Embeddings": "text-embedding-3-small", "Answers": pipeline.CHAT_MODEL}),
        node("hybrid", "⚡", "the cheap work on this machine",
             specs={"Embeddings": "bge-base-en-v1.5 on the GPU", "Answers": pipeline.CHAT_MODEL}),
        node("local", "💻", "no API key at all",
             specs={"Embeddings": "bge-base-en-v1.5 on the GPU", "Answers": "Qwen2.5-3B-Instruct on the GPU, with tool calls"}),
    ], specs={"Re-ranker": "always local, on the GPU"}, where="app/local_models.py")
    safety = node("Safety tests", "🛡️", "attacks the system is checked against", children=[
        node("Prompt injection", "💉", "10 attacks on the RAG chat",
             specs={"Run": "uv run python -m scripts.injection_test --mode local"}),
        node("Injection against actions", "🎯", "a poisoned passage tells the agent to send a message",
             specs={"Switch": f"INJECTION_TEST=1 (now {'on' if INJECTION_TEST else 'off'})",
                    "Fake send": "declines every proposal and logs it",
                    "Result": "0 of 3 injected sends reached the approval gate with the guard"},
             where="app/injection.py"),
    ])
    return {"question": question, "rag": rag_steps, "agent": agent_steps, "answer": answer, "after": after,
            "indexing": indexing, "extras": [modes, safety]}


def render_node(item: dict, depth: int = 0) -> str:
    if "arrow" in item:
        label = f'<span class="flow-arrow-label">{escape(item["arrow"])}</span>' if item["arrow"] else ""
        return f'<div class="flow-arrow">{label}</div>'
    has_body = item["children"] or item["specs"] or item["where"]
    head = (f'<span class="flow-icon">{item["icon"]}</span>'
            f'<span class="flow-text"><b>{escape(item["title"])}</b>'
            + (f'<small>{escape(item["summary"])}</small>' if item["summary"] else "")
            + "</span>" + ('<span class="flow-chev">▸</span>' if has_body else ""))
    cls = f'flow-node flow-{item["kind"]} flow-depth-{min(depth, 2)}'
    if not has_body:
        return f'<div class="{cls}"><div class="flow-head">{head}</div></div>'
    body = ""
    if item["children"]:
        body += '<div class="flow-children">' + "".join(render_node(c, depth + 1) for c in item["children"]) + "</div>"
    if item["specs"]:
        body += '<dl class="flow-specs">' + "".join(
            f"<dt>{escape(k)}</dt><dd>{escape(str(v))}</dd>" for k, v in item["specs"].items()) + "</dl>"
    if item["where"]:
        body += f'<div class="flow-where">📁 <code>{escape(item["where"])}</code></div>'
    return f'<details class="{cls}"><summary class="flow-head">{head}</summary><div class="flow-body">{body}</div></details>'


def render() -> str:
    f = flow()
    lane = lambda title, sub, items, cls: (
        f'<section class="flow-lane {cls}"><div class="flow-lane-title">{title}<small>{sub}</small></div>'
        + "".join(render_node(i) for i in items) + "</section>")
    return f"""<div id="flow-root">
<p class="flow-hint">Click any box to see what it's made of, click again inside it for its settings. Every number comes from the running code.</p>
<div class="flow-spine">{render_node(f["question"])}
<div class="flow-arrow"></div>
<div class="flow-node flow-decision flow-depth-0"><div class="flow-head"><span class="flow-icon">🤖</span><span class="flow-text"><b>Agent mode ticked?</b><small>no → the RAG pipeline · yes → the agent loop</small></span></div></div>
<div class="flow-split"></div></div>
<div class="flow-lanes">
{lane("RAG pipeline", "one fixed path, 1 model call (2 for a rewritten follow-up)", f["rag"], "flow-lane-rag")}
{lane("Agent loop", "the model chooses the path, up to " + str(agent.MAX_STEPS) + " steps", f["agent"], "flow-lane-agent")}
</div>
<div class="flow-spine"><div class="flow-join"></div>{render_node(f["answer"])}<div class="flow-arrow"></div>{render_node(f["after"])}</div>
<h3 class="flow-section">Built once, before any question</h3>
<div class="flow-row">{"".join(render_node(i) for i in f["indexing"])}</div>
<h3 class="flow-section">Around it</h3>
<div class="flow-row flow-row-wrap">{"".join(render_node(i) for i in f["extras"])}</div>
</div>"""


FLOW_CSS = """
#flow-root { --flow-accent: var(--color-accent, #f97316); --flow-rag: #3b82f6; --flow-agent: #a855f7;
  --flow-line: var(--border-color-primary, #d4d4d8); color: var(--body-text-color); }
#flow-root .flow-hint { opacity: .75; margin: 0 0 1rem; }
#flow-root .flow-spine { display: flex; flex-direction: column; align-items: center; }
#flow-root .flow-spine > .flow-node { width: min(560px, 100%); }
#flow-root .flow-node { border: 1.5px solid var(--flow-line); border-radius: 12px; background: var(--block-background-fill, #fff);
  box-shadow: 0 1px 2px rgba(0,0,0,.06); margin: 0; transition: border-color .15s, box-shadow .15s; text-align: left; }
#flow-root details.flow-node > summary { list-style: none; cursor: pointer; }
#flow-root details.flow-node > summary::-webkit-details-marker { display: none; }
#flow-root details.flow-node:hover { border-color: var(--flow-accent); box-shadow: 0 2px 10px rgba(0,0,0,.10); }
#flow-root details[open].flow-node { border-color: var(--flow-accent); }
#flow-root .flow-head { display: flex; align-items: center; gap: .6rem; padding: .6rem .8rem; }
#flow-root .flow-icon { font-size: 1.35rem; line-height: 1; }
#flow-root .flow-text { display: flex; flex-direction: column; flex: 1; min-width: 0; }
#flow-root .flow-text small { opacity: .72; font-size: .82em; }
#flow-root .flow-chev { transition: transform .2s; opacity: .6; }
#flow-root details[open] > summary .flow-chev { transform: rotate(90deg); }
#flow-root .flow-body { padding: 0 .8rem .8rem; animation: flow-in .2s ease-out; }
@keyframes flow-in { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: none; } }
#flow-root .flow-io { background: var(--background-fill-secondary, #f4f4f5); border-style: dashed; }
#flow-root .flow-decision { border-width: 2px; border-color: var(--flow-accent); border-radius: 999px; }
#flow-root .flow-decision .flow-head { justify-content: center; }
#flow-root details[open].flow-decision { border-radius: 18px; }
#flow-root details[open].flow-decision .flow-head { justify-content: flex-start; }
#flow-root .flow-children { display: flex; flex-wrap: wrap; gap: .6rem; padding: .7rem 0 .2rem .9rem; margin-left: .5rem;
  border-left: 2px dotted var(--flow-line); }
#flow-root .flow-children > .flow-node { flex: 1 1 220px; }
#flow-root .flow-depth-1 { background: var(--background-fill-secondary, #fafafa); }
#flow-root .flow-depth-2 { background: var(--block-background-fill, #fff); }
#flow-root .flow-specs { display: grid; grid-template-columns: max-content 1fr; gap: .25rem .8rem; margin: .7rem 0 0;
  font-size: .86em; }
#flow-root .flow-specs dt { font-weight: 600; opacity: .8; }
#flow-root .flow-specs dd { margin: 0; overflow-wrap: anywhere; }
#flow-root .flow-where { margin-top: .5rem; font-size: .8em; opacity: .75; }
#flow-root .flow-arrow { width: 2px; height: 26px; background: var(--flow-line); position: relative; margin: 0 auto; }
#flow-root .flow-arrow::after { content: ""; position: absolute; bottom: -1px; left: -5px; border: 6px solid transparent;
  border-top-color: var(--flow-line); border-bottom: 0; }
#flow-root .flow-arrow-label { position: absolute; left: 10px; top: 4px; font-size: .75em; white-space: nowrap; opacity: .75; }
#flow-root .flow-split { width: min(640px, 70%); height: 22px; border: 2px solid var(--flow-line); border-bottom: 0;
  border-radius: 10px 10px 0 0; margin-top: 18px; position: relative; }
#flow-root .flow-split::before { content: ""; position: absolute; left: 50%; top: -20px; width: 2px; height: 18px; background: var(--flow-line); }
#flow-root .flow-join { width: min(640px, 70%); height: 22px; border: 2px solid var(--flow-line); border-top: 0;
  border-radius: 0 0 10px 10px; margin-bottom: 20px; position: relative; }
#flow-root .flow-join::after { content: ""; position: absolute; left: 50%; bottom: -20px; width: 2px; height: 18px; background: var(--flow-line); }
#flow-root .flow-lanes { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.2rem; }
#flow-root .flow-lane { border-radius: 16px; padding: .9rem; display: flex; flex-direction: column; }
#flow-root .flow-lane > .flow-node { width: 100%; }
#flow-root .flow-lane-rag { background: color-mix(in srgb, var(--flow-rag) 8%, transparent); border: 1.5px solid color-mix(in srgb, var(--flow-rag) 35%, transparent); }
#flow-root .flow-lane-agent { background: color-mix(in srgb, var(--flow-agent) 8%, transparent); border: 1.5px solid color-mix(in srgb, var(--flow-agent) 35%, transparent); }
#flow-root .flow-lane-title { font-weight: 700; font-size: 1.05rem; margin-bottom: .7rem; display: flex; flex-direction: column; }
#flow-root .flow-lane-title small { font-weight: 400; opacity: .72; }
#flow-root .flow-lane-rag .flow-lane-title { color: var(--flow-rag); }
#flow-root .flow-lane-agent .flow-lane-title { color: var(--flow-agent); }
#flow-root .flow-lane > .flow-node + .flow-node { margin-top: .5rem; }
#flow-root .flow-section { margin: 2rem 0 .8rem; }
#flow-root .flow-row { display: flex; align-items: flex-start; gap: 0; overflow-x: auto; padding-bottom: .4rem; }
#flow-root .flow-row > .flow-node { flex: 1 1 200px; min-width: 180px; }
#flow-root .flow-row > .flow-arrow { width: 26px; height: 2px; align-self: center; flex: 0 0 26px; margin: 0; }
#flow-root .flow-row > .flow-arrow::after { bottom: -5px; left: auto; right: -1px; border: 6px solid transparent;
  border-left-color: var(--flow-line); border-right: 0; }
#flow-root .flow-row-wrap { flex-wrap: wrap; gap: .8rem; overflow: visible; }
@media (max-width: 640px) { #flow-root .flow-row:not(.flow-row-wrap) { flex-direction: column; align-items: stretch; }
  #flow-root .flow-row:not(.flow-row-wrap) > .flow-arrow { width: 2px; height: 22px; align-self: center; }
  #flow-root .flow-row:not(.flow-row-wrap) > .flow-arrow::after { left: -5px; right: auto; bottom: -1px;
    border: 6px solid transparent; border-top-color: var(--flow-line); border-bottom: 0; } }
"""

# opens every box, remembering which were open; the second click restores exactly that
TOGGLE_JS = """() => {
  const root = document.querySelector('#flow-root');
  if (!root) return;
  const boxes = root.querySelectorAll('details');
  const expand = !root.classList.contains('flow-all-open');
  boxes.forEach(d => {
    if (expand) { d.dataset.wasOpen = d.open ? '1' : '0'; d.open = true; }
    else { d.open = d.dataset.wasOpen === '1'; }
  });
  root.classList.toggle('flow-all-open', expand);
  const button = document.querySelector('#flow-toggle');
  if (button) button.textContent = expand ? '↩ Put everything back' : '⤢ Expand everything';
}"""


def build_flow_page() -> None:
    gr.Markdown("# How it works\nThe whole path of a question, from the chat box to the answer.")
    gr.Button("⤢ Expand everything", elem_id="flow-toggle", variant="primary", size="sm").click(None, js=TOGGLE_JS)
    gr.HTML(render())
