"""The retrieval trace: what happened to one RAG question, step by step, for the UI's Trace tab.

RAG.ask_stream records the decisions in Answer.trace (rewrite, caches, the off-topic gate, the
re-ranker's cut-off, time per stage); each retrieved chunk carries where it came from (its rank in
FAISS and in BM25, after fusion, and its re-ranker relevance). This module turns both into
Markdown: a timeline, then a table of every candidate and what became of it.
"""

from app.rag import cited_refs

EMPTY_TRACE = ("### 🔍 Retrieval trace\n_Ask a question in RAG mode to see each step: the FAISS and BM25 "
               "searches, the fusion, the off-topic gate, the re-ranker and the time each one took._")


def seconds(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 1000:.0f} ms" if value < 1 else f"{value:.1f} s"


def rank_cell(rank: int | None, score: float | None, digits: int = 2) -> str:
    if rank is None:
        return "—"
    return f"#{rank}" + (f" · {score:.{digits}f}" if score is not None else "")


def outcome(c: dict, i: int, context_size: int, ids: dict, cited: set, trace: dict) -> str:
    """What became of one candidate."""
    if i <= context_size:
        if "neighbour_of" in c:
            label = f"➕ added after [{ids.get(c['neighbour_of'], '?')}]"
        elif c.get("reference"):
            label = f"📌 direct lookup ({c['reference']})"
        else:
            label = f"✅ sent as [{i}]"
        if c.get("trimmed"):
            label += " · ✂️ trimmed"
        if i in cited:
            label += " · cited"
        return label
    gate = trace.get("gate", {})
    if not gate.get("passed", True):
        return "⛔ question stopped at the gate"
    if "relevance" in c:
        floor = trace.get("floor", {}).get("value")
        if floor is not None and c["relevance"] < floor:
            return f"❌ below the cut-off ({floor:.2f})"
        return f"❌ past the {trace.get('floor', {}).get('max_passages', 5)}-passage limit"
    if not c.get("keyword_only") and c["score"] < gate.get("threshold", 0):
        return f"⛔ similarity under {gate.get('threshold', 0):.2f}, not re-ranked"
    return "—"


def timeline(turn: dict, answer: dict, trace: dict) -> list[str]:
    timings = trace.get("timings", {})
    rows = []

    def row(step: str, what: str, took: float | None = None):
        rows.append(f"| {step} | {what} | {seconds(took)} |")

    asked, searched = turn["question"], answer["standalone_question"]
    row("1. Clean the question", "instructions or fake context removed" if trace.get("cleaned") else "nothing to remove")
    kind = trace.get("rewrite")
    if kind == "navigation":
        row("2. Follow-up", f"resolved by counting → “{searched}” (no model call)", timings.get("rewrite"))
    elif kind == "model":
        row("2. Follow-up", f"rewritten by the model → “{searched}”", timings.get("rewrite"))
    else:
        row("2. Follow-up", "not a follow-up, searched as asked" if searched == asked else f"searched for “{searched}”")
    if trace.get("answer_cached"):
        row("3. Answer cache", "**hit**: the same question was answered before, served from the cache "
            "(the steps below are from that first run)", trace.get("served_in"))
    else:
        row("3. Answer cache", "miss, running the pipeline")

    retrieved = answer["retrieved"]
    vector = sum("vector_rank" in c for c in retrieved)
    keyword = sum("bm25_rank" in c for c in retrieved)
    if trace.get("retrieval_cached"):
        row("4. Search", f"FAISS {vector} + BM25 {keyword} hits, from the retrieval cache")
    else:
        row("4a. Vector search", f"question embedded, FAISS returned {vector} passages", timings.get("embed + FAISS"))
        row("4b. Keyword search", f"BM25 returned {keyword} passages, merged by Reciprocal Rank Fusion",
            timings.get("BM25 + fusion"))
    if trace.get("lookup"):
        row("5. Direct lookup", ", ".join(trace["lookup"]) + (" (pinned first)" if trace.get("pinned") else
                                                              " (added as candidates)"))
    gate = trace.get("gate")
    if gate:
        best, limit = gate.get("best_similarity"), gate["threshold"]
        best_text = f"best similarity {best:.2f}" if best is not None else "no vector hit"
        if gate["passed"] and gate.get("by_lookup") and (best is None or best < limit):
            verdict = f"✅ on topic: names an article ({best_text})"
        elif gate["passed"]:
            verdict = f"✅ on topic: {best_text} ≥ {limit:.2f}"
        else:
            verdict = f"❌ off topic: {best_text} < {limit:.2f}, refused without calling the model"
        row("6. Off-topic gate", verdict)
    if "reranked" in trace:
        r, floor = trace["reranked"], trace.get("floor")
        what = f"{r['scored']} scored on the GPU" + (f", {r['cached']} from the cache" if r["cached"] else "")
        if floor:
            what += (f"; keep relevance ≥ {floor['value']:.2f} (the larger of {floor['min_relevance']} and "
                     f"{floor['relative']:.0%} of the best, {floor['best']:.2f}), at most {floor['max_passages']}")
        row("7. Re-rank", what, timings.get("re-rank"))
    context = retrieved[:answer["context_size"]]
    if gate is None or gate.get("passed"):
        added = sum("neighbour_of" in c for c in context)
        trimmed = sum(bool(c.get("trimmed")) for c in context)
        row("8. Build the context", f"{len(context) - added} passage(s)"
            + (f" + {added} following chunk(s)" if added else "") + " sent to the model"
            + (f"; {trimmed} trimmed to their relevant sentences" if trimmed else ""),
            timings.get("build context"))
    models = trace.get("models", {})
    if answer.get("input_tokens"):
        first = trace.get("first_words")
        row("9. Write the answer", f"{models.get('answers', 'model')}: {answer['input_tokens']} tokens in, "
            f"{answer['output_tokens']} out" + (f", first words after {seconds(first)}" if first else ""),
            timings.get("write answer"))
    elif not context:
        row("9. Write the answer", "no passage to answer from: refusal returned without a model call")
    else:
        row("9. Write the answer", f"{models.get('answers', 'local model')} (local, no token count)",
            timings.get("write answer"))
    cited = len(answer.get("sources", []))
    row("10. Check the answer", f"confidence {answer['confidence']:.0%} ({answer['confidence_label']}), "
        f"{cited} source(s) cited", timings.get("check"))
    return rows


def candidates_table(answer: dict, trace: dict) -> list[str]:
    retrieved, size = answer["retrieved"], answer["context_size"]
    context = retrieved[:size]
    ids = {c["id"]: i for i, c in enumerate(context, 1)}
    cited = set(cited_refs(answer["text"], context)[0]) if answer["in_scope"] else set()
    judged = sorted((c for c in retrieved if "relevance" in c), key=lambda c: c["relevance"], reverse=True)
    rerank_rank = {c["id"]: i for i, c in enumerate(judged, 1)}
    lines = ["| Page | FAISS (cosine) | BM25 | Fused | Re-rank | What happened |", "|---|---|---|---|---|---|"]
    for i, c in enumerate(retrieved, 1):
        faiss = rank_cell(c.get("vector_rank"), None if c.get("keyword_only") else c.get("score"))
        bm25 = rank_cell(c.get("bm25_rank"), c.get("bm25_score"), 1)
        fused = f"#{c['fused_rank']}" if "fused_rank" in c else "—"
        rerank = f"{c['relevance']:.2f} (#{rerank_rank[c['id']]})" if "relevance" in c else "—"
        lines.append(f"| p. {c['page']} | {faiss} | {bm25} | {fused} | {rerank} | "
                     f"{outcome(c, i, size, ids, cited, trace)} |")
    return lines


def format_trace(turns: list[dict], index: int | None) -> str:
    if index is None or not turns:
        return EMPTY_TRACE
    turn = turns[index]
    heading = f"### 🔍 Retrieval trace for question {index + 1}"
    if "error" in turn:
        return f"{heading}\n_The question failed, nothing to trace._"
    if "agent_answer" in turn:
        return (f"{heading}\n_Agent mode: the model chose its own steps. Each `search_corpus` call runs this "
                "same retrieval; see the tool calls under the answer._")
    answer = turn["answer"]
    trace = answer.get("trace") or {}
    if trace.get("greeting"):
        return f"{heading}\n_A greeting: answered directly, no search and no model call ({seconds(trace.get('total'))})._"
    if not trace:
        return f"{heading}\n_No trace recorded for this answer (it was cached before tracing was added)._"

    models = trace.get("models", {})
    parts = [heading, f"**Question:** {turn['question']}"]
    if models:
        parts.append("**Models:** " + " · ".join(f"{k} `{v}`" for k, v in models.items() if v))
    parts.append("| Step | What happened | Time |\n|---|---|---|\n" + "\n".join(timeline(turn, answer, trace)))
    total = trace.get("served_in") if trace.get("answer_cached") else trace.get("total")
    if total is not None:
        parts.append(f"**Total: {seconds(total)}**" + (" (from the cache)" if trace.get("answer_cached") else ""))
    if answer["retrieved"]:
        parts.append(f"#### Every candidate ({len(answer['retrieved'])})\nRanks in each search, the re-ranker's "
                     "relevance (0 to 1) and what became of each passage. FAISS and BM25 scores are on "
                     "different scales, which is why they're merged by rank.")
        parts.append("\n".join(candidates_table(answer, trace)))
    return "\n\n".join(parts)
