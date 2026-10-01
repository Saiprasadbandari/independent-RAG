import json
import random
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import gradio as gr
import openai

from app.agent import run_agent
from app.checklist import build_checklist_page
from app.flow import FLOW_CSS, build_flow_page
from app.export import export
from app.rag import RAG, cited_refs
from app.send_message import send_to_teams
from app.utils import DATA_DIR, format_citation, format_retrieved

FEEDBACK_PATH = DATA_DIR / "feedback.jsonl"
N_SUGGESTIONS = 3

EXAMPLES = [
    "What AI practices are prohibited under the AI Act?",
    "What counts as a high-risk AI system?",
    "What are the transparency obligations for providers of general-purpose AI models?",
    "What is the best pizza topping?",
]


def load_question_bank() -> list[dict]:
    """The eval set doubles as a free source of 'related question' suggestions --
    no extra API call, just picks from questions we already wrote."""
    path = DATA_DIR / "questions.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


QUESTION_BANK = load_question_bank()


def suggest_followups(question: str, asked: set[str], n: int = N_SUGGESTIONS) -> list[str]:
    """Free, zero-cost 'related questions' (Perplexity's 'Related' row, without the LLM call):
    score every bank question by shared keywords with the one just asked, using the same
    tokenizer BM25 already uses, so it generalizes to free-form questions not in the bank."""
    from app.hybrid import tokenize

    asked_words = set(tokenize(question))
    scored = []
    for q in QUESTION_BANK:
        if q["question"] in asked or q["question"] == question:
            continue
        overlap = len(asked_words & set(tokenize(q["question"])))
        scored.append((overlap, q["question"]))
    scored.sort(key=lambda t: t[0], reverse=True)

    picks = [q for score, q in scored[:n] if score > 0]
    if len(picks) < n:  # not enough overlap found -- top up with a random unused sample
        rest = [q["question"] for q in QUESTION_BANK
                if q["question"] not in asked and q["question"] != question and q["question"] not in picks]
        random.shuffle(rest)
        picks += rest[:n - len(picks)]
    while len(picks) < n:
        picks.append("")  # pad so the fixed set of buttons always has something to bind to
    return picks[:n]


def load_ablation_table() -> str:
    """Static table from the one-time day-2 analysis (scripts/finish_day2.py).
    Not live -- re-run that script and refresh this if the pipeline changes."""
    path = DATA_DIR / "day2_final.json"
    if not path.exists():
        return "_Not run yet — see `scripts/finish_day2.py`._"
    data = json.loads(path.read_text(encoding="utf-8"))
    ablation = data.get("ablation", {})
    if not ablation:
        return "_Not run yet._"
    rows = [f"| {name} | {s['recall@1']:.0%} | {s['recall@5']:.0%} | {s['mrr']:.3f} |"
            for name, s in ablation.items()]
    threshold = data.get("confidence_threshold")
    threshold_line = f"\n\nConfidence threshold: **{threshold:.3f}**" if threshold else ""
    return ("| Version | Recall@1 | Recall@5 | MRR |\n|---|---|---|---|\n" + "\n".join(rows) + threshold_line)

CONFIDENCE_ICONS = {"high": "🟢", "medium": "🟡", "low": "🔴"}
# shown in the answer bubble while RAG.ask_stream works through its stages
STAGE_TEXT = {
    "rewriting": "Understanding your follow-up…",
    "searching": "Embedding the question and searching the index…",
    "reranking": "Re-ranking and trimming passages on the GPU…",
    "generating": "Writing the answer…",
    "checking": "Checking confidence…",
}
EMPTY_SOURCES = "### Sources\n_Ask a question to see the passages each answer is based on._"


def format_confidence(answer: dict) -> str:
    if not answer["confidence_label"]:
        return ""
    subject = "Answer" if answer["in_scope"] else "Refusal"
    icon = CONFIDENCE_ICONS[answer["confidence_label"]]
    notes = " · ".join(answer["confidence_notes"])
    return f"{icon} **{subject} confidence: {answer['confidence']:.0%} ({answer['confidence_label']})** — {notes}"


def format_trajectory(steps: list[dict]) -> str:
    if not steps:
        return "_No tools were needed._"
    lines = []
    for i, step in enumerate(steps, 1):
        args = ", ".join(f"{k}={v!r}" for k, v in step["arguments"].items())
        result = step["result"] if len(step["result"]) <= 200 else step["result"][:200].rstrip() + "…"
        lines.append(f"{i}. `{step['tool']}({args})`\n   → {result}")
    return "\n".join(lines)


def format_chat_answer(turn: dict) -> str:
    if "error" in turn:
        return f"⚠️ **Error:** {turn['error']}"
    if "agent_answer" in turn:
        answer = turn["agent_answer"]
        parts = [answer["text"]]
        if answer["stopped_at_cap"]:
            parts.append("_(stopped at the step limit)_")
        if answer["gave_up_retrying"]:
            parts.append("_(gave up after repeated tool calls returned nothing useful)_")
        parts.append("**🤖 Tool calls:**\n" + format_trajectory(answer["trajectory"]))
        return "\n\n".join(p for p in parts if p)
    answer = turn["answer"]
    parts = []
    if answer["standalone_question"] != turn["question"]:
        parts.append(f"_Interpreted as: {answer['standalone_question']}_")
    parts.append(answer["text"])
    parts.append(format_confidence(answer))
    if answer["sources"]:
        parts.append("**Sources:** " + " · ".join(f"[{s['ref']}] {format_citation(s)}" for s in answer["sources"]))
    return "\n\n".join(p for p in parts if p)


def quote(text: str) -> str:
    return "> " + text.replace("\n", "\n> ")


def format_sources_panel(turns: list[dict], index: int | None) -> tuple[str, str]:
    """(passages the model saw, other candidates) for one turn."""
    if index is None or not turns:
        return EMPTY_SOURCES, ""
    turn = turns[index]
    heading = f"### Sources for question {index + 1}"
    if "error" in turn:
        return f"{heading}\n_No sources: the question failed._", ""
    if "agent_answer" in turn:
        return f"{heading}\n_Agent mode: see the tool-call trace under the answer instead._", ""

    answer = turn["answer"]
    context = answer["retrieved"][:answer["context_size"]]
    others = answer["retrieved"][answer["context_size"]:]
    cited = set(cited_refs(answer["text"], context)[0]) if answer["in_scope"] else set()

    blocks = [heading]
    if answer["standalone_question"] != turn["question"]:
        blocks.append(f"**Searched for:** {answer['standalone_question']}")
    if not context:
        blocks.append("_No passage was relevant enough to send to the model._")
    ids = {c["id"]: i for i, c in enumerate(context, 1)}
    for i, c in enumerate(context, 1):
        mark = "✅ cited" if i in cited else "not cited"
        relevance = f" · relevance {c['relevance']:.2f}" if "relevance" in c else ""
        if "neighbour_of" in c:
            origin = f" · continues [{ids.get(c['neighbour_of'], '?')}]"
        elif c.get("reference"):
            origin = f" · direct lookup ({c['reference']})"
        elif c.get("keyword_only"):
            origin = " · keyword match (BM25)"
        else:
            origin = f" · similarity {c['score']:.2f}"
        trimmed = " · ✂️ trimmed to relevant sentences" if c.get("trimmed") else ""
        blocks.append(f"**[{i}] {format_citation(c)}** — {mark}{relevance}{origin}{trimmed}\n\n{quote(c['text'])}")
    return "\n\n".join(blocks), format_retrieved(others) if others else "_None._"


def feedback_label(liked) -> str:
    if liked in (True, "Like"):
        return "helpful"
    if liked in (False, "Dislike"):
        return "not helpful"
    return str(liked)


class ChatHandlers:
    """Event handlers for the Gradio app. Conversation state lives in a gr.State list of turns:
    {"question", "answer": asdict(Answer)} or {"question", "error"}; turn i is chat messages 2i, 2i+1."""

    def __init__(self, rag: RAG, export_dir: Path, feedback_path: Path = FEEDBACK_PATH):
        self.rag = rag
        self.export_dir = export_dir
        self.feedback_path = feedback_path

    def cost_line(self, turn: dict | None = None) -> str:
        if not turn or "error" in turn:
            return self.rag.tracker.summary()
        if "agent_answer" in turn:
            return f"This answer: ${turn['agent_answer']['cost']:.5f} · {self.rag.tracker.summary()}"
        answer = turn["answer"]
        if answer["cached"]:
            this = "$0 (answer from cache)"
        else:
            tokens = f", {answer['input_tokens']} in / {answer['output_tokens']} out tokens" if answer["input_tokens"] else ""
            this = f"${answer['cost']:.5f}{tokens}"
        return f"This answer: {this} · {self.rag.tracker.summary()}"

    def cost_table(self, turn: dict | None = None) -> str:
        """Live cost, as a small table instead of one line -- updates after every answer."""
        tracker = self.rag.tracker
        this_cost = "$0.00000"
        this_tokens = "—"
        if turn and "agent_answer" in turn:
            this_cost = f"${turn['agent_answer']['cost']:.5f}"
        elif turn and "answer" in turn:
            answer = turn["answer"]
            this_cost = "$0 (cached)" if answer["cached"] else f"${answer['cost']:.5f}"
            if answer.get("input_tokens"):
                this_tokens = f"{answer['input_tokens']} in / {answer['output_tokens']} out"
        remaining = max(tracker.remaining, 0)
        rows = [
            ("This query", this_cost),
            ("Tokens", this_tokens),
            ("Total spent", f"${tracker.total:.4f}"),
            ("Budget left", f"${remaining:.2f} / ${tracker.budget:.0f}"),
            ("API calls", str(tracker.calls)),
        ]
        return "| | |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)

    def ask(self, message: str, chat: list, turns: list, agent_mode: bool = False, progress=gr.Progress()):
        """Generator: the reply bubble shows each stage, then the answer as it streams in.
        In agent mode, the loop runs non-streamed and the trace is shown once it's done."""
        if not message or not message.strip():
            raise gr.Error("Please enter a question.")
        question = message.strip()
        history = [(t["answer"]["standalone_question"] or t["question"], t["answer"]["text"])
                   for t in turns if "answer" in t]
        chat = chat + [{"role": "user", "content": question}, {"role": "assistant", "content": "⏳ Starting…"}]
        # skipped during streaming: turns, index, sources, others, cost_line, cost_table, 3x suggestion button
        waiting = (gr.skip(),) * 9
        yield ("", list(chat), *waiting)

        try:
            if agent_mode:
                progress(0.20, desc="Agent is thinking and choosing tools…")
                chat[-1] = {"role": "assistant", "content": "⏳ Agent is thinking and choosing tools…"}
                answer = run_agent(question, self.rag, self.rag.client, self.rag.tracker)
                turn = {"question": question, "agent_answer": asdict(answer)}
            else:
                for kind, value in self.rag.ask_stream(question, history):
                    if kind == "stage":
                        chat[-1] = {"role": "assistant", "content": f"⏳ {STAGE_TEXT[value]}"}
                    elif kind == "delta":
                        chat[-1] = {"role": "assistant", "content": value + " ▌"}
                    else:
                        turn = {"question": question, "answer": asdict(value)}
                        break
                    # a fresh list per update, so every streamed frame is its own snapshot
                    yield (gr.skip(), list(chat), *waiting)
        except (RuntimeError, openai.OpenAIError) as e:
            turn = {"question": question, "error": str(e)}

        turns = turns + [turn]
        chat[-1] = {"role": "assistant", "content": format_chat_answer(turn)}
        index = len(turns) - 1

        asked = {t["question"] for t in turns}
        picks = suggest_followups(question, asked)
        suggestion_updates = [gr.update(value=p, visible=bool(p)) for p in picks]

        yield (gr.skip(), chat, turns, index, *format_sources_panel(turns, index),
              self.cost_line(turn), self.cost_table(turn), *suggestion_updates)

    def select(self, turns: list, evt: gr.SelectData):
        index = evt.index if isinstance(evt.index, int) else evt.index[0]
        turn_index = min(index // 2, len(turns) - 1) if turns else None
        return (turn_index, *format_sources_panel(turns, turn_index))

    def feedback(self, turns: list, evt: gr.LikeData):
        index = evt.index if isinstance(evt.index, int) else evt.index[0]
        if not turns or index // 2 >= len(turns) or "answer" not in turns[index // 2]:
            return
        turn = turns[index // 2]
        answer = turn["answer"]
        context = answer["retrieved"][:answer["context_size"]]
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "feedback": feedback_label(evt.liked),
            "question": turn["question"],
            "standalone_question": answer["standalone_question"],
            "answer": answer["text"],
            "in_scope": answer["in_scope"],
            "confidence": answer["confidence"],
            "confidence_label": answer["confidence_label"],
            "cited_pages": [[s["source"], s["page"]] for s in answer["sources"]],
            "context_pages": [[c["source"], c["page"]] for c in context],
            "cached": answer["cached"],
        }
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self.feedback_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        gr.Info(f"Thanks, feedback saved ({record['feedback']}).")

    def export_conversation(self, turns: list, fmt: str):
        if not turns:
            raise gr.Error("Nothing to export yet.")
        numbered = list(enumerate(turns, 1))
        return str(export(numbered, "conversation", fmt, self.export_dir, "eu-ai-act-conversation"))

    def export_answer(self, turns: list, selected: int | None, fmt: str):
        if not turns:
            raise gr.Error("Nothing to export yet.")
        index = selected if selected is not None else len(turns) - 1
        return str(export([(index + 1, turns[index])], f"answer {index + 1}", fmt, self.export_dir,
                          f"eu-ai-act-answer-{index + 1}"))

    def clear(self):
        hidden = gr.update(value="", visible=False)
        return ([], [], None, EMPTY_SOURCES, "", None, self.rag.tracker.summary(),
               self.cost_table(None), hidden, hidden, hidden)

    def pick_suggestion(self, label: str):
        return label

    def prepare_send(self, turns: list, selected: int | None):
        """Step 1 of the approval gate: shows an editable preview, sends nothing yet."""
        if not turns:
            raise gr.Error("Nothing to send yet.")
        index = selected if selected is not None else len(turns) - 1
        turn = turns[index]
        if "agent_answer" in turn:
            # the agent's own draft (its last send_message call) if it made one, else its answer
            drafts = [s["arguments"].get("text", "") for s in turn["agent_answer"]["trajectory"]
                      if s["tool"] == "send_message" and s["result"].startswith("Drafted")]
            preview = drafts[-1] if drafts else turn["agent_answer"]["text"]
        elif "answer" in turn:
            preview = turn["answer"]["text"]
        else:
            raise gr.Error("Can't send a failed answer.")
        shown = gr.update(visible=True)
        return gr.update(value=preview, visible=True), shown, shown, ""

    def confirm_send(self, preview_text: str):
        """Step 2: only runs when the person clicks Confirm, on whatever text is in the
        (editable) preview box at that moment -- not necessarily the original answer verbatim."""
        result = send_to_teams(preview_text)
        hidden = gr.update(visible=False)
        return result, hidden, hidden, hidden


MINIMAL_CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
footer { display: none !important; }
#suggest-row button { font-size: 0.85em; text-align: left; white-space: normal; height: auto; }
""" + FLOW_CSS


def build_ui(rag: RAG) -> gr.Blocks:
    handlers = ChatHandlers(rag, Path(tempfile.mkdtemp(prefix="rag-export-")))

    with gr.Blocks(title="EU AI Act RAG") as demo:
        turns = gr.State([])
        selected = gr.State(None)

        gr.Markdown("# EU AI Act Assistant\n"
                    "Regulation (EU) 2024/1689 · follow-ups work · answers cite the pages they come from")

        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label=None, height=520, layout="bubble", show_label=False,
                                  feedback_options=("Like", "Dislike"), buttons=["copy"],
                                  placeholder="Ask about the EU AI Act. Follow-ups like "
                                              "“what about the second one?” work too.")
                message = gr.Textbox(show_label=False, lines=1, max_lines=4, submit_btn="Ask",
                                     placeholder="Ask a question or a follow-up…")
                agent_mode = gr.Checkbox(
                    label="🤖 Agent mode (model chooses search, calculator, quote_exact, get_definition, date_diff on its own)",
                    value=False,
                )

                gr.Markdown("**Related**", elem_id="suggest-label")
                with gr.Row(elem_id="suggest-row"):
                    suggestion_buttons = [gr.Button("", visible=False, size="sm") for _ in range(N_SUGGESTIONS)]

                with gr.Accordion("Example questions", open=False):
                    gr.Examples(EXAMPLES, inputs=message)
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Cost — live"):
                        cost_table = gr.Markdown(handlers.cost_table(None))
                    with gr.Tab("Sources"):
                        sources = gr.Markdown(EMPTY_SOURCES, height=440)
                        with gr.Accordion("Other retrieved candidates", open=False):
                            others = gr.Markdown()

        with gr.Row():
            clear = gr.Button("Clear conversation", size="sm")
            chat_txt = gr.Button("Export .txt", size="sm")
            chat_pdf = gr.Button("Export .pdf", size="sm")
            answer_txt = gr.Button("Export selected answer .txt", size="sm")
            answer_pdf = gr.Button("Export selected answer .pdf", size="sm")
            send_btn = gr.Button("Send selected answer to Teams", size="sm")
        export_file = gr.File(label="Download", interactive=False)
        cost = gr.Markdown(rag.tracker.summary(), visible=False)  # kept for the clear() wiring below

        # human-approval gate for send_message: preview first, nothing sent until Confirm
        send_preview = gr.Textbox(label="Preview — edit if needed, nothing is sent yet", visible=False, lines=3)
        with gr.Row():
            send_confirm = gr.Button("Confirm & Send", variant="primary", visible=False, size="sm")
            send_cancel = gr.Button("Cancel", visible=False, size="sm")
        send_status = gr.Markdown()

        gr.Markdown("---")
        with gr.Accordion("Ablation table — RAG day 2 (vector-only → +re-ranking → +BM25)", open=True):
            gr.Markdown(load_ablation_table())

        ask_outputs = [message, chat, turns, selected, sources, others, cost, cost_table, *suggestion_buttons]
        message.submit(handlers.ask, [message, chat, turns, agent_mode], ask_outputs, api_name="ask")
        chat.select(handlers.select, [turns], [selected, sources, others])
        chat.like(handlers.feedback, [turns], None)
        clear_outputs = [chat, turns, selected, sources, others, export_file, cost, cost_table, *suggestion_buttons]
        clear.click(handlers.clear, None, clear_outputs)
        # the chatbot's own 🗑 icon must forget the conversation too, or follow-ups use stale history
        chat.clear(handlers.clear, None, clear_outputs)
        chat_txt.click(lambda t: handlers.export_conversation(t, "txt"), [turns], export_file)
        chat_pdf.click(lambda t: handlers.export_conversation(t, "pdf"), [turns], export_file)
        answer_txt.click(lambda t, s: handlers.export_answer(t, s, "txt"), [turns, selected], export_file)
        answer_pdf.click(lambda t, s: handlers.export_answer(t, s, "pdf"), [turns, selected], export_file)
        for btn in suggestion_buttons:
            btn.click(handlers.pick_suggestion, [btn], [message]).then(
                handlers.ask, [message, chat, turns], ask_outputs)

        send_btn.click(handlers.prepare_send, [turns, selected],
                       [send_preview, send_confirm, send_cancel, send_status])
        send_confirm.click(handlers.confirm_send, [send_preview],
                           [send_status, send_preview, send_confirm, send_cancel])
        send_cancel.click(lambda: ("", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),
                          None, [send_status, send_preview, send_confirm, send_cancel])

    # a second page listing every course task, ticked automatically where the repo shows it
    with demo.route("Project checklist", "/checklist"):
        build_checklist_page()

    # the whole system as a clickable flowchart
    with demo.route("How it works", "/flow"):
        build_flow_page()

    return demo
