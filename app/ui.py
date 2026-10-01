import json
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import gradio as gr
import openai

from app.checklist import build_checklist_page
from app.export import export
from app.rag import RAG, cited_refs
from app.utils import DATA_DIR, format_citation, format_retrieved

FEEDBACK_PATH = DATA_DIR / "feedback.jsonl"

EXAMPLES = [
    "What AI practices are prohibited under the AI Act?",
    "What counts as a high-risk AI system?",
    "What are the transparency obligations for providers of general-purpose AI models?",
    "What is the best pizza topping?",
]

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


def format_chat_answer(turn: dict) -> str:
    if "error" in turn:
        return f"⚠️ **Error:** {turn['error']}"
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
        answer = turn["answer"]
        if answer["cached"]:
            this = "$0 (answer from cache)"
        else:
            tokens = f", {answer['input_tokens']} in / {answer['output_tokens']} out tokens" if answer["input_tokens"] else ""
            this = f"${answer['cost']:.5f}{tokens}"
        return f"This answer: {this} · {self.rag.tracker.summary()}"

    def ask(self, message: str, chat: list, turns: list):
        """Generator: the reply bubble shows each stage, then the answer as it streams in."""
        if not message or not message.strip():
            raise gr.Error("Please enter a question.")
        question = message.strip()
        history = [(t["answer"]["standalone_question"] or t["question"], t["answer"]["text"])
                   for t in turns if "answer" in t]
        chat = chat + [{"role": "user", "content": question}, {"role": "assistant", "content": "⏳ Starting…"}]
        waiting = (gr.skip(),) * 5
        yield ("", list(chat), *waiting)

        try:
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
        yield (gr.skip(), chat, turns, index, *format_sources_panel(turns, index), self.cost_line(turn))

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
        return [], [], None, EMPTY_SOURCES, "", None, self.rag.tracker.summary()


def build_ui(rag: RAG) -> gr.Blocks:
    handlers = ChatHandlers(rag, Path(tempfile.mkdtemp(prefix="rag-export-")))

    with gr.Blocks(title="EU AI Act RAG") as demo:
        turns = gr.State([])
        selected = gr.State(None)

        gr.Markdown("# EU AI Act Assistant\nAsk questions about Regulation (EU) 2024/1689. "
                    "Follow-up questions use the conversation so far. Answers cite the pages they come from.")

        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Conversation", height=560, layout="bubble",
                                  feedback_options=("Like", "Dislike"), buttons=["copy"],
                                  placeholder="Ask about the EU AI Act. Follow-ups like "
                                              "“what about the second one?” work too.")
                message = gr.Textbox(show_label=False, lines=1, max_lines=4, submit_btn="Ask",
                                     placeholder="Ask a question or a follow-up…")
                gr.Examples(EXAMPLES, inputs=message)
            with gr.Column(scale=2):
                sources = gr.Markdown(EMPTY_SOURCES, height=560)
                with gr.Accordion("Other retrieved candidates (not sent to the model)", open=False):
                    others = gr.Markdown()

        with gr.Row():
            clear = gr.Button("Clear conversation")
        with gr.Accordion("Export", open=False):
            gr.Markdown("Click an answer in the conversation to pick it; otherwise the latest answer is exported.")
            with gr.Row():
                chat_txt = gr.Button("Conversation (.txt)")
                chat_pdf = gr.Button("Conversation (.pdf)")
                answer_txt = gr.Button("Selected answer (.txt)")
                answer_pdf = gr.Button("Selected answer (.pdf)")
            export_file = gr.File(label="Download", interactive=False)
        cost = gr.Markdown(rag.tracker.summary())

        message.submit(handlers.ask, [message, chat, turns],
                       [message, chat, turns, selected, sources, others, cost], api_name="ask")
        chat.select(handlers.select, [turns], [selected, sources, others])
        chat.like(handlers.feedback, [turns], None)
        clear_outputs = [chat, turns, selected, sources, others, export_file, cost]
        clear.click(handlers.clear, None, clear_outputs)
        # the chatbot's own 🗑 icon must forget the conversation too, or follow-ups use stale history
        chat.clear(handlers.clear, None, clear_outputs)
        chat_txt.click(lambda t: handlers.export_conversation(t, "txt"), [turns], export_file)
        chat_pdf.click(lambda t: handlers.export_conversation(t, "pdf"), [turns], export_file)
        answer_txt.click(lambda t, s: handlers.export_answer(t, s, "txt"), [turns, selected], export_file)
        answer_pdf.click(lambda t, s: handlers.export_answer(t, s, "pdf"), [turns, selected], export_file)

    # a second page listing every course task, ticked automatically where the repo shows it
    with demo.route("Project checklist", "/checklist"):
        build_checklist_page()

    return demo
