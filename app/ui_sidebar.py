from dataclasses import asdict

import gradio as gr
import openai

from app.pdf_viewer import source_pages
from app.ui import EMPTY_SOURCES, EXAMPLES, format_chat_answer, format_sources_panel


def message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):        # newer Gradio can send a list of content blocks
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def to_pairs(history) -> list[tuple[str, str]]:
    """ChatInterface history -> [(question, answer), ...] as RAG.ask expects."""
    pairs, question = [], None
    for item in history:
        if isinstance(item, dict):                       # {"role": ..., "content": ...}
            text = message_text(item["content"])
            if item["role"] == "user":
                question = text
            elif question is not None:
                pairs.append((question, text))
                question = None
        else:                                            # [user, assistant]
            pairs.append((message_text(item[0]), message_text(item[1])))
    return pairs


def build_ui(rag) -> gr.Blocks:
    def respond(message, history):
        try:
            answer = rag.ask(message, to_pairs(history))
            turn = {"question": message.strip(), "answer": asdict(answer)}
        except (RuntimeError, openai.OpenAIError) as e:
            turn = {"question": message.strip(), "error": str(e)}
        sources_md, others_md = format_sources_panel([turn], 0)
        pages = source_pages(turn["answer"]["sources"]) if "answer" in turn else []
        return (format_chat_answer(turn), sources_md.replace("question 1", "the latest question"),
                others_md, pages, rag.tracker.summary())

    with gr.Blocks(title="EU AI Act RAG") as demo:
        # created here, placed in the right-hand column below
        sources = gr.Markdown(EMPTY_SOURCES, height=300, render=False)
        others = gr.Markdown(render=False)
        page_gallery = gr.Gallery(show_label=False, columns=1, height=600, object_fit="contain", render=False)
        cost = gr.Markdown(rag.tracker.summary(), render=False)

        gr.Markdown("# EU AI Act Assistant\nAsk questions about Regulation (EU) 2024/1689. "
                    "Past conversations are in the side panel.")
        with gr.Row():
            with gr.Column(scale=3):
                gr.ChatInterface(respond, additional_outputs=[sources, others, page_gallery, cost],
                                 save_history=True, examples=EXAMPLES, cache_examples=False)
            with gr.Column(scale=2):
                sources.render()
                with gr.Accordion("Original pages (cited text highlighted)", open=True):
                    page_gallery.render()
                with gr.Accordion("Other retrieved candidates (not sent to the model)", open=False):
                    others.render()
                cost.render()
    return demo
