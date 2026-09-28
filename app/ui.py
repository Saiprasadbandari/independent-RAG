import gradio as gr

from app import rag
from app.utils import format_citation, format_retrieved

EXAMPLES = [
    "What AI practices are prohibited under the AI Act?",
    "What counts as a high-risk AI system?",
    "What are the transparency obligations for providers of general-purpose AI models?",
    "What is the best pizza topping?",
]


def format_answer_sources(answer) -> str:
    if not answer.sources:
        return "_No sources cited._"
    return "\n".join(f"- **[{s['ref']}]** {format_citation(s)}" for s in answer.sources)


def build_ui(index, chunks: list[dict]) -> gr.Blocks:
    def respond(question: str):
        if not question or not question.strip():
            return "Please enter a question.", "", "", rag.tracker.summary()
        try:
            answer = rag.ask(question, index, chunks)
        except RuntimeError as e:
            return f"**Error:** {e}", "", "", rag.tracker.summary()
        retrieved = format_retrieved(answer.retrieved) if answer.retrieved else "_Nothing retrieved._"
        cost = f"This query: ${answer.cost:.5f} · {rag.tracker.summary()}"
        return answer.text, format_answer_sources(answer), retrieved, cost

    with gr.Blocks(title="EU AI Act RAG") as demo:
        gr.Markdown("# EU AI Act Assistant\nAsk questions about Regulation (EU) 2024/1689. "
                    "Answers are grounded in the source document with page citations.")

        with gr.Row():
            question = gr.Textbox(label="Question", placeholder="Ask about the EU AI Act...",
                                  lines=2, scale=5)
            submit = gr.Button("Ask", variant="primary", scale=1)

        answer = gr.Markdown(label="Answer")
        sources = gr.Markdown(label="Sources")
        with gr.Accordion("Retrieved passages", open=False):
            retrieved = gr.Markdown()
        cost = gr.Markdown(rag.tracker.summary())

        gr.Examples(EXAMPLES, inputs=question)

        outputs = [answer, sources, retrieved, cost]
        submit.click(respond, inputs=question, outputs=outputs)
        question.submit(respond, inputs=question, outputs=outputs)

    return demo
