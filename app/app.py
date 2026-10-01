import gradio as gr

# Try the real RAG (written by teammates in app/rag.py). Fall back to a fake one.
try:
    from rag import chat_answer, cost
    DEMO = False
except Exception as e:
    print("Real RAG not available, using fake backend:", e)
    DEMO = True
    cost = {"usd": 0.0}

REFUSAL = "I don't have information about this in the documents."
SOURCES_MARK = "\n\n---\n**Sources**"


def fake_chat_answer(question, history):
    """Stand-in for the real RAG. Returns (text, hits, standalone_question)."""
    if "weather" in question.lower():
        return REFUSAL, [], question
    hits = [
        {"source": "AI_Act.pdf", "page": 12,
         "text": "High-risk AI systems are those that...", "score": 0.61},
        {"source": "AI_Act.pdf", "page": 45,
         "text": "Providers of high-risk systems shall...", "score": 0.55},
    ]
    return "This is a fake answer [1][2].", hits, question


if DEMO:
    chat_answer = fake_chat_answer


def to_messages(history):
    """Make the chat history a clean list of {"role", "content"} on any Gradio version."""
    out = []
    for item in history or []:
        if isinstance(item, dict):
            content = item.get("content", "")
            if isinstance(content, list):  # newer Gradio can send a list of parts
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            out.append({"role": item["role"], "content": content})
        else:  # old format: (user, bot) pairs
            user, bot = item
            out.append({"role": "user", "content": user})
            out.append({"role": "assistant", "content": bot})
    # Remove the Sources block so the question rewriter sees a clean conversation
    for m in out:
        m["content"] = m["content"].split(SOURCES_MARK)[0]
    return out


def format_sources(hits):
    """Turn the list of retrieved chunks into a readable Sources section."""
    if not hits:
        return ""
    lines = []
    for i, h in enumerate(hits, 1):
        preview = h["text"][:200].replace("\n", " ")
        lines.append(
            f"- **[{i}]** `{h['source']}`, p.{h['page']} "
            f"(score {h['score']:.2f})\n  > {preview}..."
        )
    return SOURCES_MARK + "\n" + "\n".join(lines)


def chat(message, history):
    """Called by Gradio on every user message. Returns the text to display."""
    try:
        text, hits, standalone = chat_answer(message, to_messages(history))
    except Exception as e:
        print("ERROR:", e)
        return "Sorry, something went wrong. Please try again."
    print(f"searched as: {standalone!r} | spend: ${cost['usd']:.4f}")
    return text + format_sources(hits)


demo = gr.ChatInterface(
    chat,
    title="AI Act Assistant",
    description="Ask about Regulation (EU) 2024/1689. Answers cite document and page."
                + ("\n\n**DEMO MODE: fake backend**" if DEMO else ""),
    examples=[
        "What is a high-risk AI system?",
        "Which AI practices are prohibited?",
        "What are the penalties for non-compliance?",
    ],
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)