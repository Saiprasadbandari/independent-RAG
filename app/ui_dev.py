from app.rag import Answer
from app.ui_sidebar import build_ui


class _Tracker:
    def summary(self):
        return "fake RAG, no real cost"


class FakeRAG:
    tracker = _Tracker()

    def ask(self, question, history=None, on_stage=None):
        chunk = {"id": 0, "source": "EU AI Regulations.pdf", "page": 12,
                 "text": "Fake passage text used to test the sources panel.", "score": 0.9}
        return Answer(question=question, text=f"Fake answer to: {question} [1]",
                      standalone_question=question, sources=[{**chunk, "ref": 1}],
                      retrieved=[chunk], context_size=1)


if __name__ == "__main__":
    build_ui(FakeRAG()).launch()
