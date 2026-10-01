"""
Layer 3: Tools — the three read-only actions the agent can choose to call.

Each tool is a plain Python function plus a JSON schema describing it (the
schema is the only thing the model reads to decide when to use it). The agent
harness (app/agent.py) imports build_tools(rag) to get both, wired together.

    search_corpus(query)      -> semantic search over the corpus (wraps the
                                  RAG object's own retrieve+rerank pipeline,
                                  so it gets the same caching and quality)
    calculator(expression)    -> arithmetic via simpleeval (never eval())
    quote_exact(source, page) -> exact, full text of one page, for verifying
                                  or quoting a passage a search pointed to
    get_definition(term)      -> the official Article 3 definition of a term
    date_diff(start, end)     -> time between two dates (days, and months + days)
    send_message(text)        -> drafts a Teams message; never sends it. The
                                  agent cannot call app.send_message.send_to_teams
                                  directly -- only a human, via the existing
                                  "Send selected answer to Teams" + "Confirm & Send"
                                  buttons in app/ui.py, can actually transmit it.

All six always return a string, never raise: an empty or failed result comes
back as readable text ("No matching passages found.", "Error: ...") so the
agent can read it and decide what to do next, instead of crashing the loop.
"""

import re
from bisect import bisect_right
from calendar import monthrange
from datetime import date, datetime

from simpleeval import simple_eval

from app.embedding import load_corpus
from app.rag import relevant_chunks
from app.send_message import MAX_CHARS

_CORPUS_BY_PAGE: dict[tuple[str, int], str] | None = None
_DEFINITIONS: dict[str, dict] | None = None

# Article 3 lists its definitions as "(n) 'term' means ...". A few put a qualifier
# between the term and "means" ("'subject', for the purpose of real-world testing, means").
DEFINITION_RE = re.compile(r"\((\d{1,2})\)\s*‘([^’]{2,80})’[^‘;]{0,60}?\bmeans\b")
DATE_FORMATS = ("%d %B %Y", "%Y-%m-%d", "%d/%m/%Y", "%B %d, %Y")


def _corpus_index() -> dict[tuple[str, int], str]:
    """Lazy-loaded (source, page) -> full page text lookup for quote_exact."""
    global _CORPUS_BY_PAGE
    if _CORPUS_BY_PAGE is None:
        _CORPUS_BY_PAGE = {(e["source"], e["page"]): e["text"] for e in load_corpus()}
    return _CORPUS_BY_PAGE


def make_search_corpus(rag):
    """Bind search_corpus to a specific RAG instance (its index, reranker, caches)."""

    def search_corpus(query: str) -> str:
        candidates = rag.retrieve(query)
        plausible = relevant_chunks(candidates)
        if not plausible:
            return "No matching passages found."

        if rag.reranker:
            ranked = rag.rerank(query, plausible)
            results = [c for c in ranked if c["relevance"] >= rag.reranker.min_relevance][:rag.top_k]
        else:
            results = plausible[:rag.top_k]

        if not results:
            return "No matching passages found."
        return "\n\n".join(f"[Source: {r['source']} p.{r['page']}] {r['text']}" for r in results)

    return search_corpus


def calculator(expression: str) -> str:
    try:
        return str(simple_eval(expression))
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


def quote_exact(source: str, page: int) -> str:
    text = _corpus_index().get((source, page))
    if text is None:
        return f"No text found for {source} page {page}."
    return f"[Source: {source} p.{page}]\n{text}"


def _definitions() -> dict[str, dict]:
    """Lazy-loaded term -> {number, text, source, page}, parsed from Article 3."""
    global _DEFINITIONS
    if _DEFINITIONS is None:
        _DEFINITIONS = {}
        for source in {s for s, _ in _corpus_index()}:
            pages = sorted((p, t) for (s, p), t in _corpus_index().items() if s == source)
            starts, text = [], ""
            for _, page_text in pages:
                starts.append(len(text))
                text += page_text + "\n"
            begin = text.find("Article 3\nDefinitions")
            if begin == -1:
                continue
            end = text.find("\nArticle 4\n", begin)
            matches = list(DEFINITION_RE.finditer(text, begin, end if end != -1 else len(text)))
            for i, m in enumerate(matches):
                stop = matches[i + 1].start() if i + 1 < len(matches) else end
                body = " ".join(text[m.start():stop].split()).rstrip(";")
                page = pages[bisect_right(starts, m.start()) - 1][0]
                _DEFINITIONS[m.group(2).lower()] = {
                    "number": int(m.group(1)), "text": body, "source": source, "page": page,
                }
    return _DEFINITIONS


def get_definition(term: str) -> str:
    key = " ".join(term.lower().strip(" '\"‘’“”").split())
    definitions = _definitions()
    found = definitions.get(key) or definitions.get(key.rstrip("s"))
    if found:
        return f"[Source: {found['source']} p.{found['page']}] Article 3, definition {found['text']}"
    close = [t for t in definitions if key and (key in t or t in key)]
    if close:
        return f"No definition found for '{term}' in Article 3. Defined terms that contain it: " + ", ".join(sorted(close)[:8])
    return f"No definition found for '{term}' in Article 3."


def _parse_date(value: str) -> date:
    value = " ".join(value.strip().split())
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"unrecognised date '{value}', use e.g. '2 August 2026' or '2026-08-02'")


def _add_months(d: date, months: int) -> date:
    year, month = divmod(d.month - 1 + months, 12)
    year += d.year
    return date(year, month + 1, min(d.day, monthrange(year, month + 1)[1]))


def date_diff(start: str, end: str) -> str:
    try:
        a, b = _parse_date(start), _parse_date(end)
    except ValueError as e:
        return f"Error: {e}"
    sign = "" if b >= a else "-"
    lo, hi = min(a, b), max(a, b)
    months = (hi.year - lo.year) * 12 + hi.month - lo.month - (hi.day < lo.day)
    days = (hi - _add_months(lo, months)).days
    total = (hi - lo).days
    return (f"From {a.day} {a:%B %Y} to {b.day} {b:%B %Y}: {sign}{total} day{'s' * (total != 1)} "
            f"({sign}{months} month{'s' * (months != 1)} and {days} day{'s' * (days != 1)}, "
            f"about {sign}{total / 365.25:.2f} years).")


def send_message(text: str) -> str:
    """Drafts a Teams message. Deliberately does not call send_to_teams(): the only
    way this text reaches Teams is a human selecting this answer in the chat and
    clicking 'Send selected answer to Teams' then 'Confirm & Send' in app/ui.py."""
    if not text or not text.strip():
        return "Error: cannot draft an empty message."
    if len(text) > MAX_CHARS:
        return f"Error: message is {len(text)} characters, limit is {MAX_CHARS}."
    return (f"Drafted, not sent. A human must select this answer and click "
            f"'Send selected answer to Teams', then 'Confirm & Send', to actually send it:\n\n{text}")


SEARCH_CORPUS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": (
            "Search the EU AI Act corpus for relevant passages. Use this whenever "
            "the user asks about the regulation, its articles, obligations, penalties, "
            "or any fact that would be found in the document. Returns the top matching "
            "passages with their source and page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for, phrased as a search query.",
                }
            },
            "required": ["query"],
        },
    },
}

CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Evaluate a math expression and return the result. Use this for any "
            "arithmetic (percentages, sums, conversions) instead of computing it yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression using digits and operators only, e.g. '35000000 * 0.07'.",
                }
            },
            "required": ["expression"],
        },
    },
}

QUOTE_EXACT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "quote_exact",
        "description": (
            "Return the exact, complete text of one page from a named document. Use this "
            "to verify or quote a passage precisely after search_corpus points you to it, "
            "since search results may be chunked mid-sentence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The exact document filename, e.g. 'EU AI Regulations.pdf'.",
                },
                "page": {
                    "type": "integer",
                    "description": "The page number to quote.",
                },
            },
            "required": ["source", "page"],
        },
    },
}


GET_DEFINITION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_definition",
        "description": (
            "Return the official definition of a term from Article 3 (Definitions) of the "
            "EU AI Act, word for word, with its page. Use this when the user asks what a "
            "term means or who counts as something ('provider', 'deployer', 'AI system', "
            "'general-purpose AI model'), instead of search_corpus, which can return a "
            "similar term's definition or a recital that only mentions the word."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "The term to define, without quotes, e.g. 'deployer'.",
                }
            },
            "required": ["term"],
        },
    },
}

DATE_DIFF_SCHEMA = {
    "type": "function",
    "function": {
        "name": "date_diff",
        "description": (
            "Compute the time between two dates, in days and in months plus days. Use this "
            "for any question about how long between two dates of the regulation (entry into "
            "force, application dates, deadlines), after finding the dates with search_corpus. "
            "Do not use calculator for dates, and do not count months yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start": {
                    "type": "string",
                    "description": "The earlier date, e.g. '1 August 2024' or '2024-08-01'.",
                },
                "end": {
                    "type": "string",
                    "description": "The later date, e.g. '2 August 2026' or '2026-08-02'.",
                },
            },
            "required": ["start", "end"],
        },
    },
}


SEND_MESSAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": (
            "Draft a message to post to the team's Teams channel. This only drafts the "
            "text -- it is never sent automatically or by this tool. A human must review "
            "the drafted message in the chat and explicitly click to send it before "
            "anything reaches Teams. Use this when asked to notify the team or post an update."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": f"The message to draft for Teams, under {MAX_CHARS} characters.",
                }
            },
            "required": ["text"],
        },
    },
}


def build_tools(rag) -> tuple[list[dict], dict]:
    """Returns (schemas, functions) for the agent's tools, ready for the agent loop."""
    functions = {
        "search_corpus": make_search_corpus(rag),
        "calculator": calculator,
        "quote_exact": quote_exact,
        "get_definition": get_definition,
        "date_diff": date_diff,
        "send_message": send_message,
    }
    schemas = [SEARCH_CORPUS_SCHEMA, CALCULATOR_SCHEMA, QUOTE_EXACT_SCHEMA, GET_DEFINITION_SCHEMA,
               DATE_DIFF_SCHEMA, SEND_MESSAGE_SCHEMA]
    return schemas, functions
