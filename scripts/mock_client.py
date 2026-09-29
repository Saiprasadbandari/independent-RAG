"""
Local-only mock of the OpenAI client — NOT part of the graded app, not imported
by anything in app/. Zero API calls, zero cost, zero accuracy.

Purpose: verify the agent loop's MECHANICS (does it call tools, thread messages
correctly, log the trajectory, stop on a final answer, respect the step cap)
without needing a real API key. It does NOT tell you anything about whether the
real model would give good answers — the "reasoning" here is a hardcoded script,
not intelligence.

Run: python scripts/test_agent_local.py
"""

import json
import random
from types import SimpleNamespace


class MockToolCall:
    def __init__(self, call_id, name, arguments: dict):
        self.id = call_id
        self.type = "function"
        self.function = SimpleNamespace(name=name, arguments=json.dumps(arguments))


class MockMessage:
    def __init__(self, content=None, tool_calls=None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_unset=True):
        d = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in self.tool_calls
            ]
        return d


class MockOpenAI:
    """Scripted responses: 1st call -> search_corpus, 2nd -> calculator, 3rd -> final answer.
    Any call beyond that (e.g. a simple question with no multi-step script left) also answers."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat_create))
        self.embeddings = SimpleNamespace(create=self._embed_create)
        self._step = 0

    def _chat_create(self, model, messages, temperature=0, tools=None, tool_choice=None, max_tokens=None):
        self._step += 1
        usage = SimpleNamespace(prompt_tokens=200 * self._step, completion_tokens=20)

        if tools and self._step == 1:
            msg = MockMessage(tool_calls=[MockToolCall("call_1", "search_corpus",
                                                        {"query": "prohibited AI practices fines"})])
        elif tools and self._step == 2:
            msg = MockMessage(tool_calls=[MockToolCall("call_2", "calculator",
                                                        {"expression": "35000000 * 2"})])
        else:
            msg = MockMessage(content="[MOCK ANSWER] This is a fake final answer, "
                                      "not a real one — the mock client scripted this, "
                                      "gpt-4o-mini never ran.")

        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)

    def _embed_create(self, model, input):
        texts = input if isinstance(input, list) else [input]
        # deterministic fake vectors so re-runs are reproducible; NOT semantically meaningful,
        # so real-index search results will be arbitrary, not relevant
        data = []
        for t in texts:
            rng = random.Random(hash(t) % (2**32))
            data.append(SimpleNamespace(embedding=[rng.uniform(-1, 1) for _ in range(1536)]))
        usage = SimpleNamespace(prompt_tokens=sum(len(t) for t in texts) // 4)
        return SimpleNamespace(data=data, usage=usage)
