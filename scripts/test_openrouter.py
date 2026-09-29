"""
Standalone experiment — NOT part of the graded app, not imported by app/*.py.

Checks whether OPENROUTER_API_KEY can reach gpt-4o-mini and text-embedding-3-small
through OpenRouter's OpenAI-compatible endpoint, purely to see if it works.

Run:
    python scripts/test_openrouter.py
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

print("Testing chat (openai/gpt-4o-mini via OpenRouter)...")
chat = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Say 'OpenRouter works' and nothing else."}],
    temperature=0,
    max_tokens=10,
)
print("  ->", chat.choices[0].message.content)

print("\nTesting embeddings (openai/text-embedding-3-small via OpenRouter)...")
emb = client.embeddings.create(
    model="openai/text-embedding-3-small",
    input="test sentence",
)
print(f"  -> got a {len(emb.data[0].embedding)}-dimension vector")
