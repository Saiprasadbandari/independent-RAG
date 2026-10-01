"""
send_message: posts a message to the team's Teams channel via an Incoming
Webhook (the spec's explicit fallback when full MCP/OAuth setup takes too
long: "a plain webhook or bot API call is an accepted fallback").

Gated behind human approval in the UI (app/ui.py): the button only shows a
preview first; nothing is sent until a second, explicit confirm click.

Keep the webhook URL in .env (TEAMS_WEBHOOK_URL), never hardcoded or
committed -- it's a bearer credential, anyone with it can post to your
channel.
"""

import os

import requests

MAX_CHARS = 2000
TIMEOUT_SECONDS = 10


def send_to_teams(text: str, webhook_url: str | None = None) -> str:
    """Posts `text` to the configured Teams channel. Always returns a
    string (never raises) so the UI can show success or failure either way."""
    webhook_url = webhook_url or os.getenv("TEAMS_WEBHOOK_URL")
    if not webhook_url:
        return "Error: TEAMS_WEBHOOK_URL is not set in .env."
    if not text or not text.strip():
        return "Error: refusing to send an empty message."
    if len(text) > MAX_CHARS:
        return f"Error: message is {len(text)} characters, limit is {MAX_CHARS}."

    payload = {"text": text}
    try:
        response = requests.post(webhook_url, json=payload, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as e:
        return f"Error sending to Teams: {e}"

    if response.status_code >= 300:
        return f"Error: Teams returned {response.status_code} — {response.text[:200]}"
    return "Sent to Teams."
