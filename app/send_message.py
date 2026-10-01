"""
send_message: posts a message to the team's Teams channel via a Power Automate
"Workflows" webhook (the spec's explicit fallback when full MCP/OAuth setup
takes too long: "a plain webhook or bot API call is an accepted fallback").

Gated behind human approval in the UI (app/ui.py): the button only shows a
preview first; nothing is sent until a second, explicit confirm click.

Keep the webhook URL in .env (TEAMS_WEBHOOK_URL), never hardcoded or
committed -- it's a bearer credential, anyone with it can post to your
channel.

The new Teams "Workflows" webhook (Store > Workflows > "Send webhook alerts
to a channel") replaced the legacy Incoming Webhook connector, and expects a
full Adaptive Card payload -- a plain {"text": ...} body fails with "The
specified Teams flowbot adaptive card request is missing or invalid" (seen
in the flow's real run history at
https://make.powerautomate.com, two failed runs, both from this cause).
"""

import os

import requests

MAX_CHARS = 2000
TIMEOUT_SECONDS = 10


def _adaptive_card_payload(text: str) -> dict:
    return {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "type": "AdaptiveCard",
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "version": "1.4",
                    "body": [{"type": "TextBlock", "text": text, "wrap": True}],
                },
            }
        ],
    }


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

    payload = _adaptive_card_payload(text)
    try:
        response = requests.post(webhook_url, json=payload, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as e:
        return f"Error sending to Teams: {e}"

    if response.status_code >= 300:
        return f"Error: Teams returned {response.status_code} — {response.text[:200]}"
    return "Sent to Teams."
