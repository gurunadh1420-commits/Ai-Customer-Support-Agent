"""
Tests for grounded template-based reply drafting (no LLM).

Run:
  python tests/test_reply.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reply import draft_reply


def test_insufficient_evidence() -> None:
    out = draft_reply("help", "other_or_unclear", [])
    assert out["draft_style"] == "insufficient_evidence"
    assert out["evidence"] == []
    assert "grounded" in out["groundedness_note"].lower()


def test_dm_redirect_style_when_history_is_mostly_dms() -> None:
    retrieved = [
        {
            "customer_tweet_id": "1",
            "agent_tweet_id": "2",
            "customer_text": "battery drains",
            "agent_text": (
                "We're here to help. Please DM us your iPhone model and iOS version. "
                "https://t.co/abc"
            ),
            "similarity_score": 0.9,
        },
        {
            "customer_tweet_id": "3",
            "agent_tweet_id": "4",
            "customer_text": "battery life bad",
            "agent_text": "Join us in a DM and we'll continue there. https://t.co/xyz",
            "similarity_score": 0.8,
        },
    ]
    out = draft_reply(
        "my battery drains overnight",
        "battery_drain",
        retrieved,
    )
    assert out["draft_style"] == "dm_redirect"
    assert "DM" in out["reply_text"]
    assert out["reply_text"] != retrieved[0]["agent_text"]
    assert out["reply_text"] != retrieved[1]["agent_text"]
    ids = {(e["customer_tweet_id"], e["agent_tweet_id"]) for e in out["evidence"]}
    assert ("1", "2") in ids
    assert ("3", "4") in ids


def test_instructional_style_uses_historical_steps_not_verbatim() -> None:
    full_agent = (
        "We're happy to help with your Wi-Fi. Go to Settings > Wi-Fi, forget the network, "
        "then restart your iPhone and reconnect. Let us know how it goes."
    )
    retrieved = [
        {
            "customer_tweet_id": "10",
            "agent_tweet_id": "11",
            "customer_text": "wifi keeps dropping",
            "agent_text": full_agent,
            "similarity_score": 0.95,
        },
        {
            "customer_tweet_id": "12",
            "agent_tweet_id": "13",
            "customer_text": "cannot connect wifi",
            "agent_text": (
                "Try these steps: open Settings and check Wi-Fi. "
                "Which iOS version are you using?"
            ),
            "similarity_score": 0.7,
        },
    ]
    out = draft_reply("wifi drops every hour", "network_connectivity", retrieved)
    assert out["draft_style"] == "instructional"
    assert "Settings" in out["reply_text"] or "settings" in out["reply_text"].lower()
    assert out["reply_text"].strip() != full_agent.strip()
    assert any(e["agent_tweet_id"] == "11" for e in out["evidence"])
    # Should not invent unrelated hardware repair advice
    assert "genius bar" not in out["reply_text"].lower()
    assert "logic board" not in out["reply_text"].lower()


def main() -> None:
    test_insufficient_evidence()
    print("PASS test_insufficient_evidence")
    test_dm_redirect_style_when_history_is_mostly_dms()
    print("PASS test_dm_redirect_style_when_history_is_mostly_dms")
    test_instructional_style_uses_historical_steps_not_verbatim()
    print("PASS test_instructional_style_uses_historical_steps_not_verbatim")
    print("All reply-drafting tests passed.")


if __name__ == "__main__":
    main()
