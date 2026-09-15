"""
Tests for Golden Set candidate sampling (not labeling / not training).

Run:
  python tests/test_golden_set_sampling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import INTENT_TAXONOMY
from src.golden_set import (
    build_golden_set_candidates,
    format_proposed_intent,
    matched_intents,
)


def _fake_pairs() -> pd.DataFrame:
    rows = []
    templates = {
        "battery_drain": "@AppleSupport my battery drains so fast since yesterday",
        "charging_issue": "@AppleSupport my iPhone won't charge with this cable",
        "software_update_issue": "@AppleSupport after the iOS 11 update nothing works right",
        "hardware_issue": "@AppleSupport my touch bar isn't working at all",
        "network_connectivity": "@AppleSupport wifi keeps dropping and bluetooth fails",
        "account_access": "@AppleSupport my apple id is locked and I need verification code",
        "app_store_issue": "@AppleSupport the app store won't load on my phone",
        "app_issue": "@AppleSupport safari keeps crashing when I open tabs",
        "payment_or_refund": "@AppleSupport I was charged twice please refund",
        "subscription_management": "@AppleSupport I can't find manage subscription button",
        "storage_issue": "@AppleSupport storage almost full after deleting photos",
        "howto_or_feature": "@AppleSupport how do I delete apps on iOS",
        "other_or_unclear": "@AppleSupport please help",
    }
    # Enough rows per template for stratified sampling.
    n = 0
    for intent, text in templates.items():
        for i in range(20):
            n += 1
            rows.append(
                {
                    "customer_tweet_id": str(1000 + n),
                    "agent_tweet_id": str(2000 + n),
                    "customer_text": f"{text} #{i}",
                    "agent_text": f"@user Thanks we can help with {intent}",
                    "customer_created_at": "2017-10-01",
                    "agent_created_at": "2017-10-01",
                }
            )
    # Multi-problem + boundary-ish
    for i in range(10):
        n += 1
        rows.append(
            {
                "customer_tweet_id": str(1000 + n),
                "agent_tweet_id": str(2000 + n),
                "customer_text": (
                    f"@AppleSupport after the iOS update my battery drains "
                    f"and safari keeps crashing #{i}"
                ),
                "agent_text": "@user Please DM us",
                "customer_created_at": "2017-10-01",
                "agent_created_at": "2017-10-01",
            }
        )
    return pd.DataFrame(rows)


def test_format_proposed_intent_marked_machine() -> None:
    s = format_proposed_intent("battery_drain", ["software_update_issue"])
    assert s.startswith("MACHINE_CANDIDATE:")
    assert "battery_drain" in s
    assert "software_update_issue" in s


def test_matched_intents_battery() -> None:
    hits = matched_intents("my battery drains overnight")
    assert "battery_drain" in hits


def test_build_candidates_schema_and_empty_annotator_fields() -> None:
    pairs = _fake_pairs()
    full, summary = build_golden_set_candidates(
        pairs=pairs, target_size=80, per_intent=5, seed=0
    )
    assert summary["n_examples"] >= 60
    assert set(INTENT_TAXONOMY).issuperset(set(summary["per_proposed_intent"]))
    for col in [
        "example_id",
        "customer_tweet_id",
        "agent_tweet_id",
        "customer_text",
        "agent_text",
        "proposed_intent",
        "annotator_label",
        "annotator_notes",
    ]:
        assert col in full.columns
    assert full["annotator_label"].fillna("").eq("").all()
    assert full["proposed_intent"].str.startswith("MACHINE_CANDIDATE:").all()
    assert full["example_id"].is_unique
    # Texts unchanged from source ids
    src = pairs.set_index("customer_tweet_id")["customer_text"]
    for _, row in full.iterrows():
        assert row["customer_text"] == src.loc[row["customer_tweet_id"]]


def main() -> None:
    test_format_proposed_intent_marked_machine()
    print("PASS test_format_proposed_intent_marked_machine")
    test_matched_intents_battery()
    print("PASS test_matched_intents_battery")
    test_build_candidates_schema_and_empty_annotator_fields()
    print("PASS test_build_candidates_schema_and_empty_annotator_fields")
    print("All golden-set sampling tests passed.")


if __name__ == "__main__":
    main()
