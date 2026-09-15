"""
Tests for conservative weak-intent rules (not ground truth / not training).

Run:
  python tests/test_weak_labels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.weak_labels import (
    build_weak_training_frame,
    choose_weak_intent,
    find_candidate_intents,
)


def test_strong_single_labels() -> None:
    assert choose_weak_intent("@AppleSupport my battery drains so fast since yesterday") == "battery_drain"
    assert choose_weak_intent("@AppleSupport my iPhone won't charge with this cable") == "charging_issue"
    assert (
        choose_weak_intent("@AppleSupport App Store won't load and I can't download any apps")
        == "app_store_issue"
    )
    assert choose_weak_intent("@AppleSupport my Apple ID is locked and I need a verification code") == (
        "account_access"
    )
    assert choose_weak_intent("@AppleSupport storage almost full after deleting photos") == "storage_issue"
    assert choose_weak_intent("how do I delete apps on iOS?") == "howto_or_feature"
    assert choose_weak_intent("@AppleSupport thank you") == "other_or_unclear"


def test_battery_preferred_over_update_context() -> None:
    text = "@AppleSupport since the iOS update my battery life is terrible"
    assert choose_weak_intent(text) == "battery_drain"


def test_ambiguous_left_unlabeled() -> None:
    # Mentions both drain-ish and charger-ish without a clear single primary.
    text = "@AppleSupport battery drain and my charger won't charge either help"
    assert "battery_drain" in find_candidate_intents(text)
    assert "charging_issue" in find_candidate_intents(text)
    assert choose_weak_intent(text) is None


def test_payment_vs_subscription_resolution() -> None:
    assert (
        choose_weak_intent("@AppleSupport please refund this unauthorized charge")
        == "payment_or_refund"
    )
    assert (
        choose_weak_intent("@AppleSupport I can't find the manage subscription button")
        == "subscription_management"
    )


def test_excludes_golden_ids_from_training_rows() -> None:
    pairs = pd.DataFrame(
        [
            {
                "customer_tweet_id": "1",
                "agent_tweet_id": "a1",
                "customer_text": "@AppleSupport my battery drains overnight",
            },
            {
                "customer_tweet_id": "2",
                "agent_tweet_id": "a2",
                "customer_text": "@AppleSupport my battery drains overnight",
            },
            {
                "customer_tweet_id": "3",
                "agent_tweet_id": "a3",
                "customer_text": "hello world no intent here",
            },
        ]
    )
    out, summary = build_weak_training_frame(pairs, exclude_customer_tweet_ids={"1"})
    assert "1" not in set(out["customer_tweet_id"])
    assert "2" in set(out["customer_tweet_id"])
    assert summary["labelled_examples"] == 1
    assert summary["unlabelled_examples"] == 1
    assert list(out.columns) == [
        "customer_tweet_id",
        "agent_tweet_id",
        "customer_text",
        "weak_intent",
    ]


def test_oou_rule_c_and_d() -> None:
    # Rule C (vague fix requests)
    assert choose_weak_intent("@AppleSupport fix your music") == "other_or_unclear"
    assert choose_weak_intent("@AppleSupport please fix it") == "other_or_unclear"
    assert choose_weak_intent("@AppleSupport fix this!") == "other_or_unclear"

    # Rule D (generic failure / escalation)
    assert choose_weak_intent("@AppleSupport I tried everything") == "other_or_unclear"
    assert choose_weak_intent("@AppleSupport tried restarting, still not working") == "other_or_unclear"
    assert choose_weak_intent("@AppleSupport Yes still the same") == "other_or_unclear"

    # Specific keyword guard checks (must NOT classify as OOU)
    assert choose_weak_intent("@AppleSupport wifi not working") == "network_connectivity"
    assert choose_weak_intent("@AppleSupport wifi still not working") != "other_or_unclear"
    assert choose_weak_intent("@AppleSupport battery still not working") != "other_or_unclear"


def main() -> None:
    test_strong_single_labels()
    print("PASS test_strong_single_labels")
    test_battery_preferred_over_update_context()
    print("PASS test_battery_preferred_over_update_context")
    test_ambiguous_left_unlabeled()
    print("PASS test_ambiguous_left_unlabeled")
    test_payment_vs_subscription_resolution()
    print("PASS test_payment_vs_subscription_resolution")
    test_excludes_golden_ids_from_training_rows()
    print("PASS test_excludes_golden_ids_from_training_rows")
    test_oou_rule_c_and_d()
    print("PASS test_oou_rule_c_and_d")
    print("All weak-label tests passed.")


if __name__ == "__main__":
    main()
