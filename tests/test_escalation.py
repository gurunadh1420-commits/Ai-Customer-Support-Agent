"""
Unit tests for escalation decision layer (src/escalation.py).

Run:
  python tests/test_escalation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.escalation import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
    REASON_AUTO_HANDLE,
    REASON_DM_HANDOFF,
    REASON_INSUFFICIENT_REPLY,
    REASON_INSUFFICIENT_RETRIEVAL,
    REASON_LOW_CONFIDENCE,
    REASON_UNSUPPORTED_INTENT,
    decide_escalation,
)


def test_rule_1_other_or_unclear() -> None:
    res = decide_escalation(
        predicted_intent="other_or_unclear",
        classifier_confidence=0.90,
        retrieval_results=[{"similarity_score": 0.80}],
        reply_result={"draft_style": "instructional"},
    )
    assert res["decision"] == "escalate"
    assert res["reason_code"] == REASON_UNSUPPORTED_INTENT
    assert "other_or_unclear" in res["reason"]


def test_rule_2_low_classifier_confidence() -> None:
    res = decide_escalation(
        predicted_intent="battery_drain",
        classifier_confidence=0.55,  # < 0.65
        retrieval_results=[{"similarity_score": 0.80}],
        reply_result={"draft_style": "instructional"},
    )
    assert res["decision"] == "escalate"
    assert res["reason_code"] == REASON_LOW_CONFIDENCE
    assert "0.55" in res["reason"]


def test_rule_3_insufficient_retrieval_evidence_no_results() -> None:
    res = decide_escalation(
        predicted_intent="battery_drain",
        classifier_confidence=0.85,
        retrieval_results=[],  # No retrieval results
        reply_result={"draft_style": "instructional"},
    )
    assert res["decision"] == "escalate"
    assert res["reason_code"] == REASON_INSUFFICIENT_RETRIEVAL


def test_rule_3_insufficient_retrieval_evidence_low_similarity() -> None:
    res = decide_escalation(
        predicted_intent="battery_drain",
        classifier_confidence=0.85,
        retrieval_results=[{"similarity_score": 0.25}],  # < 0.30
        reply_result={"draft_style": "instructional"},
    )
    assert res["decision"] == "escalate"
    assert res["reason_code"] == REASON_INSUFFICIENT_RETRIEVAL


def test_rule_4_insufficient_reply_evidence() -> None:
    res = decide_escalation(
        predicted_intent="battery_drain",
        classifier_confidence=0.85,
        retrieval_results=[{"similarity_score": 0.50}],
        reply_result={"draft_style": "insufficient_evidence"},
    )
    assert res["decision"] == "escalate"
    assert res["reason_code"] == REASON_INSUFFICIENT_REPLY


def test_rule_5_dm_redirect_handoff() -> None:
    res = decide_escalation(
        predicted_intent="battery_drain",
        classifier_confidence=0.85,
        retrieval_results=[{"similarity_score": 0.50}],
        reply_result={"draft_style": "dm_redirect"},
    )
    assert res["decision"] == "escalate"
    assert res["reason_code"] == REASON_DM_HANDOFF


def test_rule_6_auto_handle_case_1_battery() -> None:
    res = decide_escalation(
        predicted_intent="battery_drain",
        classifier_confidence=0.88,
        retrieval_results=[{"similarity_score": 0.72}],
        reply_result={"draft_style": "instructional"},
    )
    assert res["decision"] == "auto_handle"
    assert res["reason_code"] == REASON_AUTO_HANDLE
    assert res["key_evidence"]["predicted_intent"] == "battery_drain"
    assert res["key_evidence"]["classifier_confidence"] == 0.88
    assert res["key_evidence"]["top_similarity"] == 0.72


def test_rule_6_auto_handle_case_2_charging() -> None:
    res = decide_escalation(
        predicted_intent="charging_issue",
        classifier_confidence=0.72,
        retrieval_results=[{"similarity_score": 0.45}],
        reply_result={"draft_style": "instructional"},
    )
    assert res["decision"] == "auto_handle"
    assert res["reason_code"] == REASON_AUTO_HANDLE
    assert res["key_evidence"]["predicted_intent"] == "charging_issue"
    assert res["key_evidence"]["classifier_confidence"] == 0.72
    assert res["key_evidence"]["top_similarity"] == 0.45


def main() -> None:
    test_rule_1_other_or_unclear()
    print("PASS test_rule_1_other_or_unclear")
    test_rule_2_low_classifier_confidence()
    print("PASS test_rule_2_low_classifier_confidence")
    test_rule_3_insufficient_retrieval_evidence_no_results()
    print("PASS test_rule_3_insufficient_retrieval_evidence_no_results")
    test_rule_3_insufficient_retrieval_evidence_low_similarity()
    print("PASS test_rule_3_insufficient_retrieval_evidence_low_similarity")
    test_rule_4_insufficient_reply_evidence()
    print("PASS test_rule_4_insufficient_reply_evidence")
    test_rule_5_dm_redirect_handoff()
    print("PASS test_rule_5_dm_redirect_handoff")
    test_rule_6_auto_handle_case_1_battery()
    print("PASS test_rule_6_auto_handle_case_1_battery")
    test_rule_6_auto_handle_case_2_charging()
    print("PASS test_rule_6_auto_handle_case_2_charging")
    print("All escalation decision tests passed.")


if __name__ == "__main__":
    main()
