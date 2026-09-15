"""
Unit and integration tests for SupportAgent orchestration (src/agent.py).

Run:
  python tests/test_agent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import SupportAgent
from src.escalation import (
    REASON_AUTO_HANDLE,
    REASON_INSUFFICIENT_RETRIEVAL,
    REASON_LOW_CONFIDENCE,
    REASON_UNSUPPORTED_INTENT,
)


def test_normal_supported_case() -> None:
    agent = SupportAgent()
    # High confidence supported intent with strong retrieval match
    msg = "@AppleSupport my battery life is terrible after the update and drains so fast"
    res = agent.process_message(msg, k=5)

    assert res["customer_text"] == msg
    assert res["predicted_intent"] in ["battery_drain", "software_update_issue"]
    assert res["classifier_confidence"] > 0.0
    assert len(res["retrieved_evidence"]) == 5
    assert isinstance(res["reply_text"], str)
    assert res["reply_status"] in ["instructional", "dm_redirect"]
    assert res["escalation_decision"] in ["auto_handle", "escalate"]
    assert "escalation_reason_code" in res
    assert "escalation_reason" in res


def test_unclear_case_escalates() -> None:
    agent = SupportAgent()
    # "other_or_unclear" message
    msg = "@AppleSupport thank you ok"
    res = agent.process_message(msg, k=5)

    assert res["predicted_intent"] == "other_or_unclear"
    assert res["escalation_decision"] == "escalate"
    assert res["escalation_reason_code"] == REASON_UNSUPPORTED_INTENT


def test_low_confidence_case_escalates() -> None:
    agent = SupportAgent()
    msg = "@AppleSupport my battery life is terrible"
    # Set confidence_threshold above max classifier confidence (1.0) to test low_classifier_confidence escalation
    res = agent.process_message(msg, confidence_threshold=1.0001)

    assert res["escalation_decision"] == "escalate"
    assert res["escalation_reason_code"] == REASON_LOW_CONFIDENCE


def test_insufficient_retrieval_evidence_escalates() -> None:
    agent = SupportAgent()
    msg = "@AppleSupport my battery life is terrible"
    # Set similarity_threshold artificially high (e.g. 0.99) to force insufficient_retrieval_evidence escalation
    res = agent.process_message(msg, similarity_threshold=0.99)

    assert res["escalation_decision"] == "escalate"
    assert res["escalation_reason_code"] == REASON_INSUFFICIENT_RETRIEVAL


def main() -> None:
    test_normal_supported_case()
    print("PASS test_normal_supported_case")
    test_unclear_case_escalates()
    print("PASS test_unclear_case_escalates")
    test_low_confidence_case_escalates()
    print("PASS test_low_confidence_case_escalates")
    test_insufficient_retrieval_evidence_escalates()
    print("PASS test_insufficient_retrieval_evidence_escalates")
    print("All agent orchestration tests passed.")


if __name__ == "__main__":
    main()
