"""
Unit tests for automated LLM-as-a-Judge system (src/judge.py).

Run:
  python tests/test_judge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from src.judge import evaluate_with_judge, validate_and_parse_judge_response


def test_valid_judge_json() -> None:
    raw_json = """
    {
      "scores": {
        "relevance": 5,
        "groundedness": 4,
        "helpfulness": 4,
        "safety_appropriateness": 5
      },
      "serious_failure_detected": false,
      "serious_failure_type": null,
      "cited_evidence_ids": ["100234"],
      "reasoning": "Reply is well grounded in retrieved tweet 100234 and provides clear guidance."
    }
    """
    valid_ids = {"100234", "100235"}
    res = validate_and_parse_judge_response(raw_json, valid_ids, example_id="GS-0001")

    assert res["example_id"] == "GS-0001"
    assert res["relevance_score"] == 5
    assert res["groundedness_score"] == 4
    assert res["helpfulness_score"] == 4
    assert res["safety_score"] == 5
    assert res["average_score"] == 4.5
    assert res["pass_fail"] == "PASS"
    assert res["serious_failure_detected"] is False
    assert res["cited_evidence_ids"] == ["100234"]
    assert "100234" in res["judge_reasoning"]


def test_invalid_score_rejection() -> None:
    valid_ids = {"100234"}

    # Case A: score > 5
    raw_high = """
    {
      "scores": {"relevance": 6, "groundedness": 4, "helpfulness": 4, "safety_appropriateness": 5},
      "serious_failure_detected": false,
      "cited_evidence_ids": ["100234"]
    }
    """
    with pytest.raises(ValueError, match="Score for 'relevance' must be an integer between 1 and 5"):
        validate_and_parse_judge_response(raw_high, valid_ids)

    # Case B: score < 1
    raw_low = """
    {
      "scores": {"relevance": 0, "groundedness": 4, "helpfulness": 4, "safety_appropriateness": 5},
      "serious_failure_detected": false,
      "cited_evidence_ids": ["100234"]
    }
    """
    with pytest.raises(ValueError, match="Score for 'relevance' must be an integer between 1 and 5"):
        validate_and_parse_judge_response(raw_low, valid_ids)

    # Case C: non-numeric string score
    raw_str = """
    {
      "scores": {"relevance": "five", "groundedness": 4, "helpfulness": 4, "safety_appropriateness": 5},
      "serious_failure_detected": false,
      "cited_evidence_ids": ["100234"]
    }
    """
    with pytest.raises(ValueError, match="Score for 'relevance' must be an integer 1-5"):
        validate_and_parse_judge_response(raw_str, valid_ids)


def test_invalid_evidence_id_rejection() -> None:
    valid_ids = {"100234", "100235"}
    # Judge cites '999999' which was NOT in retrieved evidence
    raw_invalid_id = """
    {
      "scores": {"relevance": 4, "groundedness": 4, "helpfulness": 4, "safety_appropriateness": 5},
      "serious_failure_detected": false,
      "cited_evidence_ids": ["999999"],
      "reasoning": "Citing fabricated evidence ID"
    }
    """
    with pytest.raises(ValueError, match="Invalid cited_evidence_id '999999': ID was not present in retrieved evidence"):
        validate_and_parse_judge_response(raw_invalid_id, valid_ids)


def test_pass_calculation() -> None:
    valid_ids = {"1001"}
    raw_pass = """
    {
      "scores": {"relevance": 4, "groundedness": 4, "helpfulness": 4, "safety_appropriateness": 4},
      "serious_failure_detected": false,
      "cited_evidence_ids": ["1001"],
      "reasoning": "All scores >= 3 and average >= 3.5"
    }
    """
    res = validate_and_parse_judge_response(raw_pass, valid_ids)
    assert res["average_score"] == 4.0
    assert res["pass_fail"] == "PASS"


def test_fail_calculation_low_dimension() -> None:
    valid_ids = {"1001"}
    # One dimension score < 3 -> must FAIL
    raw_fail_dim = """
    {
      "scores": {"relevance": 4, "groundedness": 2, "helpfulness": 4, "safety_appropriateness": 4},
      "serious_failure_detected": false,
      "cited_evidence_ids": ["1001"],
      "reasoning": "Groundedness is 2"
    }
    """
    res = validate_and_parse_judge_response(raw_fail_dim, valid_ids)
    assert res["pass_fail"] == "FAIL"


def test_serious_failure_forcing_fail() -> None:
    valid_ids = {"1001"}
    # All scores 5, but serious_failure_detected is True -> MUST FAIL
    raw_sf = """
    {
      "scores": {"relevance": 5, "groundedness": 5, "helpfulness": 5, "safety_appropriateness": 5},
      "serious_failure_detected": true,
      "serious_failure_type": "false_guarantee",
      "cited_evidence_ids": ["1001"],
      "reasoning": "Promised unauthorized free replacement."
    }
    """
    res = validate_and_parse_judge_response(raw_sf, valid_ids)
    assert res["average_score"] == 5.0
    assert res["serious_failure_detected"] is True
    assert res["pass_fail"] == "FAIL"


def test_evaluate_with_judge_mock() -> None:
    # Test full evaluate_with_judge wrapper with custom llm_caller
    def mock_llm_caller(prompt: str, sys_prompt: str) -> str:
        return """
        {
          "scores": {"relevance": 5, "groundedness": 5, "helpfulness": 4, "safety_appropriateness": 5},
          "serious_failure_detected": false,
          "cited_evidence_ids": ["101"],
          "reasoning": "Mock evaluation passed."
        }
        """

    evidence = [{"agent_tweet_id": "101", "similarity_score": 0.85}]
    res = evaluate_with_judge(
        example_id="GS-0001",
        customer_text="@AppleSupport battery drain",
        predicted_intent="battery_drain",
        retrieved_evidence=evidence,
        reply_text="Grounded reply text",
        reply_status="instructional",
        escalation_decision="auto_handle",
        llm_caller=mock_llm_caller,
    )
    assert res["example_id"] == "GS-0001"
    assert res["pass_fail"] == "PASS"
    assert res["cited_evidence_ids"] == ["101"]


def main() -> None:
    test_valid_judge_json()
    print("PASS test_valid_judge_json")
    test_invalid_score_rejection()
    print("PASS test_invalid_score_rejection")
    test_invalid_evidence_id_rejection()
    print("PASS test_invalid_evidence_id_rejection")
    test_pass_calculation()
    print("PASS test_pass_calculation")
    test_fail_calculation_low_dimension()
    print("PASS test_fail_calculation_low_dimension")
    test_serious_failure_forcing_fail()
    print("PASS test_serious_failure_forcing_fail")
    test_evaluate_with_judge_mock()
    print("PASS test_evaluate_with_judge_mock")
    print("All LLM-as-a-Judge unit tests passed.")


if __name__ == "__main__":
    main()
