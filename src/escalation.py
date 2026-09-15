"""
Escalation decision layer for automated support pairs handling.

Implements a conservative deterministic policy to decide whether a customer query
can be safely auto-handled or must be escalated to a human agent.

Decision outcomes:
  - "auto_handle"
  - "escalate"
"""

from __future__ import annotations

from typing import Any

# Default configurable thresholds
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.65
DEFAULT_SIMILARITY_THRESHOLD: float = 0.30

# Reason codes
REASON_UNSUPPORTED_INTENT = "unsupported_or_unclear_intent"
REASON_LOW_CONFIDENCE = "low_classifier_confidence"
REASON_INSUFFICIENT_RETRIEVAL = "insufficient_retrieval_evidence"
REASON_INSUFFICIENT_REPLY = "insufficient_reply_evidence"
REASON_DM_HANDOFF = "historical_resolution_requires_handoff"
REASON_AUTO_HANDLE = "high_confidence_supported_case"


def _extract_top_similarity(
    retrieval_results: list[dict[str, Any]] | float | int | None,
) -> float | None:
    """Extract top similarity score from retrieval results list or float."""
    if retrieval_results is None:
        return None
    if isinstance(retrieval_results, (int, float)):
        return float(retrieval_results)
    if isinstance(retrieval_results, list):
        if not retrieval_results:
            return None
        scores = [
            float(r["similarity_score"])
            for r in retrieval_results
            if isinstance(r, dict) and "similarity_score" in r
        ]
        return max(scores) if scores else None
    return None


def _extract_reply_status(reply_result: dict[str, Any] | str | None) -> str:
    """Extract status / draft_style from reply dict or string."""
    if reply_result is None:
        return "insufficient_evidence"
    if isinstance(reply_result, str):
        return reply_result.strip()
    if isinstance(reply_result, dict):
        return str(
            reply_result.get("draft_style")
            or reply_result.get("reply_status")
            or reply_result.get("status")
            or ""
        ).strip()
    return ""


def decide_escalation(
    predicted_intent: str,
    classifier_confidence: float,
    retrieval_results: list[dict[str, Any]] | float | int | None = None,
    reply_result: dict[str, Any] | str | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """
    Evaluate escalation rules in order of priority and return decision payload.

    Args:
        predicted_intent: Predicted intent label string.
        classifier_confidence: Classification probability score (0.0 to 1.0).
        retrieval_results: List of retrieved example dicts, or top similarity float.
        reply_result: Reply draft dict (from draft_reply) or status string.
        confidence_threshold: Minimum classifier confidence for auto-handling (default 0.65).
        similarity_threshold: Minimum retrieval similarity for auto-handling (default 0.30).

    Returns:
        dict containing:
            decision: "auto_handle" | "escalate"
            reason_code: str
            reason: str
            key_evidence: dict
    """
    intent = str(predicted_intent or "other_or_unclear").strip()
    conf = float(classifier_confidence if classifier_confidence is not None else 0.0)
    top_sim = _extract_top_similarity(retrieval_results)
    rep_status = _extract_reply_status(reply_result)

    key_evidence = {
        "predicted_intent": intent,
        "classifier_confidence": conf,
        "top_similarity": top_sim,
        "reply_status": rep_status,
        "confidence_threshold": confidence_threshold,
        "similarity_threshold": similarity_threshold,
    }

    # Rule 1: predicted_intent == "other_or_unclear" -> escalate
    if intent == "other_or_unclear":
        return {
            "decision": "escalate",
            "reason_code": REASON_UNSUPPORTED_INTENT,
            "reason": "Predicted intent 'other_or_unclear' is unsupported for automated handling.",
            "key_evidence": key_evidence,
        }

    # Rule 2: classifier_confidence < confidence_threshold -> escalate
    if conf < confidence_threshold:
        return {
            "decision": "escalate",
            "reason_code": REASON_LOW_CONFIDENCE,
            "reason": (
                f"Classifier confidence ({conf:.4f}) is below required threshold "
                f"({confidence_threshold:.4f})."
            ),
            "key_evidence": key_evidence,
        }

    # Rule 3: No retrieval evidence OR top similarity < similarity_threshold -> escalate
    if top_sim is None or top_sim < similarity_threshold:
        sim_val_str = f"{top_sim:.4f}" if top_sim is not None else "None"
        return {
            "decision": "escalate",
            "reason_code": REASON_INSUFFICIENT_RETRIEVAL,
            "reason": (
                f"Top retrieval similarity ({sim_val_str}) is below required threshold "
                f"({similarity_threshold:.4f}) or no evidence returned."
            ),
            "key_evidence": key_evidence,
        }

    # Rule 4: reply status == "insufficient_evidence" -> escalate
    if rep_status == "insufficient_evidence":
        return {
            "decision": "escalate",
            "reason_code": REASON_INSUFFICIENT_REPLY,
            "reason": "Reply drafting layer indicated insufficient historical evidence.",
            "key_evidence": key_evidence,
        }

    # Rule 5: reply status == "dm_redirect" -> escalate
    if rep_status == "dm_redirect":
        return {
            "decision": "escalate",
            "reason_code": REASON_DM_HANDOFF,
            "reason": "Historical resolution pattern requires direct message or private agent handoff.",
            "key_evidence": key_evidence,
        }

    # Rule 6: Otherwise -> auto_handle
    return {
        "decision": "auto_handle",
        "reason_code": REASON_AUTO_HANDLE,
        "reason": (
            "High confidence classification, strong retrieval similarity, and instructional reply "
            "support automated response."
        ),
        "key_evidence": key_evidence,
    }
