"""
End-to-end agent orchestration for AppleSupport handling.

Orchestrates the 4-stage pipeline:
  1. Intent Classification (predict intent & confidence)
  2. Evidence Retrieval (top-k historical support pairs)
  3. Grounded Reply Drafting
  4. Escalation Decision Layer
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.classify import load_baseline1, predict_intent
from src.config import RETRIEVAL_INDEX_PATH, ROOT_DIR
from src.escalation import decide_escalation
from src.reply import draft_reply
from src.retrieve import load_retrieval_index, retrieve_similar

BEST_CLASSIFIER_MODEL_PATH = (
    ROOT_DIR / "models" / "best_classifier" / "tfidf_logreg.joblib"
)


class SupportAgent:
    """Orchestrates classification, retrieval, reply drafting, and escalation."""

    def __init__(
        self,
        model_path: Path | None = None,
        retrieval_index_path: Path | None = None,
        model: Any | None = None,
        retrieval_index: dict[str, Any] | None = None,
    ) -> None:
        path = Path(model_path) if model_path is not None else BEST_CLASSIFIER_MODEL_PATH
        self.model = model if model is not None else load_baseline1(path)

        r_path = Path(retrieval_index_path) if retrieval_index_path is not None else RETRIEVAL_INDEX_PATH
        self.retrieval_index = (
            retrieval_index if retrieval_index is not None else load_retrieval_index(r_path)
        )

    def process_message(
        self,
        customer_text: str,
        k: int = 5,
        confidence_threshold: float = 0.65,
        similarity_threshold: float = 0.30,
    ) -> dict[str, Any]:
        """
        Process a single customer message through the 4-stage pipeline.

        Returns a structured dictionary containing all intermediate and final outputs.
        """
        text = str(customer_text or "").strip()
        if not text:
            raise ValueError("customer_text must be a non-empty customer message")

        # 1. Classify message
        clf_res = predict_intent(text, model=self.model)
        predicted_intent = str(clf_res["predicted_intent"])
        confidence = float(clf_res["confidence"])

        # 2. Retrieve top-k historical support examples
        retrieved_evidence = retrieve_similar(
            query=text, k=k, index=self.retrieval_index
        )

        # 3. Draft grounded reply
        reply_res = draft_reply(
            customer_text=text,
            predicted_intent=predicted_intent,
            retrieved_examples=retrieved_evidence,
        )
        reply_text = str(reply_res["reply_text"])
        reply_status = str(reply_res["draft_style"])

        # 4. Escalation Decision Layer
        esc_res = decide_escalation(
            predicted_intent=predicted_intent,
            classifier_confidence=confidence,
            retrieval_results=retrieved_evidence,
            reply_result=reply_res,
            confidence_threshold=confidence_threshold,
            similarity_threshold=similarity_threshold,
        )

        # 5. Return structured result
        return {
            "customer_text": text,
            "predicted_intent": predicted_intent,
            "classifier_confidence": confidence,
            "retrieved_evidence": retrieved_evidence,
            "reply_text": reply_text,
            "reply_status": reply_status,
            "escalation_decision": esc_res["decision"],
            "escalation_reason_code": esc_res["reason_code"],
            "escalation_reason": esc_res["reason"],
            "details": {
                "classification": clf_res,
                "reply": reply_res,
                "escalation": esc_res,
            },
        }


def run_agent_pipeline(
    customer_text: str,
    model_path: Path | None = None,
    retrieval_index_path: Path | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Convenience helper to run the pipeline on a single message."""
    agent = SupportAgent(
        model_path=model_path, retrieval_index_path=retrieval_index_path
    )
    return agent.process_message(customer_text=customer_text, k=k)
