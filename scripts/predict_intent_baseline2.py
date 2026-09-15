"""
Run Baseline 2 intent prediction on one customer message.

Confidence_like is softmax(decision_function) — NOT a calibrated probability.

Usage:
  python scripts/predict_intent_baseline2.py --text "my battery drains overnight"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline2 import predict_intent_baseline2
from src.config import BASELINE2_MODEL_PATH


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict intent with Baseline 2 (LinearSVC).")
    p.add_argument("--text", required=True, help="Customer message text")
    p.add_argument("--model", type=Path, default=BASELINE2_MODEL_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_intent_baseline2(args.text, model_path=args.model)
    print(
        json.dumps(
            {
                "predicted_intent": result["predicted_intent"],
                "confidence_like": result["confidence_like"],
                "confidence_note": result["confidence_note"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
