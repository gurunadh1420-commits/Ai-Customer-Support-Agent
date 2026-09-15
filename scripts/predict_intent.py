"""
Run Baseline 1 intent prediction on one customer message.

Usage:
  python scripts/predict_intent.py --text "my battery drains overnight"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classify import predict_intent
from src.config import BASELINE1_MODEL_PATH


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict intent with Baseline 1.")
    p.add_argument("--text", required=True, help="Customer message text")
    p.add_argument("--model", type=Path, default=BASELINE1_MODEL_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_intent(args.text, model_path=args.model)
    print(
        json.dumps(
            {
                "predicted_intent": result["predicted_intent"],
                "confidence": result["confidence"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
