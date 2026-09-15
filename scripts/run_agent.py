"""
Run the end-to-end SupportAgent pipeline on a customer query.

Usage:
  python scripts/run_agent.py --text "@AppleSupport my battery drains so fast overnight"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import SupportAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end AppleSupport Agent.")
    parser.add_argument(
        "--text",
        type=str,
        default="@AppleSupport my battery drains so fast overnight please help",
        help="Customer message text to process.",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Number of retrieval examples (default 5)."
    )
    args = parser.parse_args()

    agent = SupportAgent()
    res = agent.process_message(args.text, k=args.k)

    # Print summary
    print("=" * 60)
    print("SUPPORT AGENT PIPELINE RESULT")
    print("=" * 60)
    print(f"Customer Text:          {res['customer_text']}")
    print(f"Predicted Intent:       {res['predicted_intent']}")
    print(f"Classifier Confidence:  {res['classifier_confidence']:.4f}")
    print(f"Retrieved Evidence:     {len(res['retrieved_evidence'])} historical pairs")
    print(f"Reply Status:           {res['reply_status']}")
    print(f"Escalation Decision:    {res['escalation_decision'].upper()}")
    print(f"Reason Code:            {res['escalation_reason_code']}")
    print(f"Reason:                 {res['escalation_reason']}")
    print("-" * 60)
    print("Draft Reply Text:")
    print(res["reply_text"])
    print("=" * 60)


if __name__ == "__main__":
    main()
