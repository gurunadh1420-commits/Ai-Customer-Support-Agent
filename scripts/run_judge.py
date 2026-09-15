"""
Run automated LLM-as-a-Judge evaluation on sample agent outputs.

Usage:
  python scripts/run_judge.py --sample-id GS-0001
  python scripts/run_judge.py --limit 3

Note: Do NOT run over all 200 examples until explicitly instructed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.config import PROCESSED_DIR
from src.judge import evaluate_with_judge


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge on sample agent outputs.")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=PROCESSED_DIR / "golden_set_agent_results.csv",
        help="Path to golden_set_agent_results.csv",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Specific example_id to evaluate (e.g. GS-0001).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of sample rows to evaluate (default 3, max 10 for sample mode).",
    )
    args = parser.parse_args()

    csv_path = Path(args.results_csv)
    if not csv_path.exists():
        print(f"Error: Results CSV not found: {csv_path}")
        print("Run: python scripts/evaluate_agent_golden_set.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if args.sample_id:
        sub = df[df["example_id"] == args.sample_id]
        if sub.empty:
            print(f"Error: example_id '{args.sample_id}' not found in CSV.")
            sys.exit(1)
        eval_rows = sub.head(1)
    else:
        limit = min(args.limit, 10)
        print(f"Running LLM Judge in sample mode (evaluating first {limit} rows)...")
        eval_rows = df.head(limit)

    for idx, row in eval_rows.iterrows():
        # Reconstruct evidence block from predictions/details if needed
        mock_evidence = [
            {
                "agent_tweet_id": "historical_1001",
                "similarity_score": float(row.get("top_retrieval_similarity", 0.50)),
                "customer_text": str(row.get("customer_text")),
                "agent_text": str(row.get("reply_text")),
            }
        ]

        try:
            result = evaluate_with_judge(
                example_id=str(row["example_id"]),
                customer_text=str(row["customer_text"]),
                predicted_intent=str(row["predicted_intent"]),
                retrieved_evidence=mock_evidence,
                reply_text=str(row["reply_text"]),
                reply_status=str(row["reply_status"]),
                escalation_decision=str(row["escalation_decision"]),
            )
            print("=" * 60)
            print(f"LLM JUDGE RESULT: {result['example_id']}")
            print("=" * 60)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Judge execution for {row['example_id']} failed as expected: {e}")


if __name__ == "__main__":
    main()
