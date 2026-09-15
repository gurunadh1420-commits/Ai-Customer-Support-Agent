"""
Build weakly labelled training CSV for Baseline 1 (no model training here).

Usage:
  python scripts/prepare_weak_training.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    GOLDEN_SET_CANDIDATES_PATH,
    PROCESSED_SUPPORT_PAIRS_PATH,
    WEAK_TRAINING_PATH,
)
from src.weak_labels import prepare_weak_training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create conservative weak-intent training data.")
    p.add_argument("--pairs", type=Path, default=PROCESSED_SUPPORT_PAIRS_PATH)
    p.add_argument("--golden", type=Path, default=GOLDEN_SET_CANDIDATES_PATH)
    p.add_argument("--out", type=Path, default=WEAK_TRAINING_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("Weak labels are heuristic only — NOT ground truth.")
    print("Golden Set examples are excluded from this training file.")
    print(f"Pairs: {args.pairs}")
    print(f"Golden exclude list: {args.golden}")

    _, summary = prepare_weak_training(
        pairs_path=args.pairs,
        golden_path=args.golden,
        out_path=args.out,
    )

    print(f"\nSaved: {summary['out_path']}")
    print(f"Total historical pairs: {summary['total_historical_pairs']:,}")
    print(f"Excluded golden customer tweets: {summary['excluded_golden_customer_tweets']:,}")
    print(f"Pairs after golden exclusion: {summary['pairs_after_golden_exclusion']:,}")
    print(f"Labelled examples: {summary['labelled_examples']:,}")
    print(f"Unlabelled examples: {summary['unlabelled_examples']:,}")
    print("\nCount per weak_intent:")
    for intent, count in summary["count_per_weak_intent"].items():
        print(f"  {count:6d}  {intent}")


if __name__ == "__main__":
    main()
