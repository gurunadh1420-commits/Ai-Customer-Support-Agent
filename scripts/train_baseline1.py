"""
Train Baseline 1 (TF-IDF + Logistic Regression) on weak labels only.

Usage:
  python scripts/train_baseline1.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classify import train_baseline1
from src.config import BASELINE1_META_PATH, BASELINE1_MODEL_PATH, WEAK_TRAINING_PATH


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Baseline 1 intent classifier.")
    p.add_argument("--weak-csv", type=Path, default=WEAK_TRAINING_PATH)
    p.add_argument("--model-out", type=Path, default=BASELINE1_MODEL_PATH)
    p.add_argument("--meta-out", type=Path, default=BASELINE1_META_PATH)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("Baseline 1: TF-IDF (1-2 grams) + LogisticRegression(class_weight='balanced')")
    print("Training data: weak labels only (NOT Golden Set).")
    print(f"Weak CSV: {args.weak_csv}")

    result = train_baseline1(
        weak_csv=args.weak_csv,
        model_path=args.model_out,
        meta_path=args.meta_out,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )

    print(f"\nTrain size: {result['train_size']:,}")
    print(f"Validation size: {result['validation_size']:,}")
    print(f"Validation accuracy: {result['validation_accuracy']:.4f}")
    print(f"Validation macro-F1: {result['validation_macro_f1']:.4f}")
    print("\nValidation classification report (weak holdout, not Golden Set):")
    print(result["classification_report"])
    print(f"Saved model: {result['model_path']}")
    print(f"Saved meta:  {result['meta_path']}")


if __name__ == "__main__":
    main()
