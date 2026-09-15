"""
Evaluate Baseline 1 and Baseline 2 on the 200-example Golden Set.

Ground truth: annotator_label column in data/processed/golden_set_final.csv
Models are loaded from disk (pre-trained). No retraining occurs.

Usage:
  python scripts/evaluate_baselines.py
  python scripts/evaluate_baselines.py --golden data/processed/golden_set_final.csv
  python scripts/evaluate_baselines.py --out data/processed/golden_set_predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.config import (
    BASELINE1_MODEL_PATH,
    BASELINE2_MODEL_PATH,
    INTENT_TAXONOMY,
)
from src.evaluate import (
    GOLDEN_SET_FINAL_PATH,
    GOLDEN_SET_PREDICTIONS_PATH,
    run_full_evaluation,
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 30) -> str:
    """Simple ASCII progress bar for a 0-1 value."""
    filled = round(value * width)
    return "[" + "#" * filled + "." * (width - filled) + f"] {value:.3f}"


def _print_header(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_section(title: str) -> None:
    print(f"\n" + "-" * 60)
    print(f"  {title}")
    print("-" * 60)


def _print_summary_row(name: str, accuracy: float, macro_f1: float) -> None:
    print(f"  {name:<35} acc={accuracy:.4f}  macro-F1={macro_f1:.4f}")


def _print_per_intent(per_intent: dict, labels: list[str]) -> None:
    header = f"  {'Intent':<28} {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Sup':>5}"
    print(header)
    print("  " + "-" * 56)
    for label in labels:
        if label not in per_intent:
            continue
        m = per_intent[label]
        print(
            f"  {label:<28} {m['precision']:>6.3f}  {m['recall']:>6.3f}"
            f"  {m['f1']:>6.3f}  {m['support']:>5d}"
        )
    # Totals / macro
    supports = [per_intent[l]["support"] for l in labels if l in per_intent]
    total_sup = sum(supports)
    macro_p = np.mean([per_intent[l]["precision"] for l in labels if l in per_intent])
    macro_r = np.mean([per_intent[l]["recall"] for l in labels if l in per_intent])
    macro_f = np.mean([per_intent[l]["f1"] for l in labels if l in per_intent])
    print("  " + "-" * 56)
    print(
        f"  {'macro avg':<28} {macro_p:>6.3f}  {macro_r:>6.3f}"
        f"  {macro_f:>6.3f}  {total_sup:>5d}"
    )


def _print_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    """Print a compact confusion matrix with label abbreviations."""
    abbr = {lab: lab[:8] for lab in labels}
    col_labels = [abbr[l] for l in labels]
    # Header row
    col_width = 10
    header = " " * 28 + "".join(f"{c:>{col_width}}" for c in col_labels)
    print(header)
    print(" " * 28 + "-" * (col_width * len(labels)))
    for i, row_label in enumerate(labels):
        if i >= cm.shape[0]:
            break
        row = " ".join(f"{v:>{col_width}}" for v in cm[i])
        row_str = " ".join(f"{v:>{col_width}}" for v in cm[i])
        print(f"  {row_label:<26}|" + row_str)


def _print_agreement(pred_df) -> None:
    n = len(pred_df)
    both_correct = pred_df["both_correct"].sum()
    neither_correct = pred_df["neither_correct"].sum()
    b1_only = (pred_df["baseline1_correct"] & ~pred_df["baseline2_correct"]).sum()
    b2_only = (pred_df["baseline2_correct"] & ~pred_df["baseline1_correct"]).sum()
    print(f"  Both correct:               {both_correct:4d} / {n}  ({both_correct/n:.1%})")
    print(f"  Neither correct:            {neither_correct:4d} / {n}  ({neither_correct/n:.1%})")
    print(f"  Baseline 1 only correct:    {b1_only:4d} / {n}  ({b1_only/n:.1%})")
    print(f"  Baseline 2 only correct:    {b2_only:4d} / {n}  ({b2_only/n:.1%})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Baseline 1 and Baseline 2 on the Golden Set."
    )
    p.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_SET_FINAL_PATH,
        help="Path to golden_set_final.csv (default: data/processed/golden_set_final.csv)",
    )
    p.add_argument(
        "--baseline1-model",
        type=Path,
        default=BASELINE1_MODEL_PATH,
        help="Path to Baseline 1 .joblib (default: models/baseline1/tfidf_logreg.joblib)",
    )
    p.add_argument(
        "--baseline2-model",
        type=Path,
        default=BASELINE2_MODEL_PATH,
        help="Path to Baseline 2 .joblib (default: models/baseline2/tfidf_linearsvc.joblib)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=GOLDEN_SET_PREDICTIONS_PATH,
        help="Output path for per-example predictions CSV",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("\nLoading Golden Set and models ...")
    results = run_full_evaluation(
        golden_path=args.golden,
        baseline1_path=args.baseline1_model,
        baseline2_path=args.baseline2_model,
        predictions_out_path=args.out,
    )

    r1 = results["baseline1"]
    r2 = results["baseline2"]
    pred_df = results["predictions_df"]
    labels = r1["labels"]  # same taxonomy for both

    # -----------------------------------------------------------------------
    # SECTION 1: Top-line comparison
    # -----------------------------------------------------------------------
    _print_header("GOLDEN SET EVALUATION — BASELINE COMPARISON")
    print(f"\n  Evaluation set : {args.golden}")
    print(f"  Examples       : {r1['n_examples']}")
    print(f"  Ground truth   : annotator_label (human-reviewed)")
    print(f"  Taxonomy size  : {len(INTENT_TAXONOMY)} intents\n")

    _print_section("Top-Line Metrics")
    _print_summary_row("Baseline 1 (TF-IDF + LogReg)  ", r1["accuracy"], r1["macro_f1"])
    _print_summary_row("Baseline 2 (TF-IDF + LinearSVC)", r2["accuracy"], r2["macro_f1"])

    print()
    print(f"  Baseline 1  accuracy  {_bar(r1['accuracy'])}")
    print(f"  Baseline 1  macro-F1  {_bar(r1['macro_f1'])}")
    print(f"  Baseline 2  accuracy  {_bar(r2['accuracy'])}")
    print(f"  Baseline 2  macro-F1  {_bar(r2['macro_f1'])}")

    # Winner
    b1_wins_acc = r1["accuracy"] > r2["accuracy"]
    b1_wins_f1 = r1["macro_f1"] > r2["macro_f1"]
    if b1_wins_acc and b1_wins_f1:
        winner = "Baseline 1 (TF-IDF + LogReg) is better on both accuracy and macro-F1."
    elif not b1_wins_acc and not b1_wins_f1:
        winner = "Baseline 2 (TF-IDF + LinearSVC) is better on both accuracy and macro-F1."
    elif b1_wins_acc:
        winner = "Baseline 1 is better on accuracy; Baseline 2 is better on macro-F1."
    else:
        winner = "Baseline 2 is better on accuracy; Baseline 1 is better on macro-F1."
    print(f"\n  --> {winner}")

    # -----------------------------------------------------------------------
    # SECTION 2: Per-intent metrics — Baseline 1
    # -----------------------------------------------------------------------
    _print_section("Per-Intent Metrics — Baseline 1 (TF-IDF + LogReg)")
    _print_per_intent(r1["per_intent"], labels)

    # -----------------------------------------------------------------------
    # SECTION 3: Per-intent metrics — Baseline 2
    # -----------------------------------------------------------------------
    _print_section("Per-Intent Metrics — Baseline 2 (TF-IDF + LinearSVC)")
    _print_per_intent(r2["per_intent"], labels)

    # -----------------------------------------------------------------------
    # SECTION 4: Confusion matrices
    # -----------------------------------------------------------------------
    _print_section("Confusion Matrix — Baseline 1 (rows=true, cols=predicted)")
    _print_confusion_matrix(r1["confusion_matrix"], labels)

    _print_section("Confusion Matrix — Baseline 2 (rows=true, cols=predicted)")
    _print_confusion_matrix(r2["confusion_matrix"], labels)

    # -----------------------------------------------------------------------
    # SECTION 5: Agreement analysis
    # -----------------------------------------------------------------------
    _print_section("Prediction Agreement Between Baselines")
    _print_agreement(pred_df)

    # -----------------------------------------------------------------------
    # SECTION 6: Full sklearn classification reports
    # -----------------------------------------------------------------------
    _print_section("Full Classification Report — Baseline 1")
    print(r1["classification_report_text"])

    _print_section("Full Classification Report — Baseline 2")
    print(r2["classification_report_text"])

    # -----------------------------------------------------------------------
    # Output file
    # -----------------------------------------------------------------------
    _print_section("Predictions CSV Saved")
    print(f"  Path: {results['predictions_path']}")
    print(f"  Rows: {len(pred_df)}")
    print(
        f"  Columns: {', '.join(pred_df.columns.tolist())}"
    )

    print("\n" + "=" * 70)
    print("  Evaluation complete. No models were retrained.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
