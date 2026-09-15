"""
Golden Set evaluation harness for Baseline 1 and Baseline 2.

IMPORTANT:
  - Evaluates ONLY on data/processed/golden_set_final.csv.
  - Ground-truth label column: annotator_label.
  - proposed_intent is NEVER used as a target here.
  - Neither model is retrained.
  - The Golden Set is never used for training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.config import (
    BASELINE1_MODEL_PATH,
    BASELINE2_MODEL_PATH,
    GOLDEN_SET_FINAL_PATH,
    GOLDEN_SET_PREDICTIONS_PATH,
    INTENT_TAXONOMY,
)


# Required columns in the Golden Set file.
_GOLDEN_REQUIRED_COLS = {
    "example_id",
    "customer_tweet_id",
    "customer_text",
    "annotator_label",
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_golden_set(path: Path | None = None) -> pd.DataFrame:
    """
    Load golden_set_final.csv and validate it for evaluation use.

    Returns a DataFrame with at least:
      example_id, customer_tweet_id, customer_text, annotator_label

    Raises:
      FileNotFoundError  if the file is missing.
      ValueError         if required columns are absent, annotator_label is
                         empty, or any label is outside the taxonomy.
    """
    csv_path = Path(path) if path is not None else GOLDEN_SET_FINAL_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Golden Set not found: {csv_path}. "
            "Expected: data/processed/golden_set_final.csv"
        )

    df = pd.read_csv(
        csv_path,
        dtype={"customer_tweet_id": str, "agent_tweet_id": str},
    )

    missing = _GOLDEN_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Golden Set CSV missing columns: {missing}")

    # Check for NaN labels BEFORE astype(str), because astype converts NaN -> 'nan'.
    null_labels = df["annotator_label"].isna()
    df["annotator_label"] = df["annotator_label"].fillna("").astype(str).str.strip()
    df["customer_text"] = df["customer_text"].fillna("").astype(str).str.strip()

    empty_labels = null_labels | df["annotator_label"].eq("")
    if empty_labels.any():
        raise ValueError(
            f"Golden Set has {empty_labels.sum()} row(s) with empty annotator_label. "
            "The reviewed golden_set_final.csv must be fully annotated."
        )

    empty_texts = df["customer_text"].eq("")
    if empty_texts.any():
        raise ValueError(
            f"Golden Set has {empty_texts.sum()} row(s) with empty customer_text."
        )

    oov = set(df["annotator_label"]) - set(INTENT_TAXONOMY)
    if oov:
        raise ValueError(
            f"Golden Set annotator_label contains out-of-taxonomy labels: {sorted(oov)}"
        )

    return df.reset_index(drop=True)


def load_baseline1_model(model_path: Path | None = None):
    """Load the saved Baseline 1 sklearn Pipeline (TF-IDF + LogReg)."""
    from src.classify import load_baseline1
    return load_baseline1(model_path)


def load_baseline2_model(model_path: Path | None = None):
    """Load the saved Baseline 2 sklearn Pipeline (TF-IDF + LinearSVC)."""
    from src.baseline2 import load_baseline2
    return load_baseline2(model_path)


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def batch_predict(model, texts: list[str]) -> list[str]:
    """
    Run model.predict() on a list of texts.

    Works with any sklearn Pipeline that exposes .predict().
    Returns a plain Python list of predicted intent strings.
    """
    if not texts:
        return []
    return [str(p) for p in model.predict(texts)]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute classification metrics for one set of predictions.

    Args:
        y_true:  Ground-truth labels (annotator_label values).
        y_pred:  Predicted labels from a baseline model.
        labels:  Ordered label list; defaults to INTENT_TAXONOMY.

    Returns a dict with:
        accuracy         float
        macro_f1         float
        per_intent       dict[intent -> {precision, recall, f1, support}]
        confusion_matrix np.ndarray  (rows=true, cols=predicted, ordered by `labels`)
        labels           list[str]  the ordered label list used
        classification_report_text  str  (sklearn human-readable report)
    """
    if labels is None:
        labels = list(INTENT_TAXONOMY)

    # Only include labels that appear in either split (avoids zero-div warnings
    # for taxonomy labels completely absent from this particular eval set).
    present = sorted(set(y_true) | set(y_pred))
    # Preserve taxonomy ordering for labels that are present.
    eval_labels = [lab for lab in labels if lab in present]
    # Append any predicted labels not in taxonomy (should not happen if model
    # was trained correctly, but defensive).
    for lab in present:
        if lab not in eval_labels:
            eval_labels.append(lab)

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=eval_labels, zero_division=0)
    )

    prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=eval_labels,
        zero_division=0,
    )

    per_intent: dict[str, dict[str, float | int]] = {}
    for label, p, r, f, s in zip(eval_labels, prec_arr, rec_arr, f1_arr, sup_arr):
        per_intent[label] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }

    cm = confusion_matrix(y_true, y_pred, labels=eval_labels)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=eval_labels,
        digits=3,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_intent": per_intent,
        "confusion_matrix": cm,
        "labels": eval_labels,
        "classification_report_text": report_text,
    }


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_baseline(
    model,
    golden_df: pd.DataFrame,
    model_name: str = "baseline",
) -> dict[str, Any]:
    """
    Evaluate one baseline model against the Golden Set.

    Args:
        model:      Fitted sklearn Pipeline.
        golden_df:  DataFrame from load_golden_set().
        model_name: Human-readable name for reporting.

    Returns a dict with all metrics plus 'predictions' (list of str).
    """
    texts = golden_df["customer_text"].tolist()
    y_true = golden_df["annotator_label"].tolist()

    predictions = batch_predict(model, texts)
    metrics = compute_metrics(y_true, predictions)

    return {
        "model_name": model_name,
        "n_examples": len(texts),
        "predictions": predictions,
        **metrics,
    }


def run_full_evaluation(
    golden_path: Path | None = None,
    baseline1_path: Path | None = None,
    baseline2_path: Path | None = None,
    predictions_out_path: Path | None = None,
) -> dict[str, Any]:
    """
    End-to-end evaluation of both baselines on the Golden Set.

    Loads Golden Set, loads both models, predicts, computes metrics,
    builds and saves the predictions CSV.

    Returns:
        dict with keys 'golden_df', 'baseline1', 'baseline2',
        'predictions_df', 'predictions_path'.
    """
    golden_df = load_golden_set(golden_path)
    model1 = load_baseline1_model(baseline1_path)
    model2 = load_baseline2_model(baseline2_path)

    result1 = evaluate_baseline(model1, golden_df, model_name="baseline1_tfidf_logreg")
    result2 = evaluate_baseline(model2, golden_df, model_name="baseline2_tfidf_linearsvc")

    # Build predictions DataFrame.
    pred_df = pd.DataFrame(
        {
            "example_id": golden_df["example_id"],
            "customer_tweet_id": golden_df["customer_tweet_id"],
            "customer_text": golden_df["customer_text"],
            "annotator_label": golden_df["annotator_label"],
            "baseline1_prediction": result1["predictions"],
            "baseline2_prediction": result2["predictions"],
        }
    )
    pred_df["baseline1_correct"] = (
        pred_df["baseline1_prediction"] == pred_df["annotator_label"]
    )
    pred_df["baseline2_correct"] = (
        pred_df["baseline2_prediction"] == pred_df["annotator_label"]
    )
    pred_df["both_correct"] = pred_df["baseline1_correct"] & pred_df["baseline2_correct"]
    pred_df["neither_correct"] = (
        ~pred_df["baseline1_correct"] & ~pred_df["baseline2_correct"]
    )

    out_path = Path(predictions_out_path) if predictions_out_path else GOLDEN_SET_PREDICTIONS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)

    return {
        "golden_df": golden_df,
        "baseline1": result1,
        "baseline2": result2,
        "predictions_df": pred_df,
        "predictions_path": str(out_path),
    }
