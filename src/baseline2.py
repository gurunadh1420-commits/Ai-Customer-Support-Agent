"""
Baseline 2: TF-IDF (unigrams+bigrams) + LinearSVC.

Kept separate from Baseline 1 (Logistic Regression).
Trained only on weakly labelled data — never on the Golden Set.

Confidence note:
  LinearSVC does NOT provide calibrated probabilities (no predict_proba).
  We expose a confidence-like score derived from decision_function via
  softmax over class decision scores. This is useful for ranking but is
  NOT a true probability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.classify import RANDOM_SEED, VALIDATION_FRACTION, load_weak_training
from src.config import (
    BASELINE2_META_PATH,
    BASELINE2_MODEL_PATH,
    INTENT_TAXONOMY,
)


def build_baseline2_pipeline(seed: int = RANDOM_SEED) -> Pipeline:
    """
    Baseline 2 pipeline.

    - Same TF-IDF setup as Baseline 1 (unigrams + bigrams, min_df=2)
    - LinearSVC with class_weight='balanced'
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    lowercase=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LinearSVC(
                    class_weight="balanced",
                    random_state=seed,
                    max_iter=5000,
                    dual="auto",
                ),
            ),
        ]
    )


def decision_scores_to_confidence_like(scores: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Convert LinearSVC decision_function scores to a confidence-like value.

    Uses softmax over class scores. This is NOT a calibrated probability.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    weights = exp / exp.sum()
    return weights, float(np.max(weights))


def train_baseline2(
    weak_csv: Path | None = None,
    model_path: Path | None = None,
    meta_path: Path | None = None,
    seed: int = RANDOM_SEED,
    val_fraction: float = VALIDATION_FRACTION,
) -> dict[str, Any]:
    """
    Train Baseline 2 on weak labels with the same stratified split as Baseline 1.

    Validation metrics are on the weak holdout only — not the Golden Set.
    """
    df = load_weak_training(weak_csv)
    x = df["customer_text"]
    y = df["weak_intent"]

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=val_fraction,
        random_state=seed,
        stratify=y,
    )

    pipe = build_baseline2_pipeline(seed=seed)
    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_val)
    labels = [lab for lab in INTENT_TAXONOMY if lab in set(y_val) | set(y_pred)]

    accuracy = float(accuracy_score(y_val, y_pred))
    macro_f1 = float(f1_score(y_val, y_pred, average="macro", labels=labels, zero_division=0))
    report = classification_report(
        y_val, y_pred, labels=labels, digits=3, zero_division=0
    )

    model_path = Path(model_path) if model_path is not None else BASELINE2_MODEL_PATH
    meta_path = Path(meta_path) if meta_path is not None else BASELINE2_META_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, model_path)
    meta = {
        "model": "baseline2_tfidf_linearsvc",
        "train_size": int(len(x_train)),
        "validation_size": int(len(x_val)),
        "seed": seed,
        "val_fraction": val_fraction,
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1,
        "classes": list(pipe.named_steps["clf"].classes_),
        "confidence_note": (
            "Prediction confidence_like is softmax(decision_function scores). "
            "It is NOT a calibrated probability (LinearSVC has no predict_proba)."
        ),
        "notes": (
            "Trained on weak_intent labels only. "
            "Validation metrics are on a weak holdout, not the Golden Set."
        ),
    }
    joblib.dump(meta, meta_path)

    return {
        **meta,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "classification_report": report,
    }


def load_baseline2(model_path: Path | None = None) -> Pipeline:
    """Load the saved Baseline 2 sklearn pipeline."""
    path = Path(model_path) if model_path is not None else BASELINE2_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline 2 model not found: {path}. "
            "Run: python scripts/train_baseline2.py"
        )
    return joblib.load(path)


def predict_intent_baseline2(
    text: str,
    model: Pipeline | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """
    Predict intent with Baseline 2.

    Returns:
      predicted_intent: str
      confidence_like: float
        Softmax of LinearSVC decision_function scores.
        NOT a calibrated probability.
      decision_scores: dict[str, float] per-class raw decision scores
    """
    pipe = model if model is not None else load_baseline2(model_path)
    text = str(text or "").strip()
    if not text:
        raise ValueError("text must be a non-empty customer message")

    pred = str(pipe.predict([text])[0])
    raw_scores = pipe.decision_function([text])[0]
    classes = list(pipe.named_steps["clf"].classes_)
    weights, confidence_like = decision_scores_to_confidence_like(raw_scores)

    return {
        "predicted_intent": pred,
        "confidence_like": confidence_like,
        "decision_scores": {
            str(c): float(s) for c, s in zip(classes, raw_scores, strict=True)
        },
        "score_weights": {
            str(c): float(w) for c, w in zip(classes, weights, strict=True)
        },
        "confidence_note": (
            "confidence_like = softmax(decision_function). "
            "Not a calibrated probability."
        ),
    }
