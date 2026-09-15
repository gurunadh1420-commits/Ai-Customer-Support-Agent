"""
Baseline 1: TF-IDF (unigrams+bigrams) + Logistic Regression.

Trained only on weakly labelled data — never on the Golden Set.
Weak labels are not ground truth; validation metrics are on a weak holdout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    BASELINE1_META_PATH,
    BASELINE1_MODEL_PATH,
    INTENT_TAXONOMY,
    WEAK_TRAINING_PATH,
)

RANDOM_SEED = 42
VALIDATION_FRACTION = 0.2


def load_weak_training(path: Path | None = None) -> pd.DataFrame:
    """Load weak training CSV (customer_text, weak_intent)."""
    csv_path = Path(path) if path is not None else WEAK_TRAINING_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Weak training file not found: {csv_path}. "
            "Run: python scripts/prepare_weak_training.py"
        )
    df = pd.read_csv(csv_path, dtype={"customer_tweet_id": str, "agent_tweet_id": str})
    required = {"customer_text", "weak_intent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Weak training CSV missing columns: {missing}")
    df = df.dropna(subset=["customer_text", "weak_intent"]).copy()
    df["customer_text"] = df["customer_text"].astype(str)
    df["weak_intent"] = df["weak_intent"].astype(str)
    # Keep only known taxonomy labels.
    df = df[df["weak_intent"].isin(INTENT_TAXONOMY)].reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable weak training rows after filtering.")
    return df


def build_baseline1_pipeline() -> Pipeline:
    """
    Explainable Baseline 1 pipeline.

    - TF-IDF word unigrams + bigrams
    - Logistic Regression with class_weight='balanced' for imbalance
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
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def train_baseline1(
    weak_csv: Path | None = None,
    model_path: Path | None = None,
    meta_path: Path | None = None,
    seed: int = RANDOM_SEED,
    val_fraction: float = VALIDATION_FRACTION,
) -> dict[str, Any]:
    """
    Train on weak labels, validate on a stratified holdout, save model.

    Returns a metrics/summary dict (not Golden Set evaluation).
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

    pipe = build_baseline1_pipeline()
    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_val)
    labels = [lab for lab in INTENT_TAXONOMY if lab in set(y_val) | set(y_pred)]

    accuracy = float(accuracy_score(y_val, y_pred))
    macro_f1 = float(f1_score(y_val, y_pred, average="macro", labels=labels, zero_division=0))
    report = classification_report(
        y_val, y_pred, labels=labels, digits=3, zero_division=0
    )

    model_path = Path(model_path) if model_path is not None else BASELINE1_MODEL_PATH
    meta_path = Path(meta_path) if meta_path is not None else BASELINE1_META_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, model_path)
    meta = {
        "model": "baseline1_tfidf_logreg",
        "train_size": int(len(x_train)),
        "validation_size": int(len(x_val)),
        "seed": seed,
        "val_fraction": val_fraction,
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1,
        "classes": list(pipe.named_steps["clf"].classes_),
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


def load_baseline1(model_path: Path | None = None) -> Pipeline:
    """Load the saved Baseline 1 sklearn pipeline."""
    path = Path(model_path) if model_path is not None else BASELINE1_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline 1 model not found: {path}. "
            "Run: python scripts/train_baseline1.py"
        )
    return joblib.load(path)


def predict_intent(
    text: str,
    model: Pipeline | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """
    Predict intent for one customer message.

    Returns:
      predicted_intent: str
      confidence: float  (max class probability)
    """
    pipe = model if model is not None else load_baseline1(model_path)
    text = str(text or "").strip()
    if not text:
        raise ValueError("text must be a non-empty customer message")

    pred = pipe.predict([text])[0]
    proba = pipe.predict_proba([text])[0]
    classes = list(pipe.named_steps["clf"].classes_)
    confidence = float(max(proba))
    # Optional full distribution for debugging (kept out of required return surface
    # but available if callers inspect the dict later — include briefly).
    return {
        "predicted_intent": str(pred),
        "confidence": confidence,
        "probabilities": {str(c): float(p) for c, p in zip(classes, proba, strict=True)},
    }
