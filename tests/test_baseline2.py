"""
Tests for Baseline 2 train/load/predict (tiny synthetic data).

Run:
  python tests/test_baseline2.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.baseline2 import (
    decision_scores_to_confidence_like,
    load_baseline2,
    predict_intent_baseline2,
    train_baseline2,
)
from src.config import INTENT_TAXONOMY
import numpy as np


def _tiny_weak_csv(path: Path) -> None:
    rows = []
    templates = {
        "battery_drain": "my battery drains so fast on this iphone {}",
        "charging_issue": "my iphone will not charge with the cable {}",
        "software_update_issue": "after the ios update my phone is broken {}",
        "hardware_issue": "the touch bar is not working on my macbook {}",
        "network_connectivity": "wifi will not connect and keeps dropping {}",
        "account_access": "my apple id is locked verification code missing {}",
        "app_store_issue": "app store will not load cannot download apps {}",
        "app_issue": "safari keeps crashing and will not open pages {}",
        "payment_or_refund": "please refund this unauthorized charge now {}",
        "subscription_management": "cannot find manage subscription button {}",
        "storage_issue": "storage almost full not enough storage space {}",
        "howto_or_feature": "how do I turn on do not disturb mode {}",
        "other_or_unclear": "thank you {}",
    }
    n = 0
    for intent, tmpl in templates.items():
        for i in range(12):
            n += 1
            rows.append(
                {
                    "customer_tweet_id": str(n),
                    "agent_tweet_id": str(1000 + n),
                    "customer_text": tmpl.format(i),
                    "weak_intent": intent,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_confidence_like_from_decision_scores() -> None:
    weights, conf = decision_scores_to_confidence_like(np.array([1.0, 3.0, 0.5]))
    assert abs(float(weights.sum()) - 1.0) < 1e-6
    assert 0.0 <= conf <= 1.0
    assert conf == float(weights.max())


def test_train_load_predict() -> None:
    base = Path(tempfile.mkdtemp())
    weak = base / "weak.csv"
    model = base / "model.joblib"
    meta = base / "meta.joblib"
    _tiny_weak_csv(weak)

    result = train_baseline2(
        weak_csv=weak,
        model_path=model,
        meta_path=meta,
        seed=0,
        val_fraction=0.25,
    )
    assert result["train_size"] > 0
    assert model.exists()
    assert "calibrated" not in result["confidence_note"].lower() or "NOT" in result["confidence_note"]

    pipe = load_baseline2(model)
    clf = pipe.named_steps["clf"]
    assert not hasattr(clf, "predict_proba")

    out = predict_intent_baseline2(
        "my battery drains so fast on this iphone today", model=pipe
    )
    assert out["predicted_intent"] in INTENT_TAXONOMY
    assert "confidence_like" in out
    assert 0.0 <= float(out["confidence_like"]) <= 1.0
    assert "calibrated probability" in out["confidence_note"].lower()
    assert "not" in out["confidence_note"].lower()


def main() -> None:
    test_confidence_like_from_decision_scores()
    print("PASS test_confidence_like_from_decision_scores")
    test_train_load_predict()
    print("PASS test_train_load_predict")
    print("All Baseline 2 tests passed.")


if __name__ == "__main__":
    main()
