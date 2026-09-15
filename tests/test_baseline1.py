"""
Tests for Baseline 1 train/load/predict (uses tiny synthetic data).

Run:
  python tests/test_baseline1.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.classify import load_baseline1, predict_intent, train_baseline1
from src.config import INTENT_TAXONOMY


def _tiny_weak_csv(path: Path) -> None:
    # Enough rows per class for a stratified 80/20 split.
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
    assert set(templates) == set(INTENT_TAXONOMY)
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


def test_train_load_predict(tmp_path: Path | None = None) -> None:
    base = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    weak = base / "weak.csv"
    model = base / "model.joblib"
    meta = base / "meta.joblib"
    _tiny_weak_csv(weak)

    result = train_baseline1(
        weak_csv=weak,
        model_path=model,
        meta_path=meta,
        seed=0,
        val_fraction=0.25,
    )
    assert result["train_size"] > 0
    assert result["validation_size"] > 0
    assert model.exists()

    pipe = load_baseline1(model)
    out = predict_intent("my battery drains so fast on this iphone today", model=pipe)
    assert "predicted_intent" in out
    assert "confidence" in out
    assert out["predicted_intent"] in INTENT_TAXONOMY
    assert 0.0 <= float(out["confidence"]) <= 1.0


def main() -> None:
    test_train_load_predict()
    print("PASS test_train_load_predict")
    print("All Baseline 1 tests passed.")


if __name__ == "__main__":
    main()
