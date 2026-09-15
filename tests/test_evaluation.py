"""
Unit tests for src/evaluate.py.

These tests use ONLY small synthetic data — they do NOT load the
Golden Set CSV or either trained model file, so they run fast and
in isolation from the full dataset.

Run:
  python tests/test_evaluation.py
  python -m pytest tests/test_evaluation.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

try:
    import pytest
    _PYTEST_AVAILABLE = True
except ImportError:
    _PYTEST_AVAILABLE = False
    # Provide a minimal stub so the class/function bodies parse correctly
    # when running via main() without pytest installed.
    class _PytestStub:
        class raises:
            def __init__(self, exc, match=None):
                import re
                self._exc = exc
                self._match = match
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, tb):
                if exc_type is None:
                    raise AssertionError(f"Expected {self._exc.__name__} to be raised")
                if not issubclass(exc_type, self._exc):
                    return False  # re-raise unexpected exception
                if self._match and not __import__("re").search(self._match, str(exc_val)):
                    raise AssertionError(
                        f"Pattern {self._match!r} not found in {str(exc_val)!r}"
                    )
                return True  # suppress the expected exception

        @staticmethod
        def approx(value, **kwargs):
            # Simple wrapper: returns a float comparison object
            class Approx:
                def __init__(self, v):
                    self._v = v
                def __eq__(self, other):
                    return abs(other - self._v) < 1e-6
                def __repr__(self):
                    return f"approx({self._v})"
            return Approx(value)

    pytest = _PytestStub()


from src.config import INTENT_TAXONOMY
from src.evaluate import (
    batch_predict,
    compute_metrics,
    load_golden_set,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic Golden Set DataFrames
# ---------------------------------------------------------------------------

def _make_golden_df(
    n: int = 20,
    labels: list[str] | None = None,
    empty_label_idx: list[int] | None = None,
    empty_text_idx: list[int] | None = None,
) -> pd.DataFrame:
    """Build a minimal synthetic golden-set DataFrame for testing load logic."""
    if labels is None:
        # Cycle through the first few taxonomy labels.
        cycle = INTENT_TAXONOMY[:5]
        labels = [cycle[i % len(cycle)] for i in range(n)]
    rows = []
    for i in range(n):
        rows.append(
            {
                "example_id": f"GS-{i+1:04d}",
                "customer_tweet_id": str(1000 + i),
                "agent_tweet_id": str(2000 + i),
                "customer_text": f"test customer message number {i}",
                "agent_text": f"test agent reply number {i}",
                "proposed_intent": f"MACHINE_CANDIDATE: {labels[i]}",
                "annotator_label": labels[i],
                "annotator_notes": "",
            }
        )
    df = pd.DataFrame(rows)
    if empty_label_idx:
        for idx in empty_label_idx:
            df.loc[idx, "annotator_label"] = ""
    if empty_text_idx:
        for idx in empty_text_idx:
            df.loc[idx, "customer_text"] = ""
    return df


def _write_golden_csv(df: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "golden_set_final.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Tests: load_golden_set
# ---------------------------------------------------------------------------

class TestLoadGoldenSet:

    def test_valid_golden_set_loads(self, tmp_path):
        df = _make_golden_df(n=20)
        path = _write_golden_csv(df, tmp_path)
        loaded = load_golden_set(path)
        assert len(loaded) == 20
        assert "annotator_label" in loaded.columns
        assert "customer_text" in loaded.columns

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_golden_set(tmp_path / "does_not_exist.csv")

    def test_empty_annotator_label_raises(self, tmp_path):
        df = _make_golden_df(n=10, empty_label_idx=[3, 7])
        path = _write_golden_csv(df, tmp_path)
        with pytest.raises(ValueError, match="empty annotator_label"):
            load_golden_set(path)

    def test_empty_customer_text_raises(self, tmp_path):
        df = _make_golden_df(n=10, empty_text_idx=[2])
        path = _write_golden_csv(df, tmp_path)
        with pytest.raises(ValueError, match="empty customer_text"):
            load_golden_set(path)

    def test_out_of_taxonomy_label_raises(self, tmp_path):
        labels = INTENT_TAXONOMY[:9] + ["UNKNOWN_INTENT"] * 11
        df = _make_golden_df(n=20, labels=labels)
        path = _write_golden_csv(df, tmp_path)
        with pytest.raises(ValueError, match="out-of-taxonomy"):
            load_golden_set(path)

    def test_missing_required_column_raises(self, tmp_path):
        df = _make_golden_df(n=5)
        df = df.drop(columns=["annotator_label"])
        path = _write_golden_csv(df, tmp_path)
        with pytest.raises(ValueError, match="missing columns"):
            load_golden_set(path)


# ---------------------------------------------------------------------------
# Tests: compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_perfect_predictions(self):
        labels = ["battery_drain", "charging_issue", "app_issue"] * 4
        metrics = compute_metrics(labels, labels)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["macro_f1"] == pytest.approx(1.0)
        for intent in ["battery_drain", "charging_issue", "app_issue"]:
            assert metrics["per_intent"][intent]["precision"] == pytest.approx(1.0)
            assert metrics["per_intent"][intent]["recall"] == pytest.approx(1.0)
            assert metrics["per_intent"][intent]["f1"] == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        y_true = ["battery_drain"] * 6
        y_pred = ["charging_issue"] * 6
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(0.0)

    def test_mixed_predictions(self):
        y_true = ["battery_drain", "charging_issue", "battery_drain", "charging_issue"]
        y_pred = ["battery_drain", "battery_drain", "battery_drain", "charging_issue"]
        metrics = compute_metrics(y_true, y_pred)
        # 3 out of 4 correct
        assert metrics["accuracy"] == pytest.approx(0.75)
        assert 0.0 <= metrics["macro_f1"] <= 1.0

    def test_accuracy_range(self):
        y_true = INTENT_TAXONOMY[:5] * 4
        y_pred = INTENT_TAXONOMY[1:6] * 4  # all wrong (shifted by one)
        metrics = compute_metrics(y_true, y_pred)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0

    def test_per_intent_keys(self):
        y_true = ["battery_drain", "app_issue", "storage_issue"]
        y_pred = ["battery_drain", "app_issue", "app_issue"]
        metrics = compute_metrics(y_true, y_pred)
        for intent in ["battery_drain", "app_issue", "storage_issue"]:
            assert intent in metrics["per_intent"]
            row = metrics["per_intent"][intent]
            assert "precision" in row
            assert "recall" in row
            assert "f1" in row
            assert "support" in row

    def test_support_counts(self):
        y_true = ["battery_drain"] * 3 + ["app_issue"] * 5
        y_pred = ["battery_drain"] * 3 + ["app_issue"] * 5
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["per_intent"]["battery_drain"]["support"] == 3
        assert metrics["per_intent"]["app_issue"]["support"] == 5

    def test_confusion_matrix_shape(self):
        y_true = ["battery_drain", "charging_issue", "app_issue"]
        y_pred = ["battery_drain", "charging_issue", "charging_issue"]
        metrics = compute_metrics(
            y_true, y_pred, labels=["battery_drain", "charging_issue", "app_issue"]
        )
        cm = metrics["confusion_matrix"]
        assert cm.shape == (3, 3)

    def test_confusion_matrix_diagonal_correct(self):
        y_true = ["battery_drain", "charging_issue"]
        y_pred = ["battery_drain", "charging_issue"]
        metrics = compute_metrics(
            y_true, y_pred, labels=["battery_drain", "charging_issue"]
        )
        cm = metrics["confusion_matrix"]
        assert cm[0, 0] == 1  # battery_drain correctly predicted
        assert cm[1, 1] == 1  # charging_issue correctly predicted

    def test_confusion_matrix_off_diagonal(self):
        y_true = ["battery_drain", "battery_drain"]
        y_pred = ["battery_drain", "charging_issue"]
        metrics = compute_metrics(
            y_true, y_pred, labels=["battery_drain", "charging_issue"]
        )
        cm = metrics["confusion_matrix"]
        assert cm[0, 0] == 1   # true battery_drain, pred battery_drain
        assert cm[0, 1] == 1   # true battery_drain, pred charging_issue

    def test_labels_field_returned(self):
        y_true = ["battery_drain", "app_issue"]
        y_pred = ["battery_drain", "app_issue"]
        metrics = compute_metrics(y_true, y_pred)
        assert "labels" in metrics
        assert isinstance(metrics["labels"], list)
        assert len(metrics["labels"]) > 0

    def test_classification_report_text_returned(self):
        y_true = ["battery_drain", "app_issue"]
        y_pred = ["battery_drain", "app_issue"]
        metrics = compute_metrics(y_true, y_pred)
        assert "classification_report_text" in metrics
        assert "battery_drain" in metrics["classification_report_text"]


# ---------------------------------------------------------------------------
# Tests: batch_predict
# ---------------------------------------------------------------------------

class TestBatchPredict:

    def _make_minimal_pipeline(self):
        """Build a tiny real sklearn Pipeline trained on minimal data."""
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=500, random_state=0)),
        ])
        texts = [
            "battery drains fast",
            "my iphone won't charge",
            "after the update nothing works",
            "battery is dying so quickly",
            "charger is broken",
            "software update broke my phone",
        ]
        labels = [
            "battery_drain",
            "charging_issue",
            "software_update_issue",
            "battery_drain",
            "charging_issue",
            "software_update_issue",
        ]
        pipe.fit(texts, labels)
        return pipe

    def test_batch_predict_returns_list(self):
        pipe = self._make_minimal_pipeline()
        result = batch_predict(pipe, ["my battery is draining"])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_batch_predict_length_matches_input(self):
        pipe = self._make_minimal_pipeline()
        texts = ["text one", "text two", "text three"]
        result = batch_predict(pipe, texts)
        assert len(result) == len(texts)

    def test_batch_predict_empty_input(self):
        pipe = self._make_minimal_pipeline()
        result = batch_predict(pipe, [])
        assert result == []

    def test_batch_predict_returns_strings(self):
        pipe = self._make_minimal_pipeline()
        result = batch_predict(pipe, ["battery drains", "won't charge"])
        for r in result:
            assert isinstance(r, str)

    def test_batch_predict_known_labels(self):
        pipe = self._make_minimal_pipeline()
        known_labels = {"battery_drain", "charging_issue", "software_update_issue"}
        result = batch_predict(pipe, ["battery drains fast", "won't charge"])
        for r in result:
            assert r in known_labels


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all tests manually without pytest (fallback)."""
    t = TestLoadGoldenSet()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        t.test_valid_golden_set_loads(tmp)
        print("PASS test_valid_golden_set_loads")
        t.test_missing_file_raises(tmp)
        print("PASS test_missing_file_raises")
        t.test_empty_annotator_label_raises(tmp)
        print("PASS test_empty_annotator_label_raises")
        t.test_empty_customer_text_raises(tmp)
        print("PASS test_empty_customer_text_raises")
        t.test_out_of_taxonomy_label_raises(tmp)
        print("PASS test_out_of_taxonomy_label_raises")
        t.test_missing_required_column_raises(tmp)
        print("PASS test_missing_required_column_raises")

    m = TestComputeMetrics()
    m.test_perfect_predictions()
    print("PASS test_perfect_predictions")
    m.test_all_wrong_predictions()
    print("PASS test_all_wrong_predictions")
    m.test_mixed_predictions()
    print("PASS test_mixed_predictions")
    m.test_accuracy_range()
    print("PASS test_accuracy_range")
    m.test_per_intent_keys()
    print("PASS test_per_intent_keys")
    m.test_support_counts()
    print("PASS test_support_counts")
    m.test_confusion_matrix_shape()
    print("PASS test_confusion_matrix_shape")
    m.test_confusion_matrix_diagonal_correct()
    print("PASS test_confusion_matrix_diagonal_correct")
    m.test_confusion_matrix_off_diagonal()
    print("PASS test_confusion_matrix_off_diagonal")
    m.test_labels_field_returned()
    print("PASS test_labels_field_returned")
    m.test_classification_report_text_returned()
    print("PASS test_classification_report_text_returned")

    b = TestBatchPredict()
    b.test_batch_predict_returns_list()
    print("PASS test_batch_predict_returns_list")
    b.test_batch_predict_length_matches_input()
    print("PASS test_batch_predict_length_matches_input")
    b.test_batch_predict_empty_input()
    print("PASS test_batch_predict_empty_input")
    b.test_batch_predict_returns_strings()
    print("PASS test_batch_predict_returns_strings")
    b.test_batch_predict_known_labels()
    print("PASS test_batch_predict_known_labels")

    print("\nAll evaluation unit tests passed.")


if __name__ == "__main__":
    main()
