"""
Tests for TF-IDF cosine retrieval (tiny synthetic pairs).

Run:
  python tests/test_retrieve.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.retrieve import build_retrieval_index, load_retrieval_index, retrieve_similar


def _write_pairs(path: Path) -> None:
    rows = [
        {
            "customer_tweet_id": "10",
            "agent_tweet_id": "11",
            "customer_text": "my battery drains so fast overnight on iphone",
            "agent_text": "Which iOS version are you using for the battery drain?",
        },
        {
            "customer_tweet_id": "20",
            "agent_tweet_id": "21",
            "customer_text": "app store will not load and cannot download apps",
            "agent_text": "Try restarting then open the App Store again.",
        },
        {
            "customer_tweet_id": "30",
            "agent_tweet_id": "31",
            "customer_text": "wifi keeps dropping and bluetooth will not connect",
            "agent_text": "Forget the network and reconnect to Wi-Fi.",
        },
        {
            "customer_tweet_id": "40",
            "agent_tweet_id": "41",
            "customer_text": "iphone battery life is terrible and drains quickly",
            "agent_text": "Check battery usage in Settings for drain causes.",
        },
        # Duplicate-ish for min_df=2 vocabulary
        {
            "customer_tweet_id": "50",
            "agent_tweet_id": "51",
            "customer_text": "my battery drains quickly during the day on iphone",
            "agent_text": "We can look into battery performance together.",
        },
        {
            "customer_tweet_id": "60",
            "agent_tweet_id": "61",
            "customer_text": "cannot download apps because app store is broken",
            "agent_text": "Confirm App Store connectivity on Wi-Fi and cellular.",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_load_retrieve_and_exclude_golden() -> None:
    base = Path(tempfile.mkdtemp())
    pairs = base / "pairs.csv"
    golden = base / "golden.csv"
    index_path = base / "index.joblib"
    _write_pairs(pairs)

    # Mark one battery example as golden → must not appear in retrieval results.
    pd.DataFrame(
        [
            {
                "example_id": "GS-0001",
                "customer_tweet_id": "10",
                "agent_tweet_id": "11",
                "customer_text": "x",
                "agent_text": "y",
                "proposed_intent": "MACHINE_CANDIDATE: battery_drain",
                "annotator_label": "",
                "annotator_notes": "",
            }
        ]
    ).to_csv(golden, index=False)

    summary = build_retrieval_index(
        pairs_path=pairs,
        golden_path=golden,
        index_path=index_path,
        exclude_golden=True,
    )
    assert summary["n_documents"] == 5
    assert index_path.exists()

    payload = load_retrieval_index(index_path)
    assert "10" not in payload["customer_tweet_id"]

    results = retrieve_similar(
        "battery drains overnight on my iphone",
        k=3,
        index=payload,
    )
    assert len(results) == 3
    for row in results:
        assert set(row) == {
            "customer_tweet_id",
            "agent_tweet_id",
            "customer_text",
            "agent_text",
            "similarity_score",
        }
        assert "10" != row["customer_tweet_id"]
        assert 0.0 <= row["similarity_score"] <= 1.0 + 1e-6

    # Top hit should be another battery example.
    assert "battery" in results[0]["customer_text"].lower()
    # Deterministic: same query twice → identical ranking
    again = retrieve_similar("battery drains overnight on my iphone", k=3, index=payload)
    assert [r["customer_tweet_id"] for r in results] == [
        r["customer_tweet_id"] for r in again
    ]


def main() -> None:
    test_build_load_retrieve_and_exclude_golden()
    print("PASS test_build_load_retrieve_and_exclude_golden")
    print("All retrieval tests passed.")


if __name__ == "__main__":
    main()
