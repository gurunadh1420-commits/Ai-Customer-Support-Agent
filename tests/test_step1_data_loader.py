"""
Step 1 tests: mixed/malformed IDs and conversation reconstruction.

Run from repo root:
  python tests/test_step1_data_loader.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import (
    normalize_id,
    parse_response_ids,
    prepare_brand_conversations,
)


def test_normalize_id_mixed_values() -> None:
    assert normalize_id(None) is None
    assert normalize_id("") is None
    assert normalize_id("nan") is None
    assert normalize_id("  ") is None
    # Float artifact from CSV parsers — strip only trailing ".0"
    assert normalize_id("119237.0") == "119237"
    # Large snowflake-like id must stay exact (no float conversion)
    big = "936425179630665728"
    assert normalize_id(big) == big
    # Non-numeric brand / author handles stay strings
    assert normalize_id("AppleSupport") == "AppleSupport"
    # Garbage that would break int64 inference
    assert normalize_id("not-an-int") == "not-an-int"
    assert normalize_id("123,456") == "123,456"


def test_parse_response_ids() -> None:
    assert parse_response_ids(None) == []
    assert parse_response_ids("10") == ["10"]
    assert parse_response_ids("10,11,12") == ["10", "11", "12"]
    assert parse_response_ids("10.0, 11.0") == ["10", "11"]


def test_prepare_handles_mixed_ids_via_csv(tmp_path: Path | None = None) -> None:
    base = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    csv_path = base / "twcs_mixed.csv"
    # Deliberately messy IDs: float-like strings, brand handle, comma responses.
    csv_path.write_text(
        "tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id\n"
        "100.0,custA,True,2017-10-31 00:00:00,battery dies fast,200.0,\n"
        "200.0,AppleSupport,False,2017-10-31 00:01:00,Which iOS version?,201,100.0\n"
        "201,custA,True,2017-10-31 00:02:00,iOS 11,,200.0\n"
        "900,custB,True,2017-10-31 01:00:00,where is my order?,901,\n"
        "901,AmazonHelp,False,2017-10-31 01:01:00,send order id,,900\n"
        "bad-id,AppleSupport,False,2017-10-31 02:00:00,orphan brand note,,\n",
        encoding="utf-8",
    )

    brand_tweets, conversations = prepare_brand_conversations(
        csv_path=csv_path,
        brand="AppleSupport",
        chunksize=2,  # force multiple chunks
    )

    # Amazon thread excluded; AppleSupport threads included.
    assert "AmazonHelp" not in set(brand_tweets["author_id"])
    assert "AppleSupport" in set(brand_tweets["author_id"])
    # Float artifacts normalized in tweet ids
    assert "100" in set(brand_tweets["tweet_id"].astype(str))
    assert "200" in set(brand_tweets["tweet_id"].astype(str))
    # At least the main 3-tweet thread + orphan brand note
    assert len(conversations) >= 1
    main = conversations.loc[conversations["conversation_id"] == "100"].iloc[0]
    assert main["num_tweets"] == 3
    assert "battery" in main["customer_text"]
    assert "iOS" in main["agent_text"]


def main() -> None:
    test_normalize_id_mixed_values()
    print("PASS test_normalize_id_mixed_values")
    test_parse_response_ids()
    print("PASS test_parse_response_ids")
    test_prepare_handles_mixed_ids_via_csv()
    print("PASS test_prepare_handles_mixed_ids_via_csv")
    print("All Step 1 tests passed.")


if __name__ == "__main__":
    main()
