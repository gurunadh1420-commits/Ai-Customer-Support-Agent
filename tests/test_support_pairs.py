"""
Tests for direct customer → AppleSupport support-pair extraction.

Run from repo root:
  python tests/test_support_pairs.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.support_pairs import extract_support_pairs, prepare_support_pairs


def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


def test_keeps_direct_inbound_customer_to_applesupport() -> None:
    tweets = _rows_to_df(
        [
            {
                "tweet_id": "1",
                "author_id": "custA",
                "inbound": True,
                "created_at": "2017-10-01 10:00:00+00:00",
                "text": "@AppleSupport iPhone won't charge",
                "response_tweet_id": "2",
                "in_response_to_tweet_id": None,
            },
            {
                "tweet_id": "2",
                "author_id": "AppleSupport",
                "inbound": False,
                "created_at": "2017-10-01 10:05:00+00:00",
                "text": "@custA Which iOS version are you using?",
                "response_tweet_id": None,
                "in_response_to_tweet_id": "1",
            },
        ]
    )
    pairs, stats = extract_support_pairs(tweets, brand="AppleSupport")
    assert len(pairs) == 1
    assert pairs.iloc[0]["customer_tweet_id"] == "1"
    assert pairs.iloc[0]["agent_tweet_id"] == "2"
    assert "won't charge" in pairs.iloc[0]["customer_text"]
    assert "iOS" in pairs.iloc[0]["agent_text"]
    assert stats["final_pairs"] == 1


def test_excludes_bystander_to_bystander() -> None:
    tweets = _rows_to_df(
        [
            {
                "tweet_id": "10",
                "author_id": "user1",
                "inbound": True,
                "created_at": "2017-10-01 11:00:00+00:00",
                "text": "my phone is broken",
                "response_tweet_id": "11",
                "in_response_to_tweet_id": None,
            },
            {
                "tweet_id": "11",
                "author_id": "user2",
                "inbound": True,
                "created_at": "2017-10-01 11:01:00+00:00",
                "text": "@user1 same here",
                "response_tweet_id": None,
                "in_response_to_tweet_id": "10",
            },
            # Unrelated valid Apple pair so dataframe is non-trivial
            {
                "tweet_id": "20",
                "author_id": "custB",
                "inbound": True,
                "created_at": "2017-10-01 12:00:00+00:00",
                "text": "@AppleSupport help",
                "response_tweet_id": "21",
                "in_response_to_tweet_id": None,
            },
            {
                "tweet_id": "21",
                "author_id": "AppleSupport",
                "inbound": False,
                "created_at": "2017-10-01 12:01:00+00:00",
                "text": "@custB What is happening?",
                "response_tweet_id": None,
                "in_response_to_tweet_id": "20",
            },
        ]
    )
    pairs, _ = extract_support_pairs(tweets, brand="AppleSupport")
    assert set(zip(pairs["customer_tweet_id"], pairs["agent_tweet_id"])) == {("20", "21")}


def test_excludes_promo_root_without_parent() -> None:
    tweets = _rows_to_df(
        [
            {
                "tweet_id": "100",
                "author_id": "AppleSupport",
                "inbound": False,
                "created_at": "2017-10-01 09:00:00+00:00",
                "text": "Welcome to Apple Support on Twitter!",
                "response_tweet_id": "101",
                "in_response_to_tweet_id": None,  # promo / root
            },
            {
                "tweet_id": "101",
                "author_id": "custC",
                "inbound": True,
                "created_at": "2017-10-01 09:10:00+00:00",
                "text": "@AppleSupport my screen is black",
                "response_tweet_id": "102",
                "in_response_to_tweet_id": "100",
            },
            {
                "tweet_id": "102",
                "author_id": "AppleSupport",
                "inbound": False,
                "created_at": "2017-10-01 09:15:00+00:00",
                "text": "@custC Sorry to hear that. Which device?",
                "response_tweet_id": None,
                "in_response_to_tweet_id": "101",
            },
        ]
    )
    pairs, _ = extract_support_pairs(tweets, brand="AppleSupport")
    # Promo root alone is not a pair; only customer 101 → agent 102.
    assert len(pairs) == 1
    assert pairs.iloc[0]["customer_tweet_id"] == "101"
    assert pairs.iloc[0]["agent_tweet_id"] == "102"


def test_excludes_agent_reply_to_outbound_non_customer() -> None:
    tweets = _rows_to_df(
        [
            {
                "tweet_id": "200",
                "author_id": "SomeBrand",
                "inbound": False,
                "created_at": "2017-10-01 08:00:00+00:00",
                "text": "outbound non-customer",
                "response_tweet_id": "201",
                "in_response_to_tweet_id": None,
            },
            {
                "tweet_id": "201",
                "author_id": "AppleSupport",
                "inbound": False,
                "created_at": "2017-10-01 08:05:00+00:00",
                "text": "@SomeBrand hello",
                "response_tweet_id": None,
                "in_response_to_tweet_id": "200",
            },
        ]
    )
    pairs, stats = extract_support_pairs(tweets, brand="AppleSupport")
    assert len(pairs) == 0
    assert stats["candidate_pairs"] >= 1


def test_preserves_source_tweet_ids() -> None:
    tweets = _rows_to_df(
        [
            {
                "tweet_id": "936425179630665728",
                "author_id": "custD",
                "inbound": True,
                "created_at": "2017-10-01 13:00:00+00:00",
                "text": "@AppleSupport battery drains overnight",
                "response_tweet_id": "936425179630665729",
                "in_response_to_tweet_id": None,
            },
            {
                "tweet_id": "936425179630665729",
                "author_id": "AppleSupport",
                "inbound": False,
                "created_at": "2017-10-01 13:05:00+00:00",
                "text": "@custD Let's look into that battery issue.",
                "response_tweet_id": None,
                "in_response_to_tweet_id": "936425179630665728",
            },
        ]
    )
    pairs, _ = extract_support_pairs(tweets, brand="AppleSupport")
    assert pairs.iloc[0]["customer_tweet_id"] == "936425179630665728"
    assert pairs.iloc[0]["agent_tweet_id"] == "936425179630665729"


def test_prepare_support_pairs_from_csv() -> None:
    base = Path(tempfile.mkdtemp())
    csv_path = base / "twcs_pairs.csv"
    csv_path.write_text(
        "tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id\n"
        "1,custA,True,Tue Oct 31 10:00:00 +0000 2017,@AppleSupport wifi drops,2,\n"
        "2,AppleSupport,False,Tue Oct 31 10:05:00 +0000 2017,@custA What iOS version?,,1\n"
        "9,userX,True,Tue Oct 31 11:00:00 +0000 2017,hey friend,10,\n"
        "10,userY,True,Tue Oct 31 11:01:00 +0000 2017,@userX hi,,9\n"
        "100,AppleSupport,False,Tue Oct 31 09:00:00 +0000 2017,Promo root,101,\n",
        encoding="utf-8",
    )
    pairs, stats = prepare_support_pairs(
        csv_path=csv_path, brand="AppleSupport", chunksize=2
    )
    assert len(pairs) == 1
    assert pairs.iloc[0]["customer_tweet_id"] == "1"
    assert pairs.iloc[0]["agent_tweet_id"] == "2"
    assert stats["final_pairs"] == 1
    assert "customer_created_at" in pairs.columns
    assert "agent_created_at" in pairs.columns


def main() -> None:
    test_keeps_direct_inbound_customer_to_applesupport()
    print("PASS test_keeps_direct_inbound_customer_to_applesupport")
    test_excludes_bystander_to_bystander()
    print("PASS test_excludes_bystander_to_bystander")
    test_excludes_promo_root_without_parent()
    print("PASS test_excludes_promo_root_without_parent")
    test_excludes_agent_reply_to_outbound_non_customer()
    print("PASS test_excludes_agent_reply_to_outbound_non_customer")
    test_preserves_source_tweet_ids()
    print("PASS test_preserves_source_tweet_ids")
    test_prepare_support_pairs_from_csv()
    print("PASS test_prepare_support_pairs_from_csv")
    print("All support-pair tests passed.")


if __name__ == "__main__":
    main()
