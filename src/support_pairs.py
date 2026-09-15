"""
Extract high-confidence AppleSupport customer → agent reply pairs.

Unlike conversation-level grouping (which merges whole reply trees and can
pull in bystanders / promo roots), this module builds pairs from tweet-level
links only:

  - Primary: AppleSupport tweet.in_response_to_tweet_id → customer tweet
  - Secondary: customer.response_tweet_id lists an AppleSupport reply

A pair is kept only when:
  - customer tweet is inbound and not authored by the brand
  - agent tweet is authored by the brand and not inbound
  - both texts are non-empty
  - the agent tweet is a direct reply to that customer tweet

This is cleaner than conversation bags, but not perfect (topic mismatches and
DM-redirect boilerplate can still appear).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import BRAND, RAW_CSV_PATH
from src.data_loader import DEFAULT_CHUNKSIZE, load_raw_tweets, parse_response_ids

PAIR_COLUMNS = [
    "customer_tweet_id",
    "agent_tweet_id",
    "customer_text",
    "agent_text",
    "customer_created_at",
    "agent_created_at",
    "customer_author_id",
    "link_source",
]


def _tweet_lookup(tweets: pd.DataFrame) -> dict[str, pd.Series]:
    """Map tweet_id → row (first occurrence if duplicates)."""
    lookup: dict[str, pd.Series] = {}
    for _, row in tweets.iterrows():
        tid = row["tweet_id"]
        if tid is not None and tid not in lookup:
            lookup[str(tid)] = row
    return lookup


def _is_customer_tweet(row: pd.Series, brand: str) -> bool:
    return bool(row["inbound"]) and str(row["author_id"]) != brand


def _is_agent_tweet(row: pd.Series, brand: str) -> bool:
    return (not bool(row["inbound"])) and str(row["author_id"]) == brand


def _nonempty_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def extract_support_pairs(
    tweets: pd.DataFrame,
    brand: str = BRAND,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build clean customer→agent pairs from a tweet-level dataframe.

    Returns:
      pairs: one row per accepted pair
      stats: counts for quality reporting
    """
    lookup = _tweet_lookup(tweets)
    agent_tweets = tweets[tweets["author_id"].astype(str) == brand]
    # Brand rows that are true outbound support account tweets.
    agent_outbound = agent_tweets[~agent_tweets["inbound"].astype(bool)]

    stats: dict[str, Any] = {
        "apple_support_tweets": int(len(agent_tweets)),
        "apple_support_outbound_tweets": int(len(agent_outbound)),
        "candidate_pairs": 0,
        "final_pairs": 0,
        "dropped_missing_parent": 0,
        "dropped_parent_not_inbound_customer": 0,
        "dropped_empty_text": 0,
        "dropped_not_direct_reply": 0,
        "added_via_agent_parent": 0,
        "added_via_customer_response_id": 0,
    }

    # pair key → row dict
    pairs: dict[tuple[str, str], dict[str, Any]] = {}

    def try_add(customer: pd.Series, agent: pd.Series, link_source: str) -> None:
        cust_id = str(customer["tweet_id"])
        agent_id = str(agent["tweet_id"])
        key = (cust_id, agent_id)

        if not _is_customer_tweet(customer, brand):
            stats["dropped_parent_not_inbound_customer"] += 1
            return
        if not _is_agent_tweet(agent, brand):
            stats["dropped_parent_not_inbound_customer"] += 1
            return
        # Direct link check: agent must point at this customer.
        parent = agent["in_response_to_tweet_id"]
        if parent is None or str(parent) != cust_id:
            stats["dropped_not_direct_reply"] += 1
            return
        if not _nonempty_text(customer["text"]) or not _nonempty_text(agent["text"]):
            stats["dropped_empty_text"] += 1
            return

        if key in pairs:
            # Prefer recording that both link fields agreed.
            if link_source not in pairs[key]["link_source"]:
                pairs[key]["link_source"] = "agent_parent+customer_response_id"
            return

        pairs[key] = {
            "customer_tweet_id": cust_id,
            "agent_tweet_id": agent_id,
            "customer_text": str(customer["text"]).strip(),
            "agent_text": str(agent["text"]).strip(),
            "customer_created_at": customer["created_at"],
            "agent_created_at": agent["created_at"],
            "customer_author_id": str(customer["author_id"]),
            "link_source": link_source,
        }
        if link_source == "agent_parent":
            stats["added_via_agent_parent"] += 1
        else:
            stats["added_via_customer_response_id"] += 1

    # --- Primary path: AppleSupport reply → parent customer tweet ---
    for _, agent in agent_outbound.iterrows():
        parent_id = agent["in_response_to_tweet_id"]
        if parent_id is None:
            # Outbound brand tweet with no parent ≈ promo / root, not a support reply.
            continue
        stats["candidate_pairs"] += 1
        parent_id = str(parent_id)
        customer = lookup.get(parent_id)
        if customer is None:
            stats["dropped_missing_parent"] += 1
            continue
        try_add(customer, agent, link_source="agent_parent")

    # --- Secondary path: customer.response_tweet_id → AppleSupport tweets ---
    inbound = tweets[tweets["inbound"].astype(bool)]
    for _, customer in inbound.iterrows():
        if str(customer["author_id"]) == brand:
            continue
        for resp_id in parse_response_ids(customer.get("response_tweet_id")):
            agent = lookup.get(resp_id)
            if agent is None:
                continue
            if not _is_agent_tweet(agent, brand):
                continue
            # Count as candidate only if not already considered via parent path.
            key = (str(customer["tweet_id"]), str(agent["tweet_id"]))
            if key not in pairs:
                stats["candidate_pairs"] += 1
            try_add(customer, agent, link_source="customer_response_id")

    out = pd.DataFrame(list(pairs.values()), columns=PAIR_COLUMNS)
    if not out.empty:
        out = out.sort_values(["customer_created_at", "agent_created_at"]).reset_index(drop=True)
    stats["final_pairs"] = int(len(out))
    return out, stats


def prepare_support_pairs(
    csv_path: Path | None = None,
    brand: str = BRAND,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load brand-linked tweets and extract clean support pairs."""
    path = Path(csv_path) if csv_path is not None else RAW_CSV_PATH
    tweets = load_raw_tweets(path, brand=brand, chunksize=chunksize)
    pairs, stats = extract_support_pairs(tweets, brand=brand)
    stats["brand_linked_tweet_rows"] = int(len(tweets))
    return pairs, stats


def format_pair_examples(pairs: pd.DataFrame, n: int = 10) -> list[dict[str, Any]]:
    """Return up to n example pairs for console / README quality checks."""
    if pairs.empty:
        return []
    sample = pairs.head(n)
    examples: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        examples.append(
            {
                "customer_tweet_id": row["customer_tweet_id"],
                "agent_tweet_id": row["agent_tweet_id"],
                "customer_text": row["customer_text"],
                "agent_text": row["agent_text"],
                "link_source": row["link_source"],
            }
        )
    return examples
