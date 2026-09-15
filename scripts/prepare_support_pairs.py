"""
CLI: build clean AppleSupport customer → agent support pairs.

Does not replace applesupport_conversations.csv — writes a separate file.

Usage (from repo root):
  python scripts/prepare_support_pairs.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import BRAND, PROCESSED_DIR, PROCESSED_SUPPORT_PAIRS_PATH, RAW_CSV_PATH
from src.support_pairs import format_pair_examples, prepare_support_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Extract clean {BRAND} customer→agent support pairs."
    )
    parser.add_argument("--raw", type=Path, default=RAW_CSV_PATH)
    parser.add_argument("--out", type=Path, default=PROCESSED_SUPPORT_PAIRS_PATH)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--examples", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Brand: {BRAND}")
    print(f"Loading: {args.raw}")
    print(f"Chunk size: {args.chunksize:,}")

    pairs, stats = prepare_support_pairs(
        csv_path=args.raw,
        brand=BRAND,
        chunksize=args.chunksize,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # Traceable columns only for the saved dataset (+ link_source for audit).
    save_cols = [
        "customer_tweet_id",
        "agent_tweet_id",
        "customer_text",
        "agent_text",
        "customer_created_at",
        "agent_created_at",
        "customer_author_id",
        "link_source",
    ]
    pairs.to_csv(args.out, index=False, columns=save_cols)

    print("--- quality check ---")
    print(f"Brand-linked tweet rows loaded: {stats['brand_linked_tweet_rows']:,}")
    print(f"AppleSupport tweets considered: {stats['apple_support_tweets']:,}")
    print(f"AppleSupport outbound tweets: {stats['apple_support_outbound_tweets']:,}")
    print(f"Candidate customer→agent pairs: {stats['candidate_pairs']:,}")
    print(f"Final clean pairs: {stats['final_pairs']:,}")
    print(f"Dropped missing parent: {stats['dropped_missing_parent']:,}")
    print(
        "Dropped parent not inbound customer: "
        f"{stats['dropped_parent_not_inbound_customer']:,}"
    )
    print(f"Dropped empty text: {stats['dropped_empty_text']:,}")
    print(f"Dropped not direct reply: {stats['dropped_not_direct_reply']:,}")
    print(f"Added via agent_parent: {stats['added_via_agent_parent']:,}")
    print(f"Added via customer_response_id: {stats['added_via_customer_response_id']:,}")
    print(f"Saved: {args.out}")
    print()
    print(
        "Limitations: pairs are high-confidence direct replies, not perfectly clean. "
        "Topic mismatches, DM-redirect boilerplate, and missing parent tweets can remain."
    )
    print()
    print(f"--- {args.examples} example pairs ---")
    for i, ex in enumerate(format_pair_examples(pairs, n=args.examples), start=1):
        print(f"\n[{i}] customer_tweet_id={ex['customer_tweet_id']} "
              f"agent_tweet_id={ex['agent_tweet_id']} link={ex['link_source']}")
        print(f"  customer: {ex['customer_text']}")
        print(f"  agent:    {ex['agent_text']}")


if __name__ == "__main__":
    main()
