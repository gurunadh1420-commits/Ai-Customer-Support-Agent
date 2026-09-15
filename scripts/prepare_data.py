"""
CLI for Step 1: load twcs.csv, rebuild AppleSupport conversations, save CSV.

Usage (from repo root):
  python scripts/prepare_data.py
  python scripts/prepare_data.py --raw data/raw/twcs.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/prepare_data.py` without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import BRAND, PROCESSED_CONVERSATIONS_PATH, PROCESSED_DIR, RAW_CSV_PATH
from src.data_loader import conversation_summary, prepare_brand_conversations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Prepare {BRAND} conversations from Customer Support on Twitter."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=RAW_CSV_PATH,
        help=f"Path to twcs.csv (default: {RAW_CSV_PATH})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROCESSED_CONVERSATIONS_PATH,
        help=f"Output conversations CSV (default: {PROCESSED_CONVERSATIONS_PATH})",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Rows per pandas chunk when reading the raw CSV (default: 100000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Brand: {BRAND}")
    print(f"Loading: {args.raw}")
    print(f"Chunk size: {args.chunksize:,}")

    brand_tweets, conversations = prepare_brand_conversations(
        csv_path=args.raw,
        brand=BRAND,
        chunksize=args.chunksize,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    conversations.to_csv(args.out, index=False)

    stats = conversation_summary(conversations)
    print(f"Brand tweet rows (in brand conversations): {len(brand_tweets):,}")
    print(f"Conversations written: {stats['num_conversations']:,}")
    print(f"Avg tweets / conversation: {stats['avg_tweets_per_conversation']:.2f}")
    print(f"With customer message: {stats['conversations_with_customer_message']:,}")
    print(f"With agent reply: {stats['conversations_with_agent_reply']:,}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
