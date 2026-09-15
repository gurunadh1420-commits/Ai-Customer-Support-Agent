"""
Build TF-IDF cosine retrieval index from support pairs (excludes Golden Set).

Usage:
  python scripts/build_retrieval_index.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    GOLDEN_SET_CANDIDATES_PATH,
    PROCESSED_SUPPORT_PAIRS_PATH,
    RETRIEVAL_INDEX_PATH,
)
from src.retrieve import build_retrieval_index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build historical support retrieval index.")
    p.add_argument("--pairs", type=Path, default=PROCESSED_SUPPORT_PAIRS_PATH)
    p.add_argument("--golden", type=Path, default=GOLDEN_SET_CANDIDATES_PATH)
    p.add_argument("--out", type=Path, default=RETRIEVAL_INDEX_PATH)
    p.add_argument(
        "--include-golden",
        action="store_true",
        help="Do NOT exclude Golden Set IDs (default excludes them).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exclude_golden = not args.include_golden
    print("Building TF-IDF (1-2 grams) + cosine similarity retrieval index.")
    print(f"Pairs: {args.pairs}")
    print(f"Exclude Golden Set: {exclude_golden}")

    summary = build_retrieval_index(
        pairs_path=args.pairs,
        golden_path=args.golden,
        index_path=args.out,
        exclude_golden=exclude_golden,
    )
    print(f"Documents indexed: {summary['n_documents']:,}")
    print(f"TF-IDF features: {summary['n_features']:,}")
    print(f"Saved: {summary['index_path']}")


if __name__ == "__main__":
    main()
