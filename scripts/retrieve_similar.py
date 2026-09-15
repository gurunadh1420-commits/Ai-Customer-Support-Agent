"""
Retrieve top historical AppleSupport pairs for a customer message.

Usage:
  python scripts/retrieve_similar.py --text "my battery drains overnight" --k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RETRIEVAL_INDEX_PATH
from src.retrieve import retrieve_similar


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieve similar historical support pairs.")
    p.add_argument("--text", required=True, help="New customer message")
    p.add_argument("--k", type=int, default=5, help="Number of neighbors (default 5)")
    p.add_argument("--index", type=Path, default=RETRIEVAL_INDEX_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = retrieve_similar(args.text, k=args.k, index_path=args.index)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
