"""
Build a ~200-example Golden Set candidate CSV for manual annotation.

proposed_intent is MACHINE-GENERATED only — not ground truth.

Usage:
  python scripts/prepare_golden_set_candidates.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import GOLDEN_SET_CANDIDATES_PATH, PROCESSED_SUPPORT_PAIRS_PATH
from src.golden_set import export_golden_set_candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample Golden Set annotation candidates.")
    p.add_argument("--pairs", type=Path, default=PROCESSED_SUPPORT_PAIRS_PATH)
    p.add_argument("--out", type=Path, default=GOLDEN_SET_CANDIDATES_PATH)
    p.add_argument("--size", type=int, default=200)
    p.add_argument("--per-intent", type=int, default=14)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("Building Golden Set candidates for MANUAL annotation only.")
    print("proposed_intent = MACHINE_CANDIDATE suggestions — NOT ground truth.")
    print(f"Pairs: {args.pairs}")

    _, summary = export_golden_set_candidates(
        out_path=args.out,
        pairs_path=args.pairs,
        target_size=args.size,
        per_intent=args.per_intent,
        seed=args.seed,
    )

    print(f"\nSaved: {summary['out_path']}")
    print(f"Examples sampled: {summary['n_examples']}")
    print(f"Ambiguous/multi-problem (heuristic): {summary['n_ambiguous_multi']}")
    print(f"Vague/unclear candidates (heuristic): {summary['n_vague_unclear_candidates']}")
    print(f"Boundary-tagged examples: {summary['n_boundary_tagged']}")
    print("\nExamples per proposed_intent (machine candidate primary only):")
    for intent, count in sorted(
        summary["per_proposed_intent"].items(), key=lambda kv: (-kv[1], kv[0])
    ):
        print(f"  {count:4d}  {intent}")


if __name__ == "__main__":
    main()
