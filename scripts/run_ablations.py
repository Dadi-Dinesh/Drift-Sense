#!/usr/bin/env python3
"""
Ablation and Evaluation Suite Entry Point (R5 Technical Validation).

Calls collect_final_results.py to process all final model predictions (seeds 42, 43, 44),
execute trajectory correction with E4vec, and generate the full set of ablation CSVs.

Usage:
    python scripts/run_ablations.py [--check]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from scripts.collect_final_results import run_collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R5 Technical Validation and Ablation Suite.")
    parser.add_argument("--check", action="store_true", help="Run validation checks after collection")
    args = parser.parse_args()

    success = run_collection(check_only=args.check)
    if not success and args.check:
        sys.exit(1)


if __name__ == "__main__":
    main()
