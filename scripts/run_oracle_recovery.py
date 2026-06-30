"""Compute the example-level oracle upper-bound recovery curve."""

import argparse

from src.oracle_recovery import oracle_recovery_report
from src.utils import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparisons", default="runs/comparison.jsonl")
    parser.add_argument("--output", default="runs/oracle_recovery.json")
    args = parser.parse_args()
    rows = read_jsonl(args.comparisons)
    report = oracle_recovery_report(rows, [0.0, 0.05, 0.10, 0.20, 0.50, 1.0])
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
