"""Compute the example-level oracle upper-bound recovery curve."""

import argparse

from src.oracle_recovery import example_recovery_accuracy
from src.utils import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparisons", default="runs/comparison.jsonl")
    parser.add_argument("--output", default="runs/oracle_recovery.json")
    args = parser.parse_args()
    rows = read_jsonl(args.comparisons)
    ranked = sorted(rows, key=lambda row: row["comparison_label"] == "fp_correct_q_wrong", reverse=True)
    curve = []
    for budget in (0.0, 0.05, 0.10, 0.20, 0.50, 1.0):
        count = round(len(rows) * budget)
        selected = {row["example_id"] for row in ranked[:count]}
        curve.append(
            {"budget": budget, "num_interventions": count, "accuracy": example_recovery_accuracy(rows, selected)}
        )
    write_json(args.output, {"method": "known-final-flip example oracle", "curve": curve})
    print(curve)


if __name__ == "__main__":
    main()
