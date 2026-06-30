"""Train four-outcome utility-aware precision routers on existing trajectories."""

import argparse
import json

from src.plots import load_records
from src.utility_controller import evaluate_controller
from src.utils import ensure_parent, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-records", default="results/trajectory/train_trajectory_records.json")
    parser.add_argument("--test-records", default="results/trajectory/test_trajectory_records.json")
    parser.add_argument("--output-dir", default="results/utility_controller")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report, predictions = evaluate_controller(
        load_records(args.train_records), load_records(args.test_records), args.output_dir, args.seed
    )
    write_json(f"{args.output_dir}/metrics.json", report)
    predictions.to_csv(ensure_parent(f"{args.output_dir}/predictions.csv"), index=False)
    print(json.dumps({
        "best_learned_at_10pct": report["best_learned_at_10pct"],
        "best_learned_accuracy_at_10pct": report["best_learned_accuracy_at_10pct"],
        "static": report["static"],
        "baseline_accuracy_at_10pct": {
            name: next(point["accuracy"] for point in curve if point["budget"] == 0.10)
            for name, curve in report["baselines"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
