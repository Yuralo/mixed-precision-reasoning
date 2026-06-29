"""Train learned sensitivity models and compare cheap baselines."""

import argparse

from src.train_sensitivity_model import train_predictors
from src.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="runs/example_features.parquet")
    parser.add_argument("--output-dir", default="runs/models")
    parser.add_argument("--metrics", default="runs/predictor_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    args = parser.parse_args()
    report = train_predictors(args.features, args.output_dir, args.seed, args.test_size)
    write_json(args.metrics, report)
    print(report)


if __name__ == "__main__":
    main()
