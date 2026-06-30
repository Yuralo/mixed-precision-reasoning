"""Evaluate counterfactual precision routing from early quantized prefixes."""

import argparse
import json

from src.plots import load_records
from src.prefix_prediction import evaluate_prefixes
from src.utils import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-records", default="results/trajectory/train_trajectory_records.json")
    parser.add_argument("--test-records", default="results/trajectory/test_trajectory_records.json")
    parser.add_argument("--train-tokens", default="runs/gsm8k_train/quant_token_features.jsonl")
    parser.add_argument("--test-tokens", default="runs/gsm8k_test/quant_token_features.jsonl")
    parser.add_argument("--budgets", default="16,32,64")
    parser.add_argument("--output-dir", default="results/prefix_prediction")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    budgets = tuple(int(value) for value in args.budgets.split(",") if value.strip())
    report, _ = evaluate_prefixes(
        load_records(args.train_records),
        load_records(args.test_records),
        read_jsonl(args.train_tokens),
        read_jsonl(args.test_tokens),
        args.output_dir,
        budgets,
        args.seed,
    )
    write_json(f"{args.output_dir}/metrics.json", report)
    compact = {}
    for budget, models in report["results"].items():
        compact[budget] = {}
        for name, values in models.items():
            point = next(item for item in values["intervention_curve"] if item["budget"] == 0.10)
            compact[budget][name] = {
                "beneficial_roc_auc": values["beneficial_roc_auc"],
                "beneficial_pr_auc": values["beneficial_pr_auc"],
                "accuracy_at_10pct": point["accuracy"],
            }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
