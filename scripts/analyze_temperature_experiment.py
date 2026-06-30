"""Compare FP16 temperature success sets with deterministic BNB4 rescues."""

import argparse
import json

from src.temperature_experiment import analyze_temperature_outputs
from src.utils import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp", default="runs/gsm8k_test/fp_outputs.jsonl")
    parser.add_argument("--quant", default="runs/gsm8k_test/quant_outputs.jsonl")
    parser.add_argument("--temperature", default="results/temperature/temperature_outputs.jsonl")
    parser.add_argument("--fp-tokens", default="runs/gsm8k_test/fp_token_features.jsonl")
    parser.add_argument("--quant-tokens", default="runs/gsm8k_test/quant_token_features.jsonl")
    parser.add_argument("--temperature-tokens", default="results/temperature/temperature_token_features.jsonl")
    parser.add_argument("--output", default="results/temperature/analysis.json")
    args = parser.parse_args()
    report = analyze_temperature_outputs(
        read_jsonl(args.fp),
        read_jsonl(args.quant),
        read_jsonl(args.temperature),
        read_jsonl(args.fp_tokens),
        read_jsonl(args.quant_tokens),
        read_jsonl(args.temperature_tokens),
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
