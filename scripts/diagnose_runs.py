"""Report truncation and strict-answer quality for existing FP/quantized runs."""

import argparse

from src.diagnostics import generation_diagnostics
from src.utils import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp", default="runs/fp_outputs.jsonl")
    parser.add_argument("--quant", default="runs/quant_outputs.jsonl")
    parser.add_argument("--assumed-max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default="runs/diagnostics.json")
    args = parser.parse_args()
    report = generation_diagnostics(
        read_jsonl(args.fp), read_jsonl(args.quant), args.assumed_max_new_tokens
    )
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
