"""Create the four-way FP/quantized comparison labels and summary."""

import argparse

from src.compare_outputs import compare_runs
from src.utils import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp", default="runs/fp_outputs.jsonl")
    parser.add_argument("--quant", default="runs/quant_outputs.jsonl")
    parser.add_argument("--output", default="runs/comparison.jsonl")
    parser.add_argument("--summary", default="runs/summary.json")
    args = parser.parse_args()
    comparisons, summary = compare_runs(read_jsonl(args.fp), read_jsonl(args.quant))
    write_jsonl(args.output, comparisons)
    write_json(args.summary, summary)
    print(summary)


if __name__ == "__main__":
    main()
