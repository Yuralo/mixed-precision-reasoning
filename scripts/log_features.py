"""Build token- and example-level Parquet feature tables."""

import argparse

from src.feature_logging import build_feature_tables
from src.utils import read_jsonl, write_parquet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-tokens", default="runs/fp_token_features.jsonl")
    parser.add_argument("--quant-tokens", default="runs/quant_token_features.jsonl")
    parser.add_argument("--comparisons", default="runs/comparison.jsonl")
    parser.add_argument("--token-output", default="runs/token_features.parquet")
    parser.add_argument("--example-output", default="runs/example_features.parquet")
    args = parser.parse_args()
    tokens, examples = build_feature_tables(
        read_jsonl(args.fp_tokens), read_jsonl(args.quant_tokens), read_jsonl(args.comparisons)
    )
    write_parquet(args.token_output, tokens)
    write_parquet(args.example_output, examples)
    print(f"Wrote {len(tokens)} token rows and {len(examples)} example rows")


if __name__ == "__main__":
    main()
