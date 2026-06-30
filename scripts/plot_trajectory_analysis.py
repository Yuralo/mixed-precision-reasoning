"""Create structure, divergence, and entropy-trajectory figures."""

import argparse
from pathlib import Path

from src.plots import (
    load_records,
    normalized_entropy_curves,
    plot_clean_structure_deltas,
    plot_entropy_trajectories,
    plot_first_divergence,
)
from src.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="results/trajectory/test_trajectory_records.json")
    parser.add_argument("--fp-tokens", default="runs/gsm8k_test/fp_token_features.jsonl")
    parser.add_argument("--quant-tokens", default="runs/gsm8k_test/quant_token_features.jsonl")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()
    records = load_records(args.records)
    output = Path(args.output_dir)
    plot_clean_structure_deltas(records, output / "trajectory_structure_deltas.png")
    plot_first_divergence(records, output / "first_divergence_histogram.png")
    curves = normalized_entropy_curves(records, args.fp_tokens, args.quant_tokens)
    write_json(output / "entropy_trajectories.json", curves)
    plot_entropy_trajectories(curves, output / "entropy_trajectories.png")
    print(f"Wrote trajectory figures to {output}")


if __name__ == "__main__":
    main()
