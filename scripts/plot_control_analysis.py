"""Create routing, calibration, and early-prefix figures."""

import argparse
import json
from pathlib import Path

from src.plots import (
    plot_controller_calibration,
    plot_prefix_prediction,
    plot_risk_coverage,
    plot_utility_controller,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--utility", default="results/utility_controller/metrics.json")
    parser.add_argument("--predictions", default="results/utility_controller/predictions.csv")
    parser.add_argument("--prefix", default="results/prefix_prediction/metrics.json")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()
    with open(args.utility, encoding="utf-8") as handle:
        utility = json.load(handle)
    with open(args.prefix, encoding="utf-8") as handle:
        prefix = json.load(handle)
    output = Path(args.output_dir)
    plot_utility_controller(utility, output / "utility_routing_curve.png")
    plot_controller_calibration(
        args.predictions,
        utility["best_learned_at_10pct"],
        output / "utility_calibration.png",
    )
    plot_risk_coverage(utility, output / "utility_risk_coverage.png")
    plot_prefix_prediction(prefix, output / "prefix_prediction.png")
    print(f"Wrote control-analysis figures to {output}")


if __name__ == "__main__":
    main()
