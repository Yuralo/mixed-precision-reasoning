"""Plot temperature accuracy and BNB4 rescue-set overlap."""

import argparse
import json

from src.plots import plot_temperature_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default="results/temperature/analysis.json")
    parser.add_argument("--output", default="results/figures/temperature_overlap.png")
    args = parser.parse_args()
    with open(args.analysis, encoding="utf-8") as handle:
        report = json.load(handle)
    plot_temperature_analysis(report, args.output)
    print(f"Wrote {args.output} and PDF counterpart")


if __name__ == "__main__":
    main()
