"""Generate the standard research figure bundle from saved run artifacts."""

import argparse

from src.visualization import make_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparisons", default="runs/comparison.jsonl")
    parser.add_argument("--features", default="runs/example_features.parquet")
    parser.add_argument("--predictor-metrics", default="runs/predictor_metrics.json")
    parser.add_argument("--oracle", default="runs/oracle_recovery.json")
    parser.add_argument("--diagnostics", default="runs/diagnostics.json")
    parser.add_argument("--output-dir", default="runs/figures")
    parser.add_argument("--audit", default="runs/research_audit.json")
    parser.add_argument("--formats", default="png,pdf")
    args = parser.parse_args()
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    report = make_figures(
        args.comparisons,
        args.features,
        args.predictor_metrics,
        args.oracle,
        args.diagnostics,
        args.output_dir,
        formats,
        args.audit,
    )
    print(report)


if __name__ == "__main__":
    main()
