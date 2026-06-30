"""Generate the evidence-gated thesis decision memo."""

import argparse
from pathlib import Path

from src.decision_memo import build_decision_memo, load_optional_json
from src.utils import ensure_parent, read_json, read_jsonl, write_json


def _require_raw_runs(runs_dir: Path) -> None:
    required = [
        runs_dir / f"gsm8k_{split}/{mode}_outputs.jsonl"
        for split in ("train", "test")
        for mode in ("fp", "quant")
    ] + [
        runs_dir / f"gsm8k_{split}/{mode}_token_features.jsonl"
        for split in ("train", "test")
        for mode in ("fp", "quant")
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise SystemExit(
            "Cannot prepare the memo because raw paired-run files are missing:\n"
            f"  - {formatted}\n"
            "Copy runs/gsm8k_train and runs/gsm8k_test to this checkout, then rerun."
        )


def prepare_missing(args: argparse.Namespace) -> None:
    """Build derived analysis files from raw JSONL without model inference."""
    runs_dir = Path(args.runs_dir)
    required_derived = [args.audit, args.trajectory, args.length, args.utility, args.prefix]
    if all(Path(path).exists() for path in required_derived):
        return
    _require_raw_runs(runs_dir)

    if not Path(args.audit).exists():
        from scripts.audit_results import audit_split

        print(f"Preparing missing audit: {args.audit}")
        report = {}
        for split in ("train", "test"):
            root = runs_dir / f"gsm8k_{split}"
            report[split] = audit_split(
                root / "fp_outputs.jsonl",
                root / "quant_outputs.jsonl",
                root / "fp_token_features.jsonl",
                root / "quant_token_features.jsonl",
                args.seed,
            )
        write_json(args.audit, report)

    train_records = Path(args.trajectory).parent / "train_trajectory_records.json"
    test_records = Path(args.trajectory).parent / "test_trajectory_records.json"
    if not Path(args.trajectory).exists() or not train_records.exists() or not test_records.exists():
        from scripts.analyze_existing_runs import analyze_split

        print(f"Preparing missing trajectory analysis in {Path(args.trajectory).parent}")
        for split in ("train", "test"):
            analyze_split(runs_dir, split, Path(args.trajectory).parent)

    if not Path(args.length).exists():
        from scripts.study_token_inflation import build_rows, run_study

        print(f"Preparing missing token-length study: {args.length}")
        root = runs_dir / "gsm8k_test"
        rows = build_rows(root / "fp_outputs.jsonl", root / "quant_outputs.jsonl")
        write_json(args.length, run_study(rows, args.seed))

    train_payload = read_json(train_records)["records"]
    test_payload = read_json(test_records)["records"]
    if not Path(args.utility).exists():
        from src.utility_controller import evaluate_controller

        print(f"Preparing missing utility controller: {args.utility}")
        report, predictions = evaluate_controller(
            train_payload, test_payload, Path(args.utility).parent, args.seed
        )
        write_json(args.utility, report)
        predictions.to_csv(ensure_parent(Path(args.utility).parent / "predictions.csv"), index=False)

    if not Path(args.prefix).exists():
        from src.prefix_prediction import evaluate_prefixes

        print(f"Preparing missing prefix analysis: {args.prefix}")
        report, _ = evaluate_prefixes(
            train_payload,
            test_payload,
            read_jsonl(runs_dir / "gsm8k_train/quant_token_features.jsonl"),
            read_jsonl(runs_dir / "gsm8k_test/quant_token_features.jsonl"),
            Path(args.prefix).parent,
            seed=args.seed,
        )
        write_json(args.prefix, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default="runs/research_audit.json")
    parser.add_argument("--trajectory", default="results/trajectory/test_trajectory_summary.json")
    parser.add_argument("--length", default="runs/gsm8k_test/token_length_study.json")
    parser.add_argument("--utility", default="results/utility_controller/metrics.json")
    parser.add_argument("--prefix", default="results/prefix_prediction/metrics.json")
    parser.add_argument("--temperature", default="results/temperature/analysis.json")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-prepare-missing",
        action="store_true",
        help="Fail instead of rebuilding derived artifacts from existing JSONL runs.",
    )
    parser.add_argument("--output", default="results/DECISION_MEMO.md")
    parser.add_argument("--decision-json", default="results/decision.json")
    args = parser.parse_args()
    if not args.no_prepare_missing:
        prepare_missing(args)
    memo, decision = build_decision_memo(
        read_json(args.audit),
        read_json(args.trajectory),
        read_json(args.length),
        read_json(args.utility),
        read_json(args.prefix),
        load_optional_json(args.temperature),
    )
    target = ensure_parent(args.output)
    target.write_text(memo, encoding="utf-8")
    write_json(args.decision_json, decision)
    print(f"Wrote {target}\n{decision}")


if __name__ == "__main__":
    main()
