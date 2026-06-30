"""Analyze existing paired outputs without new model inference."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.paired_analysis import build_paired_records, flatten_record, summarize_paired_records
from src.utils import ensure_parent, read_jsonl, write_json


def write_csv(path: str | Path, records: list[dict]) -> None:
    rows = [flatten_record(record) for record in records]
    target = ensure_parent(path)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_split(runs_dir: Path, split: str, output_dir: Path) -> None:
    source = runs_dir / f"gsm8k_{split}"
    records = build_paired_records(
        read_jsonl(source / "fp_outputs.jsonl"),
        read_jsonl(source / "quant_outputs.jsonl"),
        read_jsonl(source / "fp_token_features.jsonl"),
        read_jsonl(source / "quant_token_features.jsonl"),
    )
    write_json(output_dir / f"{split}_trajectory_summary.json", summarize_paired_records(records))
    write_json(output_dir / f"{split}_trajectory_records.json", {"records": records})
    write_csv(output_dir / f"{split}_trajectory_records.csv", records)
    print(f"{split}: wrote {len(records)} paired trajectory records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-dir", default="results/trajectory")
    args = parser.parse_args()
    for split in ("train", "test"):
        analyze_split(Path(args.runs_dir), split, Path(args.output_dir))


if __name__ == "__main__":
    main()
