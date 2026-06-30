"""Generate the evidence-gated thesis decision memo."""

import argparse

from src.decision_memo import build_decision_memo, load_optional_json
from src.utils import ensure_parent, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default="runs/research_audit.json")
    parser.add_argument("--trajectory", default="results/trajectory/test_trajectory_summary.json")
    parser.add_argument("--length", default="runs/gsm8k_test/token_length_study.json")
    parser.add_argument("--utility", default="results/utility_controller/metrics.json")
    parser.add_argument("--prefix", default="results/prefix_prediction/metrics.json")
    parser.add_argument("--temperature", default="results/temperature/analysis.json")
    parser.add_argument("--output", default="results/DECISION_MEMO.md")
    parser.add_argument("--decision-json", default="results/decision.json")
    args = parser.parse_args()
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
