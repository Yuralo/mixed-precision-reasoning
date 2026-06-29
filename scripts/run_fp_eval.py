"""Run deterministic full-precision evaluation and log token uncertainty."""

from scripts._eval_args import evaluation_parser
from src.eval_runner import run_evaluation


def main() -> None:
    parser = evaluation_parser(
        "Evaluate the full-precision model", "runs/fp_outputs.jsonl", "runs/fp_token_features.jsonl"
    )
    args = parser.parse_args()
    outputs, tokens = run_evaluation(args, forced_quantization="none")
    print(f"Wrote {outputs} and {tokens}")


if __name__ == "__main__":
    main()
