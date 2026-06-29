"""Run fake or bitsandbytes quantized evaluation on the same examples."""

from scripts._eval_args import evaluation_parser
from src.eval_runner import run_evaluation


def main() -> None:
    parser = evaluation_parser(
        "Evaluate a quantized model", "runs/quant_outputs.jsonl", "runs/quant_token_features.jsonl"
    )
    parser.set_defaults(quantization="fake")
    args = parser.parse_args()
    outputs, tokens = run_evaluation(args)
    print(f"Wrote {outputs} and {tokens}")


if __name__ == "__main__":
    main()
