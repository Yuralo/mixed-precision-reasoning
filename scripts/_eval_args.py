"""Arguments shared by full-precision and quantized evaluation scripts."""

from __future__ import annotations

import argparse


def evaluation_parser(description: str, default_output: str, default_tokens: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--model", help="Hugging Face model ID or local path")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--quantization", choices=["fake", "bnb4", "bnb8"])
    parser.add_argument("--fake-quant-bits", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0, help="Start at this dataset index")
    parser.add_argument("--dataset-split", help="Override the configured dataset split, e.g. train or test")
    parser.add_argument("--dataset-subset", help="Override the configured dataset subset")
    parser.add_argument("--tiny", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--stop-after-answer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop a few tokens after detecting an explicit #### numeric answer",
    )
    parser.add_argument("--answer-stop-grace-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-every", type=int, default=1, help="Log progress every N examples; 0 disables it")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Atomically save partial outputs every N examples; 0 disables checkpoints",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress informational progress logs")
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--token-output", default=default_tokens)
    return parser
