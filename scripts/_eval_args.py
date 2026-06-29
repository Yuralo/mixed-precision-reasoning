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
    parser.add_argument("--tiny", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--token-output", default=default_tokens)
    return parser
