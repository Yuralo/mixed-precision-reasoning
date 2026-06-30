"""Generate repeated FP16 temperature samples for the quantization-noise control."""

import argparse
import logging
import time
from pathlib import Path

import torch

from src.answer_extraction import extract_answer, is_correct
from src.datasets import load_reasoning_dataset
from src.generation import generate_with_features
from src.load_model import load_model_and_tokenizer
from src.utils import set_seed, write_jsonl_atomic


LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperatures", default="0.3,0.7")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/temperature/temperature_outputs.jsonl")
    parser.add_argument("--token-output", default="results/temperature/temperature_token_features.jsonl")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--no-stop-after-answer", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    temperatures = [float(value) for value in args.temperatures.split(",") if value.strip()]
    set_seed(args.seed)
    LOGGER.info("Loading FP model %s on %s...", args.model, args.device)
    model, tokenizer, model_info = load_model_and_tokenizer(
        args.model, quantization="none", device=args.device, dtype=args.dtype
    )
    examples = load_reasoning_dataset(
        split=args.dataset_split,
        limit=args.limit,
        offset=args.offset,
        tiny=False,
        seed=args.seed,
    )
    outputs, token_outputs = [], []
    total = len(examples) * len(temperatures) * args.samples
    started = time.perf_counter()
    for temperature_index, temperature in enumerate(temperatures):
        for sample_id in range(args.samples):
            sample_seed = args.seed + temperature_index * 100_000 + sample_id * 10_000
            set_seed(sample_seed)
            generator = torch.Generator(device=model.get_input_embeddings().weight.device)
            generator.manual_seed(sample_seed)
            for example in examples:
                result = generate_with_features(
                    model,
                    tokenizer,
                    example["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    stop_after_answer=not args.no_stop_after_answer,
                    do_sample=True,
                    temperature=temperature,
                    generator=generator,
                )
                row = {
                    **example,
                    "condition": f"fp16_temp_{temperature}",
                    "temperature": temperature,
                    "sample_id": sample_id,
                    "sample_seed": sample_seed,
                    "generation": result["generation"],
                    "predicted_answer": extract_answer(result["generation"]),
                    "reference_answer": extract_answer(example["reference"]),
                    "correct": is_correct(result["generation"], example["reference"]),
                    "prompt_tokens": result["prompt_tokens"],
                    "generation_tokens": result["generation_tokens"],
                    "latency_seconds": result["latency_seconds"],
                    "tokens_per_second": result["tokens_per_second"],
                    "stop_reason": result["stop_reason"],
                    "hit_max_new_tokens": result["hit_max_new_tokens"],
                    "has_explicit_answer": result["has_explicit_answer"],
                    "model": model_info,
                }
                outputs.append(row)
                for token in result["token_features"]:
                    token_outputs.append(
                        {
                            "example_id": example["example_id"],
                            "condition": row["condition"],
                            "temperature": temperature,
                            "sample_id": sample_id,
                            **token,
                        }
                    )
                count = len(outputs)
                if args.log_every and (count == 1 or count % args.log_every == 0 or count == total):
                    elapsed = time.perf_counter() - started
                    eta = elapsed / count * (total - count)
                    LOGGER.info(
                        "[%d/%d %.1f%%] T=%.1f sample=%d id=%s correct=%s tokens=%d elapsed=%.1fm eta=%.1fm",
                        count, total, count / total * 100, temperature, sample_id,
                        example["example_id"], row["correct"], row["generation_tokens"],
                        elapsed / 60, eta / 60,
                    )
                if args.checkpoint_every and count % args.checkpoint_every == 0:
                    write_jsonl_atomic(args.output, outputs)
                    write_jsonl_atomic(args.token_output, token_outputs)
                    LOGGER.info("Checkpoint saved: %d completions", count)
    write_jsonl_atomic(args.output, outputs)
    write_jsonl_atomic(args.token_output, token_outputs)
    LOGGER.info("Complete: wrote %s and %s", args.output, args.token_output)


if __name__ == "__main__":
    main()
