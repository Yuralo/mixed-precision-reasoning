"""Configuration-aware evaluation entry point shared by FP and quantized scripts."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .datasets import load_reasoning_dataset
from .generation import evaluate_examples
from .load_model import load_model_and_tokenizer
from .utils import load_yaml, set_seed, write_jsonl_atomic


LOGGER = logging.getLogger(__name__)


def run_evaluation(args, forced_quantization: str | None = None) -> tuple[Path, Path]:
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    model_cfg = load_yaml(args.model_config)
    eval_cfg = load_yaml(args.eval_config)
    model_section = model_cfg.get("model", {})
    generation = model_cfg.get("generation", {})
    dataset_cfg = eval_cfg.get("dataset", {})

    seed = args.seed if args.seed is not None else eval_cfg.get("seed", model_cfg.get("seed", 42))
    set_seed(seed)
    name = args.model or model_section.get("name_or_path")
    quantization = forced_quantization or args.quantization or model_section.get("quantization", "none")
    requested_device = args.device or model_section.get("device", "auto")
    requested_dtype = args.dtype or model_section.get("dtype", "auto")
    LOGGER.info(
        "Starting evaluation: model=%s quantization=%s device=%s dtype=%s seed=%d",
        name,
        quantization,
        requested_device,
        requested_dtype,
        seed,
    )
    LOGGER.info("Loading tokenizer and model (first run may download weights)...")
    load_started = time.perf_counter()
    model, tokenizer, model_info = load_model_and_tokenizer(
        name_or_path=name,
        quantization=quantization,
        device=requested_device,
        dtype=requested_dtype,
        fake_quant_bits=args.fake_quant_bits or model_section.get("fake_quant_bits", 4),
        trust_remote_code=bool(model_section.get("trust_remote_code", False)),
    )
    LOGGER.info(
        "Model ready in %.1fs: resolved_device=%s resolved_dtype=%s backend=%s",
        time.perf_counter() - load_started,
        model_info["device"],
        model_info["dtype"],
        model_info["quantization"],
    )
    tiny = args.tiny if args.tiny is not None else bool(dataset_cfg.get("tiny", False))
    limit = args.limit if args.limit is not None else dataset_cfg.get("limit")
    dataset_split = args.dataset_split or dataset_cfg.get("split", "test")
    dataset_subset = args.dataset_subset or dataset_cfg.get("subset", "main")
    LOGGER.info(
        "Loading dataset=%s split=%s tiny=%s limit=%s...",
        dataset_cfg.get("name", "gsm8k"),
        dataset_split,
        tiny,
        limit,
    )
    examples = load_reasoning_dataset(
        name=dataset_cfg.get("name", "gsm8k"),
        split=dataset_split,
        subset=dataset_subset,
        limit=limit,
        offset=args.offset,
        tiny=tiny,
        seed=seed,
    )
    max_new_tokens = args.max_new_tokens or generation.get("max_new_tokens", 64)
    output_path = Path(args.output)
    token_path = Path(args.token_output)
    LOGGER.info(
        "Generating %d examples greedily with max_new_tokens=%d; progress every %d example(s)",
        len(examples),
        max_new_tokens,
        args.log_every,
    )

    def save_checkpoint(output_rows, token_rows) -> None:
        write_jsonl_atomic(output_path, output_rows)
        write_jsonl_atomic(token_path, token_rows)
        LOGGER.info("Checkpoint saved: %d examples -> %s", len(output_rows), output_path)

    outputs, token_features = evaluate_examples(
        model,
        tokenizer,
        examples,
        model_info,
        max_new_tokens=max_new_tokens,
        use_chat_template=bool(model_section.get("use_chat_template", True)),
        log_every=max(0, args.log_every),
        checkpoint_every=max(0, args.checkpoint_every),
        checkpoint_callback=save_checkpoint,
        stop_after_answer=args.stop_after_answer,
        answer_stop_grace_tokens=max(0, args.answer_stop_grace_tokens),
    )
    write_jsonl_atomic(output_path, outputs)
    write_jsonl_atomic(token_path, token_features)
    accuracy = sum(bool(row["correct"]) for row in outputs) / len(outputs) if outputs else 0.0
    total_seconds = sum(float(row["latency_seconds"]) for row in outputs)
    LOGGER.info(
        "Evaluation complete: examples=%d accuracy=%.3f generation_time=%s",
        len(outputs),
        accuracy,
        _format_duration(total_seconds),
    )
    LOGGER.info("Saved outputs=%s token_features=%s", output_path, token_path)
    return output_path, token_path


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"
