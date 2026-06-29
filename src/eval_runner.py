"""Configuration-aware evaluation entry point shared by FP and quantized scripts."""

from __future__ import annotations

from pathlib import Path

from .datasets import load_reasoning_dataset
from .generation import evaluate_examples
from .load_model import load_model_and_tokenizer
from .utils import load_yaml, set_seed, write_jsonl


def run_evaluation(args, forced_quantization: str | None = None) -> tuple[Path, Path]:
    model_cfg = load_yaml(args.model_config)
    eval_cfg = load_yaml(args.eval_config)
    model_section = model_cfg.get("model", {})
    generation = model_cfg.get("generation", {})
    dataset_cfg = eval_cfg.get("dataset", {})

    seed = args.seed if args.seed is not None else eval_cfg.get("seed", model_cfg.get("seed", 42))
    set_seed(seed)
    name = args.model or model_section.get("name_or_path")
    quantization = forced_quantization or args.quantization or model_section.get("quantization", "none")
    model, tokenizer, model_info = load_model_and_tokenizer(
        name_or_path=name,
        quantization=quantization,
        device=args.device or model_section.get("device", "auto"),
        dtype=args.dtype or model_section.get("dtype", "auto"),
        fake_quant_bits=args.fake_quant_bits or model_section.get("fake_quant_bits", 4),
        trust_remote_code=bool(model_section.get("trust_remote_code", False)),
    )
    tiny = args.tiny if args.tiny is not None else bool(dataset_cfg.get("tiny", False))
    limit = args.limit if args.limit is not None else dataset_cfg.get("limit")
    examples = load_reasoning_dataset(
        name=dataset_cfg.get("name", "gsm8k"),
        split=dataset_cfg.get("split", "test"),
        subset=dataset_cfg.get("subset", "main"),
        limit=limit,
        tiny=tiny,
        seed=seed,
    )
    max_new_tokens = args.max_new_tokens or generation.get("max_new_tokens", 64)
    outputs, token_features = evaluate_examples(
        model,
        tokenizer,
        examples,
        model_info,
        max_new_tokens=max_new_tokens,
        use_chat_template=bool(model_section.get("use_chat_template", True)),
    )
    output_path = Path(args.output)
    token_path = Path(args.token_output)
    write_jsonl(output_path, outputs)
    write_jsonl(token_path, token_features)
    return output_path, token_path
