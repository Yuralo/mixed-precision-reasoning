"""Deterministic autoregressive generation with cheap per-token uncertainty signals."""

from __future__ import annotations

import math
import logging
import time
from typing import Any, Callable

import torch

from .answer_extraction import (
    extract_answer,
    extract_explicit_answer,
    extract_hash_answer,
    extract_marked_answer,
    is_correct,
)


LOGGER = logging.getLogger(__name__)


def format_prompt(tokenizer: Any, prompt: str, use_chat_template: bool = True) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    return prompt


def _input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


@torch.inference_mode()
def generate_with_features(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 64,
    use_chat_template: bool = True,
    stop_after_answer: bool = False,
    answer_stop_grace_tokens: int = 4,
    do_sample: bool = False,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    if do_sample and temperature <= 0:
        raise ValueError("temperature must be positive when sampling")
    rendered = format_prompt(tokenizer, prompt, use_chat_template)
    encoded = tokenizer(rendered, return_tensors="pt")
    input_ids = encoded["input_ids"].to(_input_device(model))
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(input_ids.device)
    prompt_tokens = input_ids.shape[1]
    generated: list[int] = []
    token_rows: list[dict[str, Any]] = []
    past_key_values = None
    next_input = input_ids
    started = time.perf_counter()
    answer_detected_at: int | None = None
    stop_reason = "max_new_tokens"

    for position in range(max_new_tokens):
        outputs = model(
            input_ids=next_input,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :].float()
        if do_sample:
            logits = logits / temperature
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        top_probs, top_ids = torch.topk(probs, k=2, dim=-1)
        token_id = int(
            torch.multinomial(probs[0], 1, generator=generator).item()
            if do_sample
            else top_ids[0, 0].item()
        )
        entropy = float((-(probs * log_probs).sum(dim=-1))[0].item())
        margin = float((logits[0, top_ids[0, 0]] - logits[0, top_ids[0, 1]]).item())
        probability = float(probs[0, token_id].item())
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        token_rows.append(
            {
                "token_position": position,
                "token_id": token_id,
                "token": token_text,
                "token_probability": probability,
                "surprisal": -math.log(max(probability, 1e-12)),
                "entropy": entropy,
                "logit_margin": margin,
                "top2_token_id": int(top_ids[0, 1].item()),
                "top2_probability": float(top_probs[0, 1].item()),
                "top1_token_id": int(top_ids[0, 0].item()),
                "selected_was_top1": token_id == int(top_ids[0, 0].item()),
            }
        )
        generated.append(token_id)
        past_key_values = outputs.past_key_values
        if token_id == tokenizer.eos_token_id:
            stop_reason = "eos"
            break
        if stop_after_answer:
            partial_text = tokenizer.decode(generated, skip_special_tokens=True)
            if answer_detected_at is None and extract_marked_answer(partial_text) is not None:
                answer_detected_at = position
            if answer_detected_at is not None and position - answer_detected_at >= answer_stop_grace_tokens:
                stop_reason = "answer_detected"
                break
        next_input = torch.tensor([[token_id]], device=input_ids.device)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=input_ids.device)], dim=1
        )

    elapsed = time.perf_counter() - started
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "generation": text,
        "prompt_tokens": int(prompt_tokens),
        "generation_tokens": len(generated),
        "latency_seconds": elapsed,
        "tokens_per_second": len(generated) / elapsed if elapsed else None,
        "max_new_tokens": max_new_tokens,
        "stop_reason": stop_reason,
        "hit_max_new_tokens": stop_reason == "max_new_tokens",
        "has_hash_answer": extract_hash_answer(text) is not None,
        "has_explicit_answer": extract_explicit_answer(text) is not None,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else 0.0,
        "token_features": token_rows,
    }


def evaluate_examples(
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    model_info: dict[str, Any],
    max_new_tokens: int = 64,
    use_chat_template: bool = True,
    log_every: int = 1,
    checkpoint_every: int = 0,
    checkpoint_callback: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None] | None = None,
    stop_after_answer: bool = False,
    answer_stop_grace_tokens: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outputs, token_features = [], []
    total = len(examples)
    evaluation_started = time.perf_counter()
    for index, example in enumerate(examples, start=1):
        result = generate_with_features(
            model,
            tokenizer,
            example["prompt"],
            max_new_tokens,
            use_chat_template,
            stop_after_answer,
            answer_stop_grace_tokens,
        )
        row = {
            **example,
            "generation": result["generation"],
            "predicted_answer": extract_answer(result["generation"]),
            "reference_answer": extract_answer(example["reference"]),
            "correct": is_correct(result["generation"], example["reference"]),
            "prompt_tokens": result["prompt_tokens"],
            "generation_tokens": result["generation_tokens"],
            "latency_seconds": result["latency_seconds"],
            "tokens_per_second": result["tokens_per_second"],
            "max_new_tokens": result["max_new_tokens"],
            "stop_reason": result["stop_reason"],
            "hit_max_new_tokens": result["hit_max_new_tokens"],
            "has_hash_answer": result["has_hash_answer"],
            "has_explicit_answer": result["has_explicit_answer"],
            "model": model_info,
        }
        outputs.append(row)
        for token in result["token_features"]:
            token_features.append(
                {
                    "example_id": example["example_id"],
                    "dataset": example["dataset"],
                    "split": example["split"],
                    "run_type": model_info["quantization"],
                    **token,
                }
            )
        if log_every > 0 and (index == 1 or index % log_every == 0 or index == total):
            elapsed = time.perf_counter() - evaluation_started
            rate = index / elapsed if elapsed else 0.0
            eta = (total - index) / rate if rate else 0.0
            running_accuracy = sum(bool(item["correct"]) for item in outputs) / index
            LOGGER.info(
                "[%d/%d %5.1f%%] id=%s new_tokens=%d stop=%s example=%.1fs tok/s=%.1f "
                "running_acc=%.3f elapsed=%s eta=%s",
                index,
                total,
                100.0 * index / total if total else 100.0,
                example["example_id"],
                result["generation_tokens"],
                result["stop_reason"],
                result["latency_seconds"],
                result["tokens_per_second"] or 0.0,
                running_accuracy,
                _format_duration(elapsed),
                _format_duration(eta),
            )
        if checkpoint_callback and checkpoint_every > 0 and index % checkpoint_every == 0:
            checkpoint_callback(outputs, token_features)
    return outputs, token_features


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"
