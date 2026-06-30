"""Pair FP/quantized traces, compute structural deltas, and locate first divergence."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any

from .trajectory_metrics import ARITHMETIC_PATTERN, FINAL_MARKER_PATTERN, analyze_trajectory, paired_metric_deltas


OUTCOMES = (
    "fp_correct_q_correct",
    "fp_correct_q_wrong",
    "fp_wrong_q_correct",
    "fp_wrong_q_wrong",
)


def outcome_label(fp_correct: bool, quant_correct: bool) -> str:
    if fp_correct and quant_correct:
        return "fp_correct_q_correct"
    if fp_correct:
        return "fp_correct_q_wrong"
    if quant_correct:
        return "fp_wrong_q_correct"
    return "fp_wrong_q_wrong"


def group_token_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["example_id"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["token_position"]))
    return grouped


def first_divergence(fp_tokens: list[dict], quant_tokens: list[dict]) -> dict[str, Any]:
    aligned = min(len(fp_tokens), len(quant_tokens))
    position = next(
        (
            index
            for index in range(aligned)
            if int(fp_tokens[index]["token_id"]) != int(quant_tokens[index]["token_id"])
        ),
        None,
    )
    if position is None and len(fp_tokens) != len(quant_tokens):
        position = aligned
    if position is None:
        return {
            "first_divergence_position": None,
            "normalized_first_divergence": None,
            "divergence_before_arithmetic": None,
            "divergence_before_final_marker": None,
            "fp_entropy_at_divergence": None,
            "quant_entropy_at_divergence": None,
            "fp_margin_at_divergence": None,
            "quant_margin_at_divergence": None,
        }
    fp_prefix = "".join(row["token"] for row in fp_tokens[:position])
    q_prefix = "".join(row["token"] for row in quant_tokens[:position])
    denominator = max(1, min(len(fp_tokens), len(quant_tokens)))
    fp_row = fp_tokens[position] if position < len(fp_tokens) else None
    q_row = quant_tokens[position] if position < len(quant_tokens) else None
    return {
        "first_divergence_position": position,
        "normalized_first_divergence": position / denominator,
        "divergence_before_arithmetic": not bool(
            ARITHMETIC_PATTERN.search(fp_prefix) or ARITHMETIC_PATTERN.search(q_prefix)
        ),
        "divergence_before_final_marker": not bool(
            FINAL_MARKER_PATTERN.search(fp_prefix) or FINAL_MARKER_PATTERN.search(q_prefix)
        ),
        "fp_entropy_at_divergence": float(fp_row["entropy"]) if fp_row else None,
        "quant_entropy_at_divergence": float(q_row["entropy"]) if q_row else None,
        "fp_margin_at_divergence": float(fp_row["logit_margin"]) if fp_row else None,
        "quant_margin_at_divergence": float(q_row["logit_margin"]) if q_row else None,
    }


def aggregate_token_trace(tokens: list[dict[str, Any]]) -> dict[str, Any]:
    if not tokens:
        return {}
    entropy = [float(row["entropy"]) for row in tokens]
    margins = [float(row["logit_margin"]) for row in tokens]
    probabilities = [float(row["token_probability"]) for row in tokens]
    n = len(tokens)
    x_mean = (n - 1) / 2
    denominator = sum((index - x_mean) ** 2 for index in range(n))
    entropy_mean = statistics.fmean(entropy)
    entropy_slope = (
        sum((index - x_mean) * (value - entropy_mean) for index, value in enumerate(entropy))
        / denominator
        if denominator
        else 0.0
    )
    quarter = max(1, n // 4)
    return {
        "max_entropy": max(entropy),
        "mean_entropy": entropy_mean,
        "last_entropy": entropy[-1],
        "entropy_slope": entropy_slope,
        "early_entropy": statistics.fmean(entropy[:quarter]),
        "late_entropy": statistics.fmean(entropy[-quarter:]),
        "entropy_late_minus_early": statistics.fmean(entropy[-quarter:])
        - statistics.fmean(entropy[:quarter]),
        "min_logit_margin": min(margins),
        "mean_logit_margin": statistics.fmean(margins),
        "last_logit_margin": margins[-1],
        "min_token_probability": min(probabilities),
        "mean_token_probability": statistics.fmean(probabilities),
        "last_token_probability": probabilities[-1],
    }


def build_paired_records(
    fp_outputs: list[dict[str, Any]],
    quant_outputs: list[dict[str, Any]],
    fp_token_rows: list[dict[str, Any]],
    quant_token_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fp_by_id = {row["example_id"]: row for row in fp_outputs}
    quant_by_id = {row["example_id"]: row for row in quant_outputs}
    if fp_by_id.keys() != quant_by_id.keys():
        raise ValueError("FP and quantized output IDs do not match")
    fp_tokens = group_token_rows(fp_token_rows)
    quant_tokens = group_token_rows(quant_token_rows)
    records = []
    for example_id in sorted(fp_by_id):
        fp, quant = fp_by_id[example_id], quant_by_id[example_id]
        fp_metrics = analyze_trajectory(fp["generation"], int(fp["generation_tokens"]))
        quant_metrics = analyze_trajectory(quant["generation"], int(quant["generation_tokens"]))
        clean = (
            fp_metrics["has_explicit_answer"]
            and quant_metrics["has_explicit_answer"]
            and not fp.get("hit_max_new_tokens", False)
            and not quant.get("hit_max_new_tokens", False)
        )
        record = {
            "example_id": example_id,
            "dataset": fp["dataset"],
            "split": fp["split"],
            "question": fp["question"],
            "outcome": outcome_label(bool(fp["correct"]), bool(quant["correct"])),
            "utility_fp_minus_quant": int(bool(fp["correct"])) - int(bool(quant["correct"])),
            "fp_correct": int(bool(fp["correct"])),
            "quant_correct": int(bool(quant["correct"])),
            "clean": int(clean),
            "prompt_tokens": int(fp["prompt_tokens"]),
            "fp": fp_metrics,
            "quant": quant_metrics,
            "fp_runtime": aggregate_token_trace(fp_tokens.get(example_id, [])),
            "quant_runtime": aggregate_token_trace(quant_tokens.get(example_id, [])),
        }
        record.update(paired_metric_deltas(fp_metrics, quant_metrics))
        record.update(first_divergence(fp_tokens.get(example_id, []), quant_tokens.get(example_id, [])))
        records.append(record)
    return records


def _describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None}
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values)}


def summarize_paired_records(records: list[dict[str, Any]], entropy_bins: int = 20) -> dict[str, Any]:
    metric_names = [
        "delta_generated_tokens",
        "delta_line_count",
        "delta_sentence_count",
        "delta_numbered_step_count",
        "delta_arithmetic_expression_count",
        "delta_numeric_value_count",
        "delta_self_correction_count",
        "delta_repetition_score_4gram",
        "delta_normalized_final_answer_position",
        "first_divergence_position",
        "normalized_first_divergence",
        "fp_entropy_at_divergence",
        "quant_entropy_at_divergence",
        "fp_margin_at_divergence",
        "quant_margin_at_divergence",
    ]
    groups = {}
    for outcome in OUTCOMES:
        subset = [record for record in records if record["outcome"] == outcome]
        clean = [record for record in subset if record["clean"]]
        groups[outcome] = {
            "n": len(subset),
            "clean_n": len(clean),
            "metrics": {
                name: _describe([float(record[name]) for record in subset if record.get(name) is not None])
                for name in metric_names
            },
            "clean_metrics": {
                name: _describe([float(record[name]) for record in clean if record.get(name) is not None])
                for name in metric_names
            },
            "divergence_before_arithmetic_rate": _rate(subset, "divergence_before_arithmetic"),
            "divergence_before_final_marker_rate": _rate(subset, "divergence_before_final_marker"),
        }
    return {
        "num_examples": len(records),
        "clean_examples": sum(record["clean"] for record in records),
        "outcome_counts": dict(Counter(record["outcome"] for record in records)),
        "groups": groups,
    }


def _rate(records: list[dict[str, Any]], key: str) -> float | None:
    values = [bool(record[key]) for record in records if record.get(key) is not None]
    return sum(values) / len(values) if values else None


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flat = {
        key: value
        for key, value in record.items()
        if key not in {"fp", "quant", "fp_runtime", "quant_runtime"}
    }
    for prefix in ("fp", "quant", "fp_runtime", "quant_runtime"):
        for key, value in record[prefix].items():
            flat[f"{prefix}_{key}"] = value
    return flat
