"""Align token traces and aggregate cheap uncertainty features per example."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any


def build_feature_tables(
    fp_tokens: list[dict[str, Any]],
    quant_tokens: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparison_by_id = {row["example_id"]: row for row in comparisons}
    fp_by_key = {(row["example_id"], row["token_position"]): row for row in fp_tokens}
    quant_by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    token_rows = []

    for quant in quant_tokens:
        example_id = quant["example_id"]
        if example_id not in comparison_by_id:
            continue
        comp = comparison_by_id[example_id]
        key = (example_id, quant["token_position"])
        fp = fp_by_key.get(key)
        row = {
            **quant,
            "phase": "quant",
            "fp_token_id": fp.get("token_id") if fp else None,
            "fp_entropy": fp.get("entropy") if fp else None,
            "fp_logit_margin": fp.get("logit_margin") if fp else None,
            "top_token_disagreement": bool(fp and fp["token_id"] != quant["token_id"]),
            "comparison_label": comp["comparison_label"],
            "target_quantization_failure": int(comp["comparison_label"] == "fp_correct_q_wrong"),
            "eligible_fp_correct": int(comp["fp_correct"]),
        }
        token_rows.append(row)
        quant_by_example[example_id].append(row)

    example_rows = []
    for example_id, tokens in quant_by_example.items():
        comp = comparison_by_id[example_id]
        entropies = [float(row["entropy"]) for row in tokens]
        margins = [float(row["logit_margin"]) for row in tokens]
        probabilities = [float(row["token_probability"]) for row in tokens]
        disagreements = [int(row["top_token_disagreement"]) for row in tokens]
        fp_entropies = [float(row["fp_entropy"]) for row in tokens if row["fp_entropy"] is not None]
        example_rows.append(
            {
                "example_id": example_id,
                "dataset": comp["dataset"],
                "split": comp["split"],
                "comparison_label": comp["comparison_label"],
                "target_quantization_failure": int(comp["comparison_label"] == "fp_correct_q_wrong"),
                "eligible_fp_correct": int(comp["fp_correct"]),
                "prompt_tokens": comp["prompt_tokens"],
                "generation_tokens": len(tokens),
                "fp_hit_max_new_tokens": comp.get("fp_hit_max_new_tokens"),
                "quant_hit_max_new_tokens": comp.get("quant_hit_max_new_tokens"),
                "fp_has_hash_answer": comp.get("fp_has_hash_answer"),
                "quant_has_hash_answer": comp.get("quant_has_hash_answer"),
                "fp_has_explicit_answer": comp.get("fp_has_explicit_answer", comp.get("fp_has_hash_answer")),
                "quant_has_explicit_answer": comp.get("quant_has_explicit_answer", comp.get("quant_has_hash_answer")),
                "max_entropy": max(entropies),
                "mean_entropy": mean(entropies),
                "min_logit_margin": min(margins),
                "mean_logit_margin": mean(margins),
                "min_token_probability": min(probabilities),
                "mean_token_probability": mean(probabilities),
                "top_token_disagreement_rate": mean(disagreements),
                "first_top_token_disagreement": next(
                    (row["token_position"] for row in tokens if row["top_token_disagreement"]), -1
                ),
                "mean_fp_entropy": mean(fp_entropies) if fp_entropies else None,
            }
        )
    return token_rows, example_rows
