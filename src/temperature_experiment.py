"""Temperature controls for testing whether quantization is ordinary decoding noise."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import statistics

from .paired_analysis import aggregate_token_trace


def _jaccard(left: set[str], right: set[str]) -> float | None:
    union = left | right
    return len(left & right) / len(union) if union else None


def analyze_temperature_outputs(
    fp_outputs: list[dict[str, Any]],
    quant_outputs: list[dict[str, Any]],
    sampled_outputs: list[dict[str, Any]],
    fp_tokens: list[dict[str, Any]] | None = None,
    quant_tokens: list[dict[str, Any]] | None = None,
    sampled_tokens: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    fp = {row["example_id"]: row for row in fp_outputs}
    quant = {row["example_id"]: row for row in quant_outputs}
    sampled_by_temperature: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in sampled_outputs:
        sampled_by_temperature[float(row["temperature"])].append(row)

    report: dict[str, Any] = {"temperatures": {}}
    common_sample_ids = {row["example_id"] for row in sampled_outputs} & fp.keys() & quant.keys()
    quant_rescue = {
        example_id
        for example_id in common_sample_ids
        if not fp[example_id]["correct"] and quant[example_id]["correct"]
    }
    quant_induced_failure = {
        example_id
        for example_id in common_sample_ids
        if fp[example_id]["correct"] and not quant[example_id]["correct"]
    }
    report["paired_subset"] = {
        "num_examples": len(common_sample_ids),
        "fp_accuracy": sum(bool(fp[key]["correct"]) for key in common_sample_ids) / len(common_sample_ids),
        "quant_accuracy": sum(bool(quant[key]["correct"]) for key in common_sample_ids) / len(common_sample_ids),
        "fp_mean_generation_tokens": statistics.fmean(
            int(fp[key]["generation_tokens"]) for key in common_sample_ids
        ),
        "quant_mean_generation_tokens": statistics.fmean(
            int(quant[key]["generation_tokens"]) for key in common_sample_ids
        ),
        "quant_rescue_count": len(quant_rescue),
        "quant_induced_failure_count": len(quant_induced_failure),
    }

    for temperature, rows in sorted(sampled_by_temperature.items()):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["example_id"] in common_sample_ids:
                grouped[row["example_id"]].append(row)
        any_success = {
            example_id for example_id, values in grouped.items() if any(value["correct"] for value in values)
        }
        majority_success = {
            example_id
            for example_id, values in grouped.items()
            if sum(bool(value["correct"]) for value in values) > len(values) / 2
        }
        temp_rescue = {example_id for example_id in any_success if not fp[example_id]["correct"]}
        majority_rescue = {
            example_id for example_id in majority_success if not fp[example_id]["correct"]
        }
        majority_failure = {
            example_id
            for example_id, values in grouped.items()
            if fp[example_id]["correct"]
            and sum(bool(value["correct"]) for value in values) <= len(values) / 2
        }
        all_sample_failure = {
            example_id
            for example_id, values in grouped.items()
            if fp[example_id]["correct"] and not any(value["correct"] for value in values)
        }
        answer_diversity = []
        for values in grouped.values():
            answers = {value.get("predicted_answer") for value in values}
            answer_diversity.append(len(answers))
        mean_tokens = statistics.fmean(
            int(value["generation_tokens"]) for values in grouped.values() for value in values
        )
        outcome_groups = {
            "both_correct": {
                key for key in grouped if fp[key]["correct"] and quant[key]["correct"]
            },
            "fp_only": {
                key for key in grouped if fp[key]["correct"] and not quant[key]["correct"]
            },
            "quant_only": {
                key for key in grouped if not fp[key]["correct"] and quant[key]["correct"]
            },
            "both_wrong": {
                key for key in grouped if not fp[key]["correct"] and not quant[key]["correct"]
            },
        }
        group_sampling = {}
        for label, example_id_set in outcome_groups.items():
            example_ids = sorted(example_id_set)
            successes = [
                sum(bool(value["correct"]) for value in grouped[example_id])
                for example_id in example_ids
            ]
            completions = sum(len(grouped[example_id]) for example_id in example_ids)
            group_sampling[label] = {
                "num_examples": len(example_ids),
                "per_completion_accuracy": sum(successes) / completions if completions else None,
                "any_sample_accuracy": sum(value >= 1 for value in successes) / len(successes)
                if successes
                else None,
                "majority_vote_accuracy": sum(
                    value > len(grouped[example_id]) / 2
                    for value, example_id in zip(successes, example_ids)
                )
                / len(successes)
                if successes
                else None,
                "all_samples_correct_rate": sum(
                    value == len(grouped[example_id])
                    for value, example_id in zip(successes, example_ids)
                )
                / len(successes)
                if successes
                else None,
                "success_count_histogram": {
                    str(count): successes.count(count) for count in sorted(set(successes))
                },
            }
        report["temperatures"][str(temperature)] = {
            "num_examples": len(grouped),
            "num_completions": sum(len(values) for values in grouped.values()),
            "samples_per_example": sorted({len(values) for values in grouped.values()}),
            "per_completion_accuracy": statistics.fmean(
                bool(value["correct"]) for values in grouped.values() for value in values
            ),
            "pass_at_k_empirical": len(any_success) / len(grouped),
            "majority_vote_accuracy": len(majority_success) / len(grouped),
            "mean_generation_tokens": mean_tokens,
            "token_delta_vs_fp_greedy": mean_tokens
            - report["paired_subset"]["fp_mean_generation_tokens"],
            "token_delta_vs_quant_greedy": mean_tokens
            - report["paired_subset"]["quant_mean_generation_tokens"],
            "mean_distinct_answers_per_prompt": statistics.fmean(answer_diversity),
            "any_sample_rescue_count": len(temp_rescue),
            "majority_rescue_count": len(majority_rescue),
            "quant_rescues_reproduced_by_any_sample": len(quant_rescue & temp_rescue),
            "quant_rescues_reproduced_by_majority": len(quant_rescue & majority_rescue),
            "quant_rescue_coverage": (
                len(quant_rescue & temp_rescue) / len(quant_rescue) if quant_rescue else None
            ),
            "quant_rescue_majority_coverage": (
                len(quant_rescue & majority_rescue) / len(quant_rescue)
                if quant_rescue
                else None
            ),
            "quant_vs_temperature_rescue_jaccard": _jaccard(quant_rescue, temp_rescue),
            "quant_vs_majority_rescue_jaccard": _jaccard(quant_rescue, majority_rescue),
            "quant_induced_failures_with_any_sample_failure": sum(
                any(not value["correct"] for value in grouped[example_id])
                for example_id in quant_induced_failure
            ),
            "quant_induced_failures_reproduced_by_majority": len(
                quant_induced_failure & majority_failure
            ),
            "quant_induced_failures_reproduced_by_all_samples_wrong": len(
                quant_induced_failure & all_sample_failure
            ),
            "sampling_by_greedy_outcome": group_sampling,
        }
    if fp_tokens is not None and quant_tokens is not None and sampled_tokens is not None:
        report["controlled_correctness_model"] = fit_controlled_correctness_model(
            fp_outputs,
            quant_outputs,
            sampled_outputs,
            fp_tokens,
            quant_tokens,
            sampled_tokens,
        )
    return report


def _group_greedy_tokens(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["example_id"])].append(row)
    return grouped


def _group_sampled_tokens(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, float, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["example_id"]), float(row["temperature"]), int(row["sample_id"]))].append(row)
    return grouped


def fit_controlled_correctness_model(
    fp_outputs: list[dict[str, Any]],
    quant_outputs: list[dict[str, Any]],
    sampled_outputs: list[dict[str, Any]],
    fp_tokens: list[dict[str, Any]],
    quant_tokens: list[dict[str, Any]],
    sampled_tokens: list[dict[str, Any]],
) -> dict[str, Any]:
    """Grouped CV test of whether precision mode adds signal beyond trajectory stats.

    This is associational, not causal. Grouped folds keep every completion for one
    prompt in the same fold, preventing prompt memorization across conditions.
    """
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    fp_grouped = _group_greedy_tokens(fp_tokens)
    quant_grouped = _group_greedy_tokens(quant_tokens)
    sampled_grouped = _group_sampled_tokens(sampled_tokens)
    rows = []

    def add(output: dict[str, Any], precision: str, temperature: float, sampled: int, tokens: list[dict[str, Any]]) -> None:
        trace = aggregate_token_trace(tokens)
        rows.append(
            {
                "example_id": output["example_id"],
                "correct": int(bool(output["correct"])),
                "precision": precision,
                "temperature": temperature,
                "sampled": sampled,
                "prompt_tokens": output.get("prompt_tokens"),
                "generation_tokens": output.get("generation_tokens"),
                "mean_entropy": trace.get("mean_entropy"),
                "max_entropy": trace.get("max_entropy"),
                "mean_logit_margin": trace.get("mean_logit_margin"),
                "min_logit_margin": trace.get("min_logit_margin"),
            }
        )

    for output in fp_outputs:
        add(output, "fp16", 0.0, 0, fp_grouped.get(output["example_id"], []))
    for output in quant_outputs:
        add(output, "bnb4", 0.0, 0, quant_grouped.get(output["example_id"], []))
    for output in sampled_outputs:
        key = (output["example_id"], float(output["temperature"]), int(output["sample_id"]))
        add(output, "fp16", float(output["temperature"]), 1, sampled_grouped.get(key, []))

    frame = pd.DataFrame(rows)
    numeric = [
        "temperature",
        "sampled",
        "prompt_tokens",
        "generation_tokens",
        "mean_entropy",
        "max_entropy",
        "mean_logit_margin",
        "min_logit_margin",
    ]

    def pipeline(include_precision: bool) -> Pipeline:
        transformers = [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]),
                numeric,
            )
        ]
        if include_precision:
            transformers.append(
                ("precision", OneHotEncoder(drop="first", handle_unknown="ignore"), ["precision"])
            )
        return Pipeline(
            [
                ("features", ColumnTransformer(transformers)),
                ("classifier", LogisticRegression(max_iter=3000, random_state=42)),
            ]
        )

    y = frame["correct"].to_numpy(dtype=int)
    groups = frame["example_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    result = {}
    for label, include_precision in (("trajectory_only", False), ("plus_precision_mode", True)):
        model = pipeline(include_precision)
        probability = cross_val_predict(
            model, frame, y, groups=groups, cv=splitter, method="predict_proba"
        )[:, 1]
        model.fit(frame, y)
        result[label] = {
            "grouped_cv_roc_auc": float(roc_auc_score(y, probability)),
            "grouped_cv_log_loss": float(log_loss(y, probability)),
        }
        if include_precision:
            names = model.named_steps["features"].get_feature_names_out()
            coefficients = model.named_steps["classifier"].coef_[0]
            result[label]["fitted_coefficients"] = {
                str(name): float(value) for name, value in zip(names, coefficients)
            }
    result["delta_when_adding_precision"] = {
        "roc_auc": result["plus_precision_mode"]["grouped_cv_roc_auc"]
        - result["trajectory_only"]["grouped_cv_roc_auc"],
        "log_loss": result["plus_precision_mode"]["grouped_cv_log_loss"]
        - result["trajectory_only"]["grouped_cv_log_loss"],
    }
    result["num_rows"] = len(frame)
    result["num_prompts"] = int(frame["example_id"].nunique())
    result["warning"] = "Associational grouped-CV diagnostic; precision is not randomized independently of model weights."
    return result
