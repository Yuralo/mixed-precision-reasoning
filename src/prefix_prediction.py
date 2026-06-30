"""Test whether quantized-prefix telemetry predicts the value of an FP rerun.

The labels are counterfactual: +1 means FP is correct and quantized is wrong,
-1 means the reverse, and 0 means switching precision does not change accuracy.
Only information observed from the quantized trajectory prefix is used.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .paired_analysis import aggregate_token_trace, flatten_record
from .trajectory_metrics import analyze_trajectory
from .utility_controller import (
    build_models,
    class_probability,
    intervention_curve,
    utility_scores,
)
from .utils import ensure_parent


PREFIX_BUDGETS = (16, 32, 64)


def group_tokens(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["example_id"])].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["token_position"]))
    return grouped


def _prefix_row(record: dict[str, Any], tokens: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    observed = tokens[:budget]
    text = "".join(str(row["token"]) for row in observed)
    trace = aggregate_token_trace(observed)
    structure = analyze_trajectory(text, len(observed))
    row = {
        "example_id": record["example_id"],
        "outcome": record["outcome"],
        "fp_correct": record["fp_correct"],
        "quant_correct": record["quant_correct"],
        "clean": record["clean"],
        "prompt_tokens": record["prompt_tokens"],
        "prefix_budget": budget,
        "observed_tokens": len(observed),
        "terminated_by_budget": int(len(tokens) <= budget),
    }
    row.update({f"trace_{key}": value for key, value in trace.items()})
    keep_structure = (
        "line_count",
        "sentence_count",
        "numbered_step_count",
        "arithmetic_expression_count",
        "equation_count",
        "numeric_value_count",
        "distinct_numeric_value_count",
        "self_correction_count",
        "repetition_score_4gram",
        "has_explicit_answer",
        "normalized_final_answer_position",
    )
    row.update({f"structure_{key}": structure[key] for key in keep_structure})
    return row


def build_prefix_frame(
    records: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
    budget: int,
) -> pd.DataFrame:
    grouped = group_tokens(token_rows)
    return pd.DataFrame(
        [_prefix_row(record, grouped.get(record["example_id"], []), budget) for record in records]
    )


def _features(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "example_id",
        "outcome",
        "fp_correct",
        "quant_correct",
        "clean",
        "prefix_budget",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded and frame[column].notna().any()
    ]


def evaluate_prefixes(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    train_tokens: list[dict[str, Any]],
    test_tokens: list[dict[str, Any]],
    output_dir: str | Path,
    budgets: tuple[int, ...] = PREFIX_BUDGETS,
    seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    output_dir = Path(output_dir)
    report: dict[str, Any] = {"budgets": list(budgets), "results": {}}
    prediction_frames = []

    for budget in budgets:
        train = build_prefix_frame(train_records, train_tokens, budget)
        test = build_prefix_frame(test_records, test_tokens, budget)
        features = _features(train)
        y_train = train["outcome"]
        beneficial_true = test["outcome"].eq("fp_correct_q_wrong").to_numpy(dtype=int)
        budget_results = {}
        for model_name, model in build_models(features, seed).items():
            model.fit(train[features], y_train)
            probabilities = model.predict_proba(test[features])
            scores = utility_scores(model, probabilities)
            beneficial_probability = class_probability(
                model, probabilities, "fp_correct_q_wrong"
            )
            curves = intervention_curve(test, scores, positive_only=True)
            budget_results[model_name] = {
                "features": features,
                "beneficial_roc_auc": float(
                    roc_auc_score(beneficial_true, beneficial_probability)
                ),
                "beneficial_pr_auc": float(
                    average_precision_score(beneficial_true, beneficial_probability)
                ),
                "intervention_curve": curves,
            }
            predictions = test[
                ["example_id", "outcome", "fp_correct", "quant_correct", "clean"]
            ].copy()
            predictions["prefix_budget"] = budget
            predictions["model"] = model_name
            predictions["utility_score"] = scores
            predictions["p_beneficial"] = beneficial_probability
            prediction_frames.append(predictions)
        report["results"][str(budget)] = budget_results

    prediction_frame = pd.concat(prediction_frames, ignore_index=True)
    prediction_frame.to_csv(ensure_parent(output_dir / "predictions.csv"), index=False)
    return report, prediction_frame
