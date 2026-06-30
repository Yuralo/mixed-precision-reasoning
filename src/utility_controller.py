"""Counterfactual precision-routing models over beneficial/harmful/neutral outcomes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .paired_analysis import flatten_record
from .utils import ensure_parent


OUTCOME_UTILITY = {
    "fp_correct_q_correct": 0.0,
    "fp_correct_q_wrong": 1.0,
    "fp_wrong_q_correct": -1.0,
    "fp_wrong_q_wrong": 0.0,
}
BUDGETS = [0.0, 0.05, 0.10, 0.20, 0.50, 1.0]

FEATURE_SETS = {
    "prompt_only": ["prompt_tokens"],
    "length_only": ["quant_generated_tokens"],
    "uncertainty_only": [
        "quant_runtime_max_entropy",
        "quant_runtime_mean_entropy",
        "quant_runtime_last_entropy",
        "quant_runtime_entropy_slope",
        "quant_runtime_early_entropy",
        "quant_runtime_late_entropy",
        "quant_runtime_entropy_late_minus_early",
        "quant_runtime_min_logit_margin",
        "quant_runtime_mean_logit_margin",
        "quant_runtime_last_logit_margin",
        "quant_runtime_min_token_probability",
        "quant_runtime_mean_token_probability",
        "quant_runtime_last_token_probability",
    ],
    "structure_only": [
        "quant_line_count",
        "quant_sentence_count",
        "quant_numbered_step_count",
        "quant_arithmetic_expression_count",
        "quant_equation_count",
        "quant_numeric_value_count",
        "quant_distinct_numeric_value_count",
        "quant_self_correction_count",
        "quant_repetition_score_4gram",
        "quant_normalized_final_answer_position",
        "quant_final_answer_seen_earlier_count",
        "quant_terminal_line_has_arithmetic_expression",
    ],
}
FEATURE_SETS["all_quant"] = (
    FEATURE_SETS["prompt_only"]
    + FEATURE_SETS["length_only"]
    + FEATURE_SETS["uncertainty_only"]
    + FEATURE_SETS["structure_only"]
)


def records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([flatten_record(record) for record in records])


def build_models(features: list[str], seed: int) -> dict[str, Pipeline]:
    preprocessing = ColumnTransformer(
        [("numeric", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), features)]
    )
    return {
        "logistic": Pipeline(
            [
                ("features", preprocessing),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced", max_iter=3000, random_state=seed
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("impute", SimpleImputer()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=500,
                        class_weight="balanced",
                        min_samples_leaf=3,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def utility_scores(model: Pipeline, probabilities: np.ndarray) -> np.ndarray:
    classes = model.classes_
    utilities = np.array([OUTCOME_UTILITY[str(value)] for value in classes])
    return probabilities @ utilities


def class_probability(model: Pipeline, probabilities: np.ndarray, outcome: str) -> np.ndarray:
    classes = list(model.classes_)
    return probabilities[:, classes.index(outcome)]


def intervention_curve(
    frame: pd.DataFrame, scores: np.ndarray, positive_only: bool = False
) -> list[dict[str, Any]]:
    fp_correct = frame["fp_correct"].to_numpy(dtype=int)
    quant_correct = frame["quant_correct"].to_numpy(dtype=int)
    beneficial = frame["outcome"].eq("fp_correct_q_wrong").to_numpy()
    harmful = frame["outcome"].eq("fp_wrong_q_correct").to_numpy()
    order = np.argsort(-scores)
    n = len(frame)
    total_beneficial = max(1, int(beneficial.sum()))
    total_harmful = max(1, int(harmful.sum()))
    curve = []
    for budget in BUDGETS:
        allowed = min(n, int(round(n * budget)))
        chosen = order[:allowed]
        if positive_only:
            chosen = chosen[scores[chosen] > 0]
        count = len(chosen)
        selected = np.zeros(n, dtype=bool)
        selected[chosen] = True
        beneficial_selected = int((beneficial & selected).sum())
        harmful_selected = int((harmful & selected).sum())
        correct = np.where(selected, fp_correct, quant_correct)
        curve.append(
            {
                "budget": budget,
                "allowed_interventions": allowed,
                "num_interventions": count,
                "accuracy": float(correct.mean()),
                "net_correctness_gain": beneficial_selected - harmful_selected,
                "beneficial_selected": beneficial_selected,
                "harmful_selected": harmful_selected,
                "beneficial_switch_precision": beneficial_selected / count if count else None,
                "beneficial_switch_recall": beneficial_selected / total_beneficial,
                "harmful_switch_rate": harmful_selected / count if count else None,
                "avoided_harm_rate": 1 - harmful_selected / total_harmful,
            }
        )
    return curve


def expected_random_curve(frame: pd.DataFrame) -> list[dict[str, Any]]:
    beneficial_rate = frame["outcome"].eq("fp_correct_q_wrong").mean()
    harmful_rate = frame["outcome"].eq("fp_wrong_q_correct").mean()
    quant_accuracy = frame["quant_correct"].mean()
    total_beneficial = int(frame["outcome"].eq("fp_correct_q_wrong").sum())
    total_harmful = int(frame["outcome"].eq("fp_wrong_q_correct").sum())
    curve = []
    for budget in BUDGETS:
        count = int(round(len(frame) * budget))
        beneficial = count * beneficial_rate
        harmful = count * harmful_rate
        curve.append(
            {
                "budget": budget,
                "num_interventions": count,
                "accuracy": float(quant_accuracy + (beneficial - harmful) / len(frame)),
                "net_correctness_gain": beneficial - harmful,
                "beneficial_selected": beneficial,
                "harmful_selected": harmful,
                "beneficial_switch_precision": beneficial_rate if count else None,
                "beneficial_switch_recall": count / len(frame) if total_beneficial else 0,
                "harmful_switch_rate": harmful_rate if count else None,
                "avoided_harm_rate": 1 - count / len(frame) if total_harmful else 1,
            }
        )
    return curve


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lower) & (
            probabilities < upper if upper < 1 else probabilities <= upper
        )
        if mask.any():
            error += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(error)


def selective_risk_curve(
    frame: pd.DataFrame, model: Pipeline, probabilities: np.ndarray, scores: np.ndarray, budget: float = 0.10
) -> list[dict[str, float]]:
    n = len(frame)
    count = int(round(n * budget))
    selected = np.zeros(n, dtype=bool)
    selected[np.argsort(-scores)[:count]] = True
    classes = list(model.classes_)
    both = probabilities[:, classes.index("fp_correct_q_correct")]
    fp_only = probabilities[:, classes.index("fp_correct_q_wrong")]
    q_only = probabilities[:, classes.index("fp_wrong_q_correct")]
    predicted_correctness = np.where(selected, both + fp_only, both + q_only)
    actual_correctness = np.where(
        selected,
        frame["fp_correct"].to_numpy(dtype=int),
        frame["quant_correct"].to_numpy(dtype=int),
    )
    order = np.argsort(-predicted_correctness)
    curve = []
    for coverage in (0.50, 0.60, 0.70, 0.80, 0.90, 1.0):
        answered = order[: max(1, int(round(n * coverage)))]
        accuracy = float(actual_correctness[answered].mean())
        curve.append(
            {
                "coverage": coverage,
                "selective_accuracy": accuracy,
                "selective_risk": 1 - accuracy,
                "abstention_rate": 1 - coverage,
            }
        )
    return curve


def evaluate_controller(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    output_dir: str | Path,
    seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train, test = records_frame(train_records), records_frame(test_records)
    y_train, y_test = train["outcome"], test["outcome"]
    results: dict[str, Any] = {}
    predictions = test[["example_id", "outcome", "fp_correct", "quant_correct", "clean"]].copy()
    best_key, best_accuracy = None, -1.0
    best_payload = None
    output_dir = Path(output_dir)

    for feature_set, configured_features in FEATURE_SETS.items():
        features = [name for name in configured_features if name in train.columns]
        for model_name, model in build_models(features, seed).items():
            model.fit(train[features], y_train)
            probabilities = model.predict_proba(test[features])
            scores = utility_scores(model, probabilities)
            beneficial_probability = class_probability(
                model, probabilities, "fp_correct_q_wrong"
            )
            harmful_probability = class_probability(model, probabilities, "fp_wrong_q_correct")
            beneficial_true = test["outcome"].eq("fp_correct_q_wrong").to_numpy(dtype=int)
            harmful_true = test["outcome"].eq("fp_wrong_q_correct").to_numpy(dtype=int)
            predicted_outcome = model.predict(test[features])
            curves = intervention_curve(test, scores, positive_only=True)
            key = f"{feature_set}__{model_name}"
            results[key] = {
                "feature_set": feature_set,
                "model": model_name,
                "features": features,
                "outcome_accuracy": float((predicted_outcome == y_test.to_numpy()).mean()),
                "beneficial_roc_auc": float(roc_auc_score(beneficial_true, beneficial_probability)),
                "beneficial_pr_auc": float(average_precision_score(beneficial_true, beneficial_probability)),
                "harmful_roc_auc": float(roc_auc_score(harmful_true, harmful_probability)),
                "harmful_pr_auc": float(average_precision_score(harmful_true, harmful_probability)),
                "beneficial_brier": float(np.mean((beneficial_probability - beneficial_true) ** 2)),
                "beneficial_ece": expected_calibration_error(beneficial_true, beneficial_probability),
                "intervention_curve": curves,
                "selective_risk_curve_at_10pct": selective_risk_curve(
                    test, model, probabilities, scores, 0.10
                ),
            }
            accuracy_at_10 = next(point["accuracy"] for point in curves if point["budget"] == 0.10)
            if accuracy_at_10 > best_accuracy:
                best_key, best_accuracy = key, accuracy_at_10
                best_payload = (model, probabilities, scores)
            predictions[f"score_{key}"] = scores
            predictions[f"p_beneficial_{key}"] = beneficial_probability
            predictions[f"p_harmful_{key}"] = harmful_probability
            joblib.dump(
                {"model": model, "features": features, "feature_set": feature_set},
                ensure_parent(output_dir / "models" / f"{key}.joblib"),
            )

    baselines = {
        "entropy": test["quant_runtime_max_entropy"].to_numpy(dtype=float),
        "length": test["quant_generated_tokens"].to_numpy(dtype=float),
        "margin": -test["quant_runtime_min_logit_margin"].to_numpy(dtype=float),
    }
    baseline_curves = {name: intervention_curve(test, scores) for name, scores in baselines.items()}
    baseline_curves["random_expected"] = expected_random_curve(test)
    utility = np.array([OUTCOME_UTILITY[value] for value in test["outcome"]])
    baseline_curves["oracle"] = intervention_curve(test, utility, positive_only=True)

    best_model, best_probabilities, best_scores = best_payload
    report = {
        "train_examples": len(train),
        "test_examples": len(test),
        "train_outcomes": train["outcome"].value_counts().to_dict(),
        "test_outcomes": test["outcome"].value_counts().to_dict(),
        "best_learned_at_10pct": best_key,
        "best_learned_accuracy_at_10pct": best_accuracy,
        "models": results,
        "baselines": baseline_curves,
        "static": {
            "always_quantized": float(test["quant_correct"].mean()),
            "always_fp16": float(test["fp_correct"].mean()),
        },
    }
    return report, predictions
