"""Train cheap failure predictors and evaluate their net precision-intervention utility."""

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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import binary_metrics, precision_at_budget, recall_at_budget
from .utils import ensure_parent


DEFAULT_FEATURES = [
    "max_entropy",
    "mean_entropy",
    "min_logit_margin",
    "mean_logit_margin",
    "min_token_probability",
    "mean_token_probability",
    "prompt_tokens",
    "generation_tokens",
]

BUDGETS = [0.0, 0.05, 0.10, 0.20, 0.50, 1.0]


def _score_report(y_true: np.ndarray, scores: np.ndarray, seed: int = 42) -> dict[str, Any]:
    report = binary_metrics(y_true, scores)
    report.update(_bootstrap_intervals(y_true, scores, seed))
    for budget in (0.05, 0.10, 0.20):
        percent = int(100 * budget)
        report[f"recall_at_{percent}pct"] = recall_at_budget(y_true, scores, budget)
        report[f"precision_at_{percent}pct"] = precision_at_budget(y_true, scores, budget)
    return report


def _bootstrap_intervals(
    y_true: np.ndarray, scores: np.ndarray, seed: int, samples: int = 500
) -> dict[str, Any]:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"roc_auc_ci95": None, "pr_auc_ci95": None, "bootstrap_samples": 0}
    rng = np.random.default_rng(seed)
    roc_values, pr_values = [], []
    for _ in range(samples):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        sampled_y = y_true[indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        sampled_scores = scores[indices]
        roc_values.append(roc_auc_score(sampled_y, sampled_scores))
        pr_values.append(average_precision_score(sampled_y, sampled_scores))
    if not roc_values:
        return {"roc_auc_ci95": None, "pr_auc_ci95": None, "bootstrap_samples": 0}
    return {
        "roc_auc_ci95": [float(value) for value in np.quantile(roc_values, [0.025, 0.975])],
        "pr_auc_ci95": [float(value) for value in np.quantile(pr_values, [0.025, 0.975])],
        "bootstrap_samples": len(roc_values),
    }


def _models(feature_names: list[str], seed: int) -> dict[str, Pipeline]:
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]),
                feature_names,
            )
        ]
    )
    return {
        "logistic_regression": Pipeline(
            [
                ("features", preprocessing),
                ("classifier", LogisticRegression(class_weight="balanced", random_state=seed)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("impute", SimpleImputer()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def _correctness_columns(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    labels = frame["comparison_label"].astype(str)
    fp_correct = labels.isin(["fp_correct_q_correct", "fp_correct_q_wrong"]).to_numpy(dtype=int)
    quant_correct = labels.isin(["fp_correct_q_correct", "fp_wrong_q_correct"]).to_numpy(dtype=int)
    return fp_correct, quant_correct


def _intervention_curve(frame: pd.DataFrame, scores: np.ndarray) -> list[dict[str, Any]]:
    fp_correct, quant_correct = _correctness_columns(frame)
    n = len(frame)
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))
    curve = []
    for budget in BUDGETS:
        count = min(n, int(round(n * budget)))
        selected = np.zeros(n, dtype=bool)
        selected[order[:count]] = True
        correct = np.where(selected, fp_correct, quant_correct)
        curve.append(
            {
                "budget": budget,
                "num_interventions": count,
                "accuracy": float(correct.mean()) if n else 0.0,
                "beneficial_selected": int(((fp_correct > quant_correct) & selected).sum()),
                "harmful_selected": int(((fp_correct < quant_correct) & selected).sum()),
            }
        )
    return curve


def _quality_filtered(frame: pd.DataFrame, quality_filter: str) -> pd.DataFrame:
    if quality_filter == "none":
        return frame
    required = {
        "fp_has_explicit_answer",
        "quant_has_explicit_answer",
        "fp_hit_max_new_tokens",
        "quant_hit_max_new_tokens",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"quality_filter={quality_filter!r} requires regenerated feature columns: {missing}"
        )
    valid = frame["fp_has_explicit_answer"].fillna(False).astype(bool) & frame[
        "quant_has_explicit_answer"
    ].fillna(False).astype(bool)
    if quality_filter == "clean":
        valid &= ~frame["fp_hit_max_new_tokens"].fillna(True).astype(bool)
        valid &= ~frame["quant_hit_max_new_tokens"].fillna(True).astype(bool)
    elif quality_filter != "both_hash":
        raise ValueError("quality_filter must be one of: none, both_hash, clean")
    return frame[valid].copy()


def train_predictors(
    feature_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    test_size: float = 0.3,
    test_feature_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
    quality_filter: str = "none",
) -> dict[str, Any]:
    train_all = _quality_filtered(pd.read_parquet(feature_path), quality_filter)
    train_eligible = train_all[train_all["eligible_fp_correct"] == 1].copy()
    y_all = train_eligible["target_quantization_failure"].astype(int)
    feature_names = [name for name in DEFAULT_FEATURES if name in train_eligible.columns]
    base = {
        "num_eligible_examples": int(len(train_eligible)),
        "num_positive_failures": int(y_all.sum()),
        "positive_rate": float(y_all.mean()) if len(y_all) else 0.0,
        "features": feature_names,
        "seed": seed,
        "quality_filter": quality_filter,
    }
    if len(train_eligible) < 8 or y_all.nunique() < 2 or y_all.value_counts().min() < 2:
        return {
            **base,
            "status": "insufficient_class_variation",
            "message": "Collect more FP-correct examples and at least two quantization-induced failures.",
        }

    held_out = test_feature_path is not None
    if held_out:
        train_frame = train_eligible
        eval_all = _quality_filtered(pd.read_parquet(test_feature_path).copy(), quality_filter)
        eval_frame = eval_all[eval_all["eligible_fp_correct"] == 1].copy()
        split_name = "held_out_file"
    else:
        train_indices, test_indices = train_test_split(
            np.arange(len(train_eligible)),
            test_size=test_size,
            random_state=seed,
            stratify=y_all.to_numpy(),
        )
        train_frame = train_eligible.iloc[train_indices].copy()
        eval_frame = train_eligible.iloc[test_indices].copy()
        eval_all = eval_frame.copy()
        split_name = "stratified_random"

    y_train = train_frame["target_quantization_failure"].astype(int).to_numpy()
    y_eval = eval_frame["target_quantization_failure"].astype(int).to_numpy()
    output_dir = Path(output_dir)
    results: dict[str, Any] = {}
    prediction_frame = eval_all.copy()
    all_scores: dict[str, np.ndarray] = {}

    for name, model in _models(feature_names, seed).items():
        model.fit(train_frame[feature_names], y_train)
        eval_scores = model.predict_proba(eval_frame[feature_names])[:, 1]
        scores = model.predict_proba(eval_all[feature_names])[:, 1]
        results[name] = _score_report(y_eval, eval_scores, seed)
        all_scores[name] = scores
        prediction_frame[f"score_{name}"] = scores
        joblib.dump(
            {"model": model, "features": feature_names},
            ensure_parent(output_dir / f"{name}.joblib"),
        )

    rng = np.random.default_rng(seed)
    baseline_scores = {
        "entropy_threshold": eval_all["max_entropy"].to_numpy(dtype=float),
        "logit_margin_threshold": -eval_all["min_logit_margin"].to_numpy(dtype=float),
        "random_same_rate": rng.random(len(eval_all)),
    }
    eligible_mask = eval_all["eligible_fp_correct"].to_numpy(dtype=bool)
    for name, scores in baseline_scores.items():
        clean_scores = np.nan_to_num(scores, nan=0.0)
        results[name] = _score_report(y_eval, clean_scores[eligible_mask], seed)
        all_scores[name] = clean_scores
        prediction_frame[f"score_{name}"] = clean_scores

    intervention_curves = None
    if held_out:
        intervention_curves = {
            name: _intervention_curve(eval_all, scores) for name, scores in all_scores.items()
        }
        fp_correct, quant_correct = _correctness_columns(eval_all)
        intervention_curves["oracle_at_most_budget"] = _oracle_curve(fp_correct, quant_correct)

    if predictions_path:
        target = ensure_parent(predictions_path)
        prediction_frame.to_parquet(target, index=False)

    report = {
        **base,
        "status": "ok",
        "split": split_name,
        "num_train": int(len(train_frame)),
        "num_test": int(len(eval_frame)),
        "test_num_positive_failures": int(y_eval.sum()),
        "test_positive_rate": float(y_eval.mean()) if len(y_eval) else 0.0,
        "results": results,
    }
    if intervention_curves is not None:
        report["controller_evaluation_examples"] = int(len(eval_all))
        report["intervention_curves"] = intervention_curves
    return report


def _oracle_curve(fp_correct: np.ndarray, quant_correct: np.ndarray) -> list[dict[str, Any]]:
    n = len(fp_correct)
    beneficial = np.flatnonzero(fp_correct > quant_correct)
    curve = []
    for budget in BUDGETS:
        allowance = min(n, int(round(n * budget)))
        count = min(allowance, len(beneficial))
        selected = np.zeros(n, dtype=bool)
        selected[beneficial[:count]] = True
        correct = np.where(selected, fp_correct, quant_correct)
        curve.append(
            {
                "budget": budget,
                "allowed_interventions": allowance,
                "num_interventions": count,
                "accuracy": float(correct.mean()) if n else 0.0,
                "beneficial_selected": count,
                "harmful_selected": 0,
            }
        )
    return curve
