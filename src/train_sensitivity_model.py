"""Train cheap example-level predictors for FP-correct/quantized-wrong failures."""

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


def _score_report(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    report = binary_metrics(y_true, scores)
    report["recall_at_5pct"] = recall_at_budget(y_true, scores, 0.05)
    report["recall_at_10pct"] = recall_at_budget(y_true, scores, 0.10)
    report["recall_at_20pct"] = recall_at_budget(y_true, scores, 0.20)
    report["precision_at_5pct"] = precision_at_budget(y_true, scores, 0.05)
    report["precision_at_10pct"] = precision_at_budget(y_true, scores, 0.10)
    report["precision_at_20pct"] = precision_at_budget(y_true, scores, 0.20)
    return report


def train_predictors(
    feature_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    test_size: float = 0.3,
) -> dict[str, Any]:
    frame = pd.read_parquet(feature_path)
    frame = frame[frame["eligible_fp_correct"] == 1].copy()
    y = frame["target_quantization_failure"].astype(int)
    feature_names = [name for name in DEFAULT_FEATURES if name in frame.columns]
    base = {
        "num_eligible_examples": int(len(frame)),
        "num_positive_failures": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "features": feature_names,
        "seed": seed,
    }
    if len(frame) < 8 or y.nunique() < 2 or y.value_counts().min() < 2:
        return {
            **base,
            "status": "insufficient_class_variation",
            "message": "Collect more FP-correct examples and at least two quantization-induced failures.",
        }

    indices = np.arange(len(frame))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=y.to_numpy()
    )
    x = frame[feature_names]
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx].to_numpy(), y.iloc[test_idx].to_numpy()

    preprocessing = ColumnTransformer(
        [("numeric", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), feature_names)]
    )
    models = {
        "logistic_regression": Pipeline(
            [("features", preprocessing), ("classifier", LogisticRegression(class_weight="balanced", random_state=seed))]
        ),
        "random_forest": Pipeline(
            [
                ("impute", SimpleImputer()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300, class_weight="balanced", min_samples_leaf=2, random_state=seed
                    ),
                ),
            ]
        ),
    }
    output_dir = Path(output_dir)
    results: dict[str, Any] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_test)[:, 1]
        results[name] = _score_report(y_test, scores)
        target = ensure_parent(output_dir / f"{name}.joblib")
        joblib.dump({"model": model, "features": feature_names}, target)

    rng = np.random.default_rng(seed)
    baselines = {
        "entropy_threshold": x_test["max_entropy"].to_numpy(),
        "logit_margin_threshold": -x_test["min_logit_margin"].to_numpy(),
        "random_same_rate": rng.random(len(x_test)),
    }
    for name, scores in baselines.items():
        results[name] = _score_report(y_test, np.nan_to_num(scores, nan=0.0))
    return {
        **base,
        "status": "ok",
        "split": "stratified_random",
        "num_train": int(len(train_idx)),
        "num_test": int(len(test_idx)),
        "results": results,
    }
