"""Evaluation metrics for accuracy, flips, budgets, and binary prediction."""

from __future__ import annotations

from typing import Any


def comparison_summary(rows: list[dict]) -> dict:
    n = len(rows)
    fp_correct = sum(bool(row["fp_correct"]) for row in rows)
    q_correct = sum(bool(row["quant_correct"]) for row in rows)
    induced = sum(row["comparison_label"] == "fp_correct_q_wrong" for row in rows)
    rescued = sum(row["comparison_label"] == "fp_wrong_q_correct" for row in rows)
    flips = sum(row["fp_answer"] != row["quant_answer"] for row in rows)
    return {
        "num_examples": n,
        "fp_accuracy": fp_correct / n if n else 0.0,
        "quant_accuracy": q_correct / n if n else 0.0,
        "fp_correct_quant_wrong": induced,
        "fp_wrong_quant_correct": rescued,
        "answer_flip_rate": flips / n if n else 0.0,
        "critical_failure_rate": induced / fp_correct if fp_correct else 0.0,
        "label_counts": {
            label: sum(row["comparison_label"] == label for row in rows)
            for label in (
                "fp_correct_q_correct",
                "fp_correct_q_wrong",
                "fp_wrong_q_correct",
                "fp_wrong_q_wrong",
            )
        },
    }


def binary_metrics(y_true: Any, scores: Any) -> dict:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(np.unique(y_true)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }


def recall_at_budget(y_true: Any, scores: Any, budget: float) -> float:
    import numpy as np

    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    count = max(1, int(np.ceil(len(scores) * budget)))
    selected = np.argsort(-scores)[:count]
    return float(y_true[selected].sum() / positives)


def precision_at_budget(y_true: Any, scores: Any, budget: float) -> float:
    import numpy as np

    count = max(1, int(np.ceil(len(scores) * budget)))
    selected = np.argsort(-scores)[:count]
    return float(y_true[selected].mean())
