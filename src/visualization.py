"""Reproducible, publication-friendly figures for the mixed-precision experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_parent, read_json, read_jsonl, write_json


COLORS = {
    "fp": "#355C7D",
    "quant": "#F67280",
    "good": "#2A9D8F",
    "bad": "#E76F51",
    "neutral": "#8D99AE",
    "oracle": "#6A4C93",
}


def _save(fig: plt.Figure, output_dir: Path, name: str, formats: list[str]) -> list[str]:
    paths = []
    for extension in formats:
        path = output_dir / f"{name}.{extension}"
        ensure_parent(path)
        fig.savefig(path, dpi=180 if extension == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def make_figures(
    comparison_path: str | Path,
    example_features_path: str | Path,
    predictor_metrics_path: str | Path | None,
    oracle_path: str | Path | None,
    diagnostics_path: str | Path | None,
    output_dir: str | Path,
    formats: list[str],
    audit_path: str | Path | None = None,
) -> dict[str, Any]:
    plt.style.use("seaborn-v0_8-whitegrid")
    output_dir = Path(output_dir)
    comparisons = pd.DataFrame(read_jsonl(comparison_path))
    features = pd.read_parquet(example_features_path)
    manifest: dict[str, list[str]] = {}
    manifest["accuracy_and_outcomes"] = _overview(comparisons, output_dir, formats)
    manifest["uncertainty_distributions"] = _feature_distributions(features, output_dir, formats)
    manifest["failure_probability"] = _failure_probability(features, output_dir, formats)

    if predictor_metrics_path and Path(predictor_metrics_path).exists():
        metrics = read_json(predictor_metrics_path)
        if metrics.get("status") == "ok":
            manifest["predictor_metrics"] = _predictor_metrics(metrics, output_dir, formats)
            if metrics.get("intervention_curves"):
                manifest["controller_accuracy"] = _controller_curves(metrics, output_dir, formats)
    if oracle_path and Path(oracle_path).exists():
        manifest["oracle_recovery"] = _oracle_figure(read_json(oracle_path), output_dir, formats)
    if diagnostics_path and Path(diagnostics_path).exists():
        manifest["generation_quality"] = _diagnostics_figure(
            read_json(diagnostics_path), output_dir, formats
        )
    if audit_path and Path(audit_path).exists():
        audit = read_json(audit_path)
        manifest["runtime_tradeoff"] = _runtime_figure(audit, output_dir, formats)
        manifest["split_accuracy"] = _split_accuracy_figure(audit, output_dir, formats)
        manifest["univariate_features"] = _univariate_auc_figure(audit, output_dir, formats)
    write_json(output_dir / "manifest.json", {"figures": manifest})
    return {"figures": manifest}


def _overview(frame: pd.DataFrame, output_dir: Path, formats: list[str]) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    accuracies = [frame["fp_correct"].mean(), frame["quant_correct"].mean()]
    axes[0].bar(["FP", "Quantized"], accuracies, color=[COLORS["fp"], COLORS["quant"]])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Static model accuracy")
    for index, value in enumerate(accuracies):
        axes[0].text(index, value + 0.025, f"{value:.1%}", ha="center")

    order = [
        "fp_correct_q_correct",
        "fp_correct_q_wrong",
        "fp_wrong_q_correct",
        "fp_wrong_q_wrong",
    ]
    labels = ["Both correct", "FP only", "Quant only", "Both wrong"]
    counts = frame["comparison_label"].value_counts().reindex(order, fill_value=0)
    colors = [COLORS["good"], COLORS["bad"], COLORS["fp"], COLORS["neutral"]]
    axes[1].bar(labels, counts, color=colors)
    axes[1].set_ylabel("Examples")
    axes[1].set_title("Correctness churn")
    axes[1].tick_params(axis="x", rotation=25)
    for index, value in enumerate(counts):
        axes[1].text(index, value + max(counts.max() * 0.02, 0.5), str(value), ha="center")
    fig.tight_layout()
    return _save(fig, output_dir, "01_accuracy_and_outcomes", formats)


def _feature_distributions(frame: pd.DataFrame, output_dir: Path, formats: list[str]) -> list[str]:
    eligible = frame[frame["eligible_fp_correct"] == 1]
    positive = eligible[eligible["target_quantization_failure"] == 1]
    negative = eligible[eligible["target_quantization_failure"] == 0]
    columns = [("max_entropy", "Maximum token entropy"), ("min_logit_margin", "Minimum logit margin")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, (column, title) in zip(axes, columns):
        axis.hist(negative[column].dropna(), bins=15, alpha=0.65, density=True, label="Q correct", color=COLORS["good"])
        axis.hist(positive[column].dropna(), bins=15, alpha=0.65, density=True, label="Q failure", color=COLORS["bad"])
        axis.set_title(title)
        axis.set_ylabel("Density")
        axis.legend()
    fig.tight_layout()
    return _save(fig, output_dir, "02_uncertainty_distributions", formats)


def _failure_probability(frame: pd.DataFrame, output_dir: Path, formats: list[str]) -> list[str]:
    eligible = frame[frame["eligible_fp_correct"] == 1].copy()
    quantiles = min(8, max(1, eligible["max_entropy"].nunique()))
    eligible["score_bin"] = pd.qcut(eligible["max_entropy"], q=quantiles, duplicates="drop")
    curve = eligible.groupby("score_bin", observed=True).agg(
        score=("max_entropy", "mean"),
        failure_probability=("target_quantization_failure", "mean"),
        count=("example_id", "size"),
    )
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.plot(curve["score"], curve["failure_probability"], marker="o", color=COLORS["bad"], linewidth=2)
    for _, row in curve.iterrows():
        axis.annotate(f"n={int(row['count'])}", (row["score"], row["failure_probability"]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    axis.axhline(eligible["target_quantization_failure"].mean(), linestyle="--", color=COLORS["neutral"], label="Overall prevalence")
    axis.set_xlabel("Maximum quantized-token entropy")
    axis.set_ylabel("P(FP correct, quantized wrong)")
    axis.set_ylim(bottom=0)
    axis.set_title("Failure concentration by uncertainty")
    axis.legend()
    fig.tight_layout()
    return _save(fig, output_dir, "03_failure_probability_vs_entropy", formats)


def _predictor_metrics(metrics: dict, output_dir: Path, formats: list[str]) -> list[str]:
    results = metrics["results"]
    names = list(results)
    roc = [results[name].get("roc_auc") or 0 for name in names]
    pr = [results[name].get("pr_auc") or 0 for name in names]
    roc_error = _ci_errors(roc, [results[name].get("roc_auc_ci95") for name in names])
    pr_error = _ci_errors(pr, [results[name].get("pr_auc_ci95") for name in names])
    x = np.arange(len(names))
    fig, axis = plt.subplots(figsize=(9, 4.5))
    width = 0.36
    axis.bar(x - width / 2, roc, width, yerr=roc_error, capsize=3, label="ROC-AUC", color=COLORS["fp"])
    axis.bar(x + width / 2, pr, width, yerr=pr_error, capsize=3, label="PR-AUC", color=COLORS["quant"])
    axis.axhline(metrics.get("test_positive_rate", metrics.get("positive_rate", 0)), color=COLORS["neutral"], linestyle="--", label="Positive prevalence")
    axis.set_xticks(x, [name.replace("_", "\n") for name in names])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Held-out sensitivity prediction")
    axis.legend(ncol=3)
    fig.tight_layout()
    return _save(fig, output_dir, "04_predictor_metrics", formats)


def _ci_errors(values: list[float], intervals: list[list[float] | None]) -> np.ndarray | None:
    if not any(interval is not None for interval in intervals):
        return None
    lower, upper = [], []
    for value, interval in zip(values, intervals):
        if interval is None:
            lower.append(0.0)
            upper.append(0.0)
        else:
            lower.append(max(0.0, value - interval[0]))
            upper.append(max(0.0, interval[1] - value))
    return np.array([lower, upper])


def _controller_curves(metrics: dict, output_dir: Path, formats: list[str]) -> list[str]:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    preferred = ["random_forest", "logistic_regression", "entropy_threshold", "oracle_at_most_budget"]
    for name in preferred:
        curve = metrics["intervention_curves"].get(name)
        if not curve:
            continue
        axis.plot(
            [100 * point["budget"] for point in curve],
            [point["accuracy"] for point in curve],
            marker="o",
            linewidth=2,
            label=name.replace("_", " ").title(),
        )
    axis.set_xlabel("High-precision intervention budget (%)")
    axis.set_ylabel("End-to-end accuracy")
    axis.set_title("Controller utility on all held-out examples")
    axis.legend()
    fig.tight_layout()
    return _save(fig, output_dir, "05_controller_accuracy_vs_budget", formats)


def _oracle_figure(report: dict, output_dir: Path, formats: list[str]) -> list[str]:
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    for key, label, style in [
        ("at_most_budget_curve", "Oracle: up to budget", "-"),
        ("exact_budget_curve", "Oracle: forced exact budget", "--"),
    ]:
        curve = report.get(key, [])
        axis.plot([100 * p["budget"] for p in curve], [p["accuracy"] for p in curve], marker="o", linestyle=style, label=label)
    axis.axhline(report.get("always_quantized_accuracy", 0), color=COLORS["quant"], linestyle=":", label="Always quantized")
    axis.axhline(report.get("always_fp_accuracy", 0), color=COLORS["fp"], linestyle=":", label="Always FP")
    axis.set_xlabel("High-precision intervention budget (%)")
    axis.set_ylabel("Accuracy")
    axis.set_title("Example-level oracle recovery")
    axis.legend()
    fig.tight_layout()
    return _save(fig, output_dir, "06_oracle_recovery", formats)


def _diagnostics_figure(report: dict, output_dir: Path, formats: list[str]) -> list[str]:
    labels = ["Hit token cap", "Explicit final answer", "No numeric answer"]
    fp = [report["fp"]["hit_max_new_tokens_rate"], report["fp"]["explicit_answer_rate"], report["fp"]["no_numeric_answer_rate"]]
    quant = [report["quant"]["hit_max_new_tokens_rate"], report["quant"]["explicit_answer_rate"], report["quant"]["no_numeric_answer_rate"]]
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.bar(x - width / 2, fp, width, label="FP", color=COLORS["fp"])
    axis.bar(x + width / 2, quant, width, label="Quantized", color=COLORS["quant"])
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Fraction of examples")
    axis.set_title("Generation-quality diagnostics")
    axis.legend()
    fig.tight_layout()
    return _save(fig, output_dir, "07_generation_quality", formats)


def _runtime_figure(audit: dict, output_dir: Path, formats: list[str]) -> list[str]:
    generation = audit["test"]["generation"]
    ratios = [
        generation["token_inflation_ratio_quant_over_fp"],
        generation["quant_latency_seconds"]["mean"] / generation["fp_latency_seconds"]["mean"],
        generation["quant_tokens_per_second"]["mean"] / generation["fp_tokens_per_second"]["mean"],
    ]
    labels = ["Generated tokens", "Example latency", "Throughput"]
    colors = [COLORS["quant"] if value > 1 else COLORS["good"] for value in ratios]
    fig, axis = plt.subplots(figsize=(7, 4.2))
    bars = axis.bar(labels, ratios, color=colors)
    axis.axhline(1.0, color=COLORS["neutral"], linestyle="--", label="FP16 baseline")
    axis.set_ylabel("BNB4 / FP16 ratio")
    axis.set_title("Observed runtime trade-off on RTX 3090")
    for bar, value in zip(bars, ratios):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.3f}×", ha="center")
    axis.legend()
    fig.tight_layout()
    return _save(fig, output_dir, "08_runtime_tradeoff", formats)


def _split_accuracy_figure(audit: dict, output_dir: Path, formats: list[str]) -> list[str]:
    fp = [audit[split]["paired_accuracy"]["fp_accuracy"] for split in ("train", "test")]
    quant = [audit[split]["paired_accuracy"]["quant_accuracy"] for split in ("train", "test")]
    x = np.arange(2)
    width = 0.36
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.bar(x - width / 2, fp, width, label="FP16", color=COLORS["fp"])
    axis.bar(x + width / 2, quant, width, label="BNB4", color=COLORS["quant"])
    axis.set_xticks(x, ["GSM8K train", "GSM8K test"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.set_title("Quantization degradation transfers across splits")
    axis.legend()
    fig.tight_layout()
    return _save(fig, output_dir, "09_train_test_accuracy", formats)


def _univariate_auc_figure(audit: dict, output_dir: Path, formats: list[str]) -> list[str]:
    values = audit["test"]["univariate_failure_auc_fp_correct_only"]
    preferred = [
        "generation_tokens",
        "max_entropy",
        "min_logit_margin",
        "mean_entropy",
        "mean_token_probability",
    ]
    labels = [name.replace("_", " ").title() for name in preferred]
    scores = [values[name] for name in preferred]
    fig, axis = plt.subplots(figsize=(8, 4.4))
    bars = axis.barh(labels, scores, color=COLORS["fp"])
    axis.axvline(0.5, color=COLORS["neutral"], linestyle="--", label="Random")
    axis.set_xlim(0.45, 0.75)
    axis.set_xlabel("Univariate ranking AUC")
    axis.set_title("Cheap signals separate quantization-induced failures")
    for bar, value in zip(bars, scores):
        axis.text(value + 0.005, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    axis.legend()
    fig.tight_layout()
    return _save(fig, output_dir, "10_univariate_feature_auc", formats)
