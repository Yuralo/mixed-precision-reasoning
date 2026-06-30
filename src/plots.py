"""Figures for precision-induced trajectory analysis."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .paired_analysis import OUTCOMES
from .utils import ensure_parent, read_jsonl


SHORT = {
    "fp_correct_q_correct": "Both correct",
    "fp_correct_q_wrong": "FP16 only",
    "fp_wrong_q_correct": "BNB4 only",
    "fp_wrong_q_wrong": "Both wrong",
}
COLORS = {
    "fp_correct_q_correct": "#2A9D8F",
    "fp_correct_q_wrong": "#E76F51",
    "fp_wrong_q_correct": "#5271FF",
    "fp_wrong_q_wrong": "#8D99AE",
}


def save(fig: plt.Figure, path: str | Path) -> None:
    target = ensure_parent(path)
    fig.savefig(target, dpi=180, bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)["records"]


def plot_clean_structure_deltas(records: list[dict], output: str | Path) -> None:
    clean = [record for record in records if record["clean"]]
    metrics = [
        ("delta_generated_tokens", "Generated tokens"),
        ("delta_line_count", "Non-empty lines"),
        ("delta_arithmetic_expression_count", "Arithmetic expressions"),
        ("delta_normalized_final_answer_position", "Final-answer position"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    x = np.arange(len(OUTCOMES))
    for axis, (metric, title) in zip(axes, metrics):
        means = []
        for outcome in OUTCOMES:
            values = [record[metric] for record in clean if record["outcome"] == outcome and record[metric] is not None]
            means.append(float(np.mean(values)))
        if metric == "delta_normalized_final_answer_position":
            means = [value * 100 for value in means]
            ylabel = "BNB4 − FP16 (percentage points)"
        else:
            ylabel = "BNB4 − FP16"
        axis.bar(x, means, color=[COLORS[outcome] for outcome in OUTCOMES])
        axis.axhline(0, color="#667085", linestyle="--", linewidth=1)
        axis.set_xticks(x, [SHORT[outcome] for outcome in OUTCOMES], rotation=28)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    fig.suptitle("Clean paired trajectory-structure changes")
    fig.tight_layout()
    save(fig, output)


def plot_first_divergence(records: list[dict], output: str | Path) -> None:
    categories = ["Token 0", "Tokens 1–4", "Tokens 5–15", "Token 16+", "No divergence"]
    data = []
    for outcome in OUTCOMES:
        subset = [record for record in records if record["outcome"] == outcome]
        counts = [0] * len(categories)
        for record in subset:
            value = record.get("first_divergence_position")
            if value is None:
                counts[4] += 1
            elif value == 0:
                counts[0] += 1
            elif value <= 4:
                counts[1] += 1
            elif value <= 15:
                counts[2] += 1
            else:
                counts[3] += 1
        data.append([count / len(subset) for count in counts])
    array = np.array(data)
    fig, axis = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(OUTCOMES))
    bottom = np.zeros(len(OUTCOMES))
    palette = ["#172033", "#355C7D", "#5271FF", "#30C6B0", "#D9E1EA"]
    for index, category in enumerate(categories):
        axis.bar(x, array[:, index], bottom=bottom, label=category, color=palette[index])
        bottom += array[:, index]
    axis.set_xticks(x, [SHORT[outcome] for outcome in OUTCOMES])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Fraction of paired outputs")
    axis.set_title("Independent greedy traces usually diverge before substantive reasoning")
    axis.legend(ncol=3)
    fig.tight_layout()
    save(fig, output)


def normalized_entropy_curves(
    records: list[dict], fp_token_path: str | Path, quant_token_path: str | Path, bins: int = 20
) -> dict[str, dict[str, list[float | None]]]:
    outcome_by_id = {record["example_id"]: record["outcome"] for record in records}
    result: dict[str, dict[str, list[list[float]]]] = {
        outcome: {"fp": [[] for _ in range(bins)], "quant": [[] for _ in range(bins)]}
        for outcome in OUTCOMES
    }
    for model, path in (("fp", fp_token_path), ("quant", quant_token_path)):
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in read_jsonl(path):
            grouped[row["example_id"]].append(row)
        for example_id, rows in grouped.items():
            rows.sort(key=lambda row: int(row["token_position"]))
            outcome = outcome_by_id[example_id]
            length = len(rows)
            for index, row in enumerate(rows):
                bin_index = min(bins - 1, int(index / max(1, length) * bins))
                result[outcome][model][bin_index].append(float(row["entropy"]))
    means: dict[str, dict[str, list[float | None]]] = {}
    for outcome in OUTCOMES:
        means[outcome] = {}
        for model in ("fp", "quant"):
            means[outcome][model] = [float(np.mean(values)) if values else None for values in result[outcome][model]]
    return means


def plot_entropy_trajectories(curves: dict, output: str | Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    x = np.linspace(0, 1, len(next(iter(curves.values()))["fp"]))
    for axis, outcome in zip(axes.ravel(), OUTCOMES):
        axis.plot(x, curves[outcome]["fp"], label="FP16", color="#355C7D", linewidth=2)
        axis.plot(x, curves[outcome]["quant"], label="BNB4", color="#F67280", linewidth=2)
        axis.set_title(SHORT[outcome])
        axis.set_xlabel("Normalized generation position")
        axis.set_ylabel("Mean next-token entropy")
        axis.legend()
    fig.suptitle("Entropy trajectories by paired correctness outcome")
    fig.tight_layout()
    save(fig, output)


def plot_utility_controller(report: dict, output: str | Path) -> None:
    """Plot the routing frontier without hiding harmful FP interventions."""
    best = report["models"][report["best_learned_at_10pct"]]["intervention_curve"]
    series = {
        "Learned router": best,
        "Entropy threshold": report["baselines"]["entropy"],
        "Length threshold": report["baselines"]["length"],
        "Random (expected)": report["baselines"]["random_expected"],
        "Oracle": report["baselines"]["oracle"],
    }
    colors = {
        "Learned router": "#5271FF",
        "Entropy threshold": "#E76F51",
        "Length threshold": "#F4A261",
        "Random (expected)": "#8D99AE",
        "Oracle": "#2A9D8F",
    }
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    for label, curve in series.items():
        axis.plot(
            [point["budget"] * 100 for point in curve],
            [point["accuracy"] * 100 for point in curve],
            marker="o",
            label=label,
            color=colors[label],
        )
    axis.axhline(report["static"]["always_quantized"] * 100, color="#172033", linestyle="--", label="Always BNB4")
    axis.axhline(report["static"]["always_fp16"] * 100, color="#6D597A", linestyle="--", label="Always FP16")
    axis.set_xlabel("FP16 rerun budget (%)")
    axis.set_ylabel("Test accuracy (%)")
    axis.set_title("Utility-aware routing: opportunity remains mostly unrealized")
    axis.legend(ncol=2)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, output)


def plot_controller_calibration(
    predictions_path: str | Path, best_model: str, output: str | Path
) -> None:
    import pandas as pd

    frame = pd.read_csv(predictions_path)
    probability = frame[f"p_beneficial_{best_model}"].to_numpy(dtype=float)
    actual = frame["outcome"].eq("fp_correct_q_wrong").to_numpy(dtype=float)
    edges = np.quantile(probability, np.linspace(0, 1, 11))
    edges = np.unique(edges)
    bins = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, len(edges) - 2)
    predicted, observed, counts = [], [], []
    for index in range(max(0, len(edges) - 1)):
        mask = bins == index
        if mask.any():
            predicted.append(float(probability[mask].mean()))
            observed.append(float(actual[mask].mean()))
            counts.append(int(mask.sum()))
    fig, axis = plt.subplots(figsize=(5.8, 5.2))
    axis.plot([0, 0.5], [0, 0.5], color="#8D99AE", linestyle="--", label="Perfect calibration")
    axis.plot(predicted, observed, color="#5271FF", marker="o", linewidth=2, label="Learned router")
    for x, y, count in zip(predicted, observed, counts):
        axis.annotate(f"n={count}", (x, y), xytext=(4, 5), textcoords="offset points", fontsize=8)
    axis.set_xlim(0, max(0.35, max(predicted, default=0.3) + 0.03))
    axis.set_ylim(0, max(0.35, max(observed, default=0.3) + 0.03))
    axis.set_xlabel("Predicted P(FP16 helps)")
    axis.set_ylabel("Observed FP16-help rate")
    axis.set_title("Beneficial-switch calibration on held-out test")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, output)


def plot_risk_coverage(report: dict, output: str | Path) -> None:
    best = report["models"][report["best_learned_at_10pct"]]
    curve = best["selective_risk_curve_at_10pct"]
    coverage = [point["coverage"] * 100 for point in curve]
    risk = [point["selective_risk"] * 100 for point in curve]
    accuracy = [point["selective_accuracy"] * 100 for point in curve]
    fig, axis = plt.subplots(figsize=(6.4, 4.9))
    axis.plot(coverage, risk, color="#E76F51", marker="o", linewidth=2, label="Selective risk")
    axis.plot(coverage, accuracy, color="#5271FF", marker="o", linewidth=2, label="Selective accuracy")
    axis.set_xlabel("Coverage: fraction answered (%)")
    axis.set_ylabel("Rate (%)")
    axis.set_title("Abstention after 10% precision routing")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    save(fig, output)


def plot_prefix_prediction(report: dict, output: str | Path) -> None:
    budgets = [int(value) for value in report["budgets"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for model, color in (("logistic", "#E76F51"), ("random_forest", "#5271FF")):
        auc = [report["results"][str(budget)][model]["beneficial_roc_auc"] for budget in budgets]
        accuracy = []
        for budget in budgets:
            curve = report["results"][str(budget)][model]["intervention_curve"]
            accuracy.append(next(point["accuracy"] for point in curve if point["budget"] == 0.10))
        axes[0].plot(budgets, auc, marker="o", label=model.replace("_", " ").title(), color=color)
        axes[1].plot(budgets, np.array(accuracy) * 100, marker="o", label=model.replace("_", " ").title(), color=color)
    axes[0].axhline(0.5, color="#8D99AE", linestyle="--", label="Chance ROC-AUC")
    axes[0].set_ylabel("ROC-AUC for FP16-beneficial case")
    axes[1].axhline(63.0, color="#172033", linestyle="--", label="Always BNB4")
    axes[1].axhline(69.0, color="#6D597A", linestyle="--", label="Always FP16")
    axes[1].set_ylabel("Accuracy with 10% FP16 reruns (%)")
    for axis in axes:
        axis.set_xlabel("Observed BNB4 prefix tokens")
        axis.set_xticks(budgets)
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_title("Predictive signal emerges late")
    axes[1].set_title("Early routing does not beat static FP16")
    fig.tight_layout()
    save(fig, output)


def plot_temperature_analysis(report: dict, output: str | Path) -> None:
    temperatures = sorted(report["temperatures"], key=float)
    labels = [f"T={value}" for value in temperatures]
    completion = [report["temperatures"][value]["per_completion_accuracy"] * 100 for value in temperatures]
    pass_k = [report["temperatures"][value]["pass_at_k_empirical"] * 100 for value in temperatures]
    majority = [report["temperatures"][value]["majority_vote_accuracy"] * 100 for value in temperatures]
    coverage = [report["temperatures"][value]["quant_rescue_coverage"] * 100 for value in temperatures]
    majority_coverage = [
        report["temperatures"][value]["quant_rescue_majority_coverage"] * 100
        for value in temperatures
    ]
    jaccard = [report["temperatures"][value]["quant_vs_temperature_rescue_jaccard"] * 100 for value in temperatures]
    x = np.arange(len(temperatures))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    axes[0].bar(x - width, completion, width, label="Per completion", color="#5271FF")
    axes[0].bar(x, majority, width, label="Majority vote", color="#2A9D8F")
    axes[0].bar(x + width, pass_k, width, label="Any sample", color="#F4A261")
    axes[0].axhline(report["paired_subset"]["fp_accuracy"] * 100, color="#6D597A", linestyle="--", label="FP16 greedy")
    axes[0].axhline(report["paired_subset"]["quant_accuracy"] * 100, color="#172033", linestyle="--", label="BNB4 greedy")
    axes[0].set_ylabel("Accuracy / empirical pass@k (%)")
    axes[0].set_title("Sampling outcomes")
    axes[1].bar(x - width, coverage, width, label="Any-sample rescue coverage", color="#E76F51")
    axes[1].bar(x, majority_coverage, width, label="Majority rescue coverage", color="#F4A261")
    axes[1].bar(x + width, jaccard, width, label="Rescue-set Jaccard", color="#8D99AE")
    axes[1].set_ylabel("Overlap (%)")
    axes[1].set_title("Does temperature reproduce BNB4 rescues?")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    save(fig, output)
