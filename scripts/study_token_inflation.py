"""Paired study of whether quantization changes generation length and correctness."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.answer_extraction import extract_explicit_answer
from src.utils import ensure_parent, read_jsonl, write_json


LABEL_ORDER = [
    "fp_correct_q_correct",
    "fp_correct_q_wrong",
    "fp_wrong_q_correct",
    "fp_wrong_q_wrong",
]
SHORT_LABELS = ["Both correct", "FP16 only", "BNB4 only", "Both wrong"]
COLORS = ["#2A9D8F", "#E76F51", "#5271FF", "#8D99AE"]


def label(fp_correct: bool, q_correct: bool) -> str:
    if fp_correct and q_correct:
        return "fp_correct_q_correct"
    if fp_correct:
        return "fp_correct_q_wrong"
    if q_correct:
        return "fp_wrong_q_correct"
    return "fp_wrong_q_wrong"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def bootstrap_mean_ci(values: list[float], rng: random.Random, samples: int = 10000) -> list[float]:
    means = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(samples)]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def bootstrap_mean_difference_ci(
    first: list[float], second: list[float], rng: random.Random, samples: int = 10000
) -> list[float]:
    differences = [
        statistics.fmean(rng.choices(first, k=len(first)))
        - statistics.fmean(rng.choices(second, k=len(second)))
        for _ in range(samples)
    ]
    return [percentile(differences, 0.025), percentile(differences, 0.975)]


def exact_sign_test(longer: int, shorter: int) -> float:
    n = longer + shorter
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(longer, shorter) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def summarize(rows: list[dict], rng: random.Random) -> dict:
    fp_lengths = [row["fp_tokens"] for row in rows]
    q_lengths = [row["quant_tokens"] for row in rows]
    deltas = [row["token_delta"] for row in rows]
    ratios = [row["token_ratio"] for row in rows]
    longer = sum(delta > 0 for delta in deltas)
    shorter = sum(delta < 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    return {
        "n": len(rows),
        "fp_mean_tokens": statistics.fmean(fp_lengths),
        "fp_median_tokens": statistics.median(fp_lengths),
        "quant_mean_tokens": statistics.fmean(q_lengths),
        "quant_median_tokens": statistics.median(q_lengths),
        "mean_token_delta_quant_minus_fp": statistics.fmean(deltas),
        "mean_token_delta_ci95": bootstrap_mean_ci(deltas, rng),
        "median_token_delta": statistics.median(deltas),
        "ratio_of_total_tokens_quant_over_fp": sum(q_lengths) / sum(fp_lengths),
        "mean_per_example_token_ratio": statistics.fmean(ratios),
        "quant_longer_count": longer,
        "quant_shorter_count": shorter,
        "equal_length_count": ties,
        "quant_longer_rate": longer / len(rows),
        "sign_test_p_excluding_ties": exact_sign_test(longer, shorter),
    }


def build_rows(fp_path: str | Path, quant_path: str | Path) -> list[dict]:
    fp = {row["example_id"]: row for row in read_jsonl(fp_path)}
    quant = {row["example_id"]: row for row in read_jsonl(quant_path)}
    if fp.keys() != quant.keys():
        raise ValueError("FP16 and quantized example IDs do not match")
    rows = []
    for example_id in sorted(fp):
        fp_row, q_row = fp[example_id], quant[example_id]
        fp_tokens = int(fp_row["generation_tokens"])
        q_tokens = int(q_row["generation_tokens"])
        explicit = (
            extract_explicit_answer(fp_row["generation"]) is not None
            and extract_explicit_answer(q_row["generation"]) is not None
        )
        nontruncated = not fp_row.get("hit_max_new_tokens", False) and not q_row.get(
            "hit_max_new_tokens", False
        )
        rows.append(
            {
                "example_id": example_id,
                "outcome": label(bool(fp_row["correct"]), bool(q_row["correct"])),
                "fp_correct": int(bool(fp_row["correct"])),
                "quant_correct": int(bool(q_row["correct"])),
                "fp_tokens": fp_tokens,
                "quant_tokens": q_tokens,
                "token_delta": q_tokens - fp_tokens,
                "token_ratio": q_tokens / fp_tokens,
                "quant_longer": int(q_tokens > fp_tokens),
                "clean": int(explicit and nontruncated),
                "question": fp_row["question"],
            }
        )
    return rows


def delta_bin(delta: int) -> str:
    if delta <= -25:
        return "Q at least 25 shorter"
    if delta < 0:
        return "Q 1-24 shorter"
    if delta == 0:
        return "Equal length"
    if delta < 25:
        return "Q 1-24 longer"
    return "Q at least 25 longer"


def run_study(rows: list[dict], seed: int) -> dict:
    rng = random.Random(seed)
    groups = {name: [row for row in rows if row["outcome"] == name] for name in LABEL_ORDER}
    clean_rows = [row for row in rows if row["clean"]]
    clean_groups = {
        name: [row for row in clean_rows if row["outcome"] == name] for name in LABEL_ORDER
    }
    rescue_deltas = [row["token_delta"] for row in groups["fp_wrong_q_correct"]]
    comparisons = rescue_comparisons(groups, rng)
    clean_comparisons = rescue_comparisons(clean_groups, rng)

    bins = []
    bin_order = [
        "Q at least 25 shorter",
        "Q 1-24 shorter",
        "Equal length",
        "Q 1-24 longer",
        "Q at least 25 longer",
    ]
    for name in bin_order:
        subset = [row for row in rows if delta_bin(row["token_delta"]) == name]
        counts = Counter(row["outcome"] for row in subset)
        bins.append(
            {
                "bin": name,
                "n": len(subset),
                "fp_accuracy": sum(row["fp_correct"] for row in subset) / len(subset) if subset else None,
                "quant_accuracy": sum(row["quant_correct"] for row in subset) / len(subset) if subset else None,
                "quant_rescues": counts["fp_wrong_q_correct"],
                "quant_failures": counts["fp_correct_q_wrong"],
            }
        )
    return {
        "question": "Does BNB4 generate longer solutions, especially when it corrects an FP16 error?",
        "all_examples": summarize(rows, rng),
        "by_outcome": {name: summarize(group, rng) for name, group in groups.items()},
        "clean_examples": summarize(clean_rows, rng),
        "clean_by_outcome": {name: summarize(group, rng) for name, group in clean_groups.items()},
        "rescue_delta_comparisons": comparisons,
        "clean_rescue_delta_comparisons": clean_comparisons,
        "accuracy_by_length_change": bins,
        "interpretation_guardrail": (
            "Generation tokens measure output length, not reasoning depth or quality. Longer output can reflect "
            "additional valid steps, verbosity, repetition, instability, or delayed commitment."
        ),
    }


def rescue_comparisons(groups: dict[str, list[dict]], rng: random.Random) -> dict:
    rescue_deltas = [row["token_delta"] for row in groups["fp_wrong_q_correct"]]
    comparisons = {}
    for other in ("fp_correct_q_correct", "fp_correct_q_wrong", "fp_wrong_q_wrong"):
        other_deltas = [row["token_delta"] for row in groups[other]]
        difference = statistics.fmean(rescue_deltas) - statistics.fmean(other_deltas)
        comparisons[f"quant_rescue_minus_{other}"] = {
            "difference_in_mean_token_delta": difference,
            "bootstrap_ci95": bootstrap_mean_difference_ci(rescue_deltas, other_deltas, rng),
        }
    return comparisons


def write_csv(path: str | Path, rows: list[dict]) -> None:
    target = ensure_parent(path)
    fieldnames = list(rows[0])
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict], output: str | Path) -> None:
    groups = [[row for row in rows if row["outcome"] == name] for name in LABEL_ORDER]
    fp_means = [statistics.fmean(row["fp_tokens"] for row in group) for group in groups]
    q_means = [statistics.fmean(row["quant_tokens"] for row in group) for group in groups]
    deltas = [[row["token_delta"] for row in group] for group in groups]
    direction = [
        [
            sum(row["token_delta"] < 0 for row in group) / len(group),
            sum(row["token_delta"] == 0 for row in group) / len(group),
            sum(row["token_delta"] > 0 for row in group) / len(group),
        ]
        for group in groups
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(groups))
    width = 0.36
    axes[0].bar(x - width / 2, fp_means, width, color="#355C7D", label="FP16")
    axes[0].bar(x + width / 2, q_means, width, color="#F67280", label="BNB4")
    axes[0].set_xticks(x, SHORT_LABELS, rotation=20)
    axes[0].set_ylabel("Mean generated tokens")
    axes[0].set_title("Generation length by paired outcome")
    axes[0].legend()

    boxes = axes[1].boxplot(deltas, tick_labels=SHORT_LABELS, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1].axhline(0, color="#667085", linestyle="--")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylabel("BNB4 tokens − FP16 tokens")
    axes[1].set_title("Paired token-count change")

    bottom = np.zeros(len(groups))
    direction_array = np.array(direction)
    for index, (name, color) in enumerate(
        [("BNB4 shorter", "#355C7D"), ("Equal", "#D9E1EA"), ("BNB4 longer", "#F67280")]
    ):
        axes[2].bar(x, direction_array[:, index], bottom=bottom, color=color, label=name)
        bottom += direction_array[:, index]
    axes[2].set_xticks(x, SHORT_LABELS, rotation=20)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Fraction of examples")
    axes[2].set_title("Which model generated more tokens?")
    axes[2].legend()

    fig.suptitle("Does 4-bit quantization induce longer reasoning traces?", fontsize=16)
    fig.tight_layout()
    target = ensure_parent(output)
    fig.savefig(target, dpi=180, bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_clean_figure(study: dict, output: str | Path) -> None:
    summaries = [study["clean_by_outcome"][name] for name in LABEL_ORDER]
    means = [summary["mean_token_delta_quant_minus_fp"] for summary in summaries]
    lower = [mean - summary["mean_token_delta_ci95"][0] for mean, summary in zip(means, summaries)]
    upper = [summary["mean_token_delta_ci95"][1] - mean for mean, summary in zip(means, summaries)]
    x = np.arange(len(means))
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.bar(x, means, color=COLORS, alpha=0.85, yerr=np.array([lower, upper]), capsize=5)
    axis.axhline(0, color="#667085", linestyle="--")
    axis.set_xticks(x, SHORT_LABELS)
    axis.set_ylabel("Mean BNB4 tokens − FP16 tokens")
    axis.set_title("Clean paired traces: rescues lengthen while failures shorten")
    for index, mean in enumerate(means):
        axis.text(index, mean + (3 if mean >= 0 else -7), f"{mean:+.1f}", ha="center", fontweight="bold")
    axis.text(
        0.02,
        0.02,
        "Error bars: paired-group bootstrap 95% intervals",
        transform=axis.transAxes,
        color="#667085",
        fontsize=9,
    )
    fig.tight_layout()
    target = ensure_parent(output)
    fig.savefig(target, dpi=180, bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp", default="runs/gsm8k_test/fp_outputs.jsonl")
    parser.add_argument("--quant", default="runs/gsm8k_test/quant_outputs.jsonl")
    parser.add_argument("--output", default="runs/gsm8k_test/token_length_study.json")
    parser.add_argument("--pairs", default="runs/gsm8k_test/token_length_pairs.csv")
    parser.add_argument("--figure", default="runs/gsm8k_test/figures/11_token_length_by_outcome.png")
    parser.add_argument("--clean-figure", default="runs/gsm8k_test/figures/12_clean_token_delta.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = build_rows(args.fp, args.quant)
    study = run_study(rows, args.seed)
    write_json(args.output, study)
    write_csv(args.pairs, rows)
    make_figure(rows, args.figure)
    make_clean_figure(study, args.clean_figure)
    print(json.dumps(study, indent=2))


if __name__ == "__main__":
    main()
