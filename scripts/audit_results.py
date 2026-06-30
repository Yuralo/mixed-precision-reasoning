"""Dependency-light statistical audit of matched FP/quantized reasoning runs."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.answer_extraction import extract_answer, extract_explicit_answer
from src.utils import read_json, read_jsonl, write_json


LABELS = (
    "fp_correct_q_correct",
    "fp_correct_q_wrong",
    "fp_wrong_q_correct",
    "fp_wrong_q_wrong",
)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def describe(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "q25": percentile(values, 0.25),
        "q75": percentile(values, 0.75),
    }


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def bootstrap_difference(
    fp_correct: list[int], quant_correct: list[int], seed: int = 42, samples: int = 10000
) -> list[float]:
    rng = random.Random(seed)
    differences = []
    n = len(fp_correct)
    for _ in range(samples):
        total = 0
        for _ in range(n):
            index = rng.randrange(n)
            total += fp_correct[index] - quant_correct[index]
        differences.append(total / n)
    return [percentile(differences, 0.025), percentile(differences, 0.975)]


def rank_auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = 0.0
    for pos in positive:
        for neg in negative:
            if pos > neg:
                wins += 1
            elif pos == neg:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def load_token_aggregates(path: str | Path) -> tuple[dict[str, dict[str, float]], dict[str, list[int]]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    token_ids: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in read_jsonl(path):
        example_id = row["example_id"]
        values[example_id]["entropy"].append(float(row["entropy"]))
        values[example_id]["margin"].append(float(row["logit_margin"]))
        values[example_id]["probability"].append(float(row["token_probability"]))
        token_ids[example_id].append((int(row["token_position"]), int(row["token_id"])))
    aggregates = {}
    for example_id, metrics in values.items():
        aggregates[example_id] = {
            "max_entropy": max(metrics["entropy"]),
            "mean_entropy": statistics.fmean(metrics["entropy"]),
            "min_logit_margin": min(metrics["margin"]),
            "mean_logit_margin": statistics.fmean(metrics["margin"]),
            "min_token_probability": min(metrics["probability"]),
            "mean_token_probability": statistics.fmean(metrics["probability"]),
            "generation_tokens": len(metrics["entropy"]),
        }
    ordered_ids = {
        example_id: [token_id for _, token_id in sorted(items)] for example_id, items in token_ids.items()
    }
    return aggregates, ordered_ids


def comparison_label(fp_correct: bool, quant_correct: bool) -> str:
    if fp_correct and quant_correct:
        return "fp_correct_q_correct"
    if fp_correct:
        return "fp_correct_q_wrong"
    if quant_correct:
        return "fp_wrong_q_correct"
    return "fp_wrong_q_wrong"


def audit_split(
    fp_path: str | Path,
    quant_path: str | Path,
    fp_tokens_path: str | Path,
    quant_tokens_path: str | Path,
    seed: int,
) -> dict[str, Any]:
    fp_rows, quant_rows = read_jsonl(fp_path), read_jsonl(quant_path)
    fp_by_id = {row["example_id"]: row for row in fp_rows}
    quant_by_id = {row["example_id"]: row for row in quant_rows}
    shared = sorted(fp_by_id.keys() & quant_by_id.keys())
    fp_aggregates, fp_token_ids = load_token_aggregates(fp_tokens_path)
    q_aggregates, q_token_ids = load_token_aggregates(quant_tokens_path)

    records = []
    extraction_disagreements = 0
    for example_id in shared:
        fp, quant = fp_by_id[example_id], quant_by_id[example_id]
        gold = extract_answer(fp["reference"])
        current_fp = extract_answer(fp["generation"])
        current_q = extract_answer(quant["generation"])
        extraction_disagreements += int(current_fp != fp.get("predicted_answer"))
        extraction_disagreements += int(current_q != quant.get("predicted_answer"))
        fp_correct, q_correct = bool(fp["correct"]), bool(quant["correct"])
        feature = q_aggregates.get(example_id, {})
        fp_ids, q_ids = fp_token_ids.get(example_id, []), q_token_ids.get(example_id, [])
        first_disagreement = next(
            (index for index, pair in enumerate(zip(fp_ids, q_ids)) if pair[0] != pair[1]), None
        )
        aligned = min(len(fp_ids), len(q_ids))
        disagreement_rate = (
            sum(fp_ids[index] != q_ids[index] for index in range(aligned)) / aligned if aligned else None
        )
        records.append(
            {
                "example_id": example_id,
                "fp": fp,
                "quant": quant,
                "gold": gold,
                "fp_answer": current_fp,
                "quant_answer": current_q,
                "fp_correct": fp_correct,
                "quant_correct": q_correct,
                "label": comparison_label(fp_correct, q_correct),
                "feature": feature,
                "first_token_disagreement": first_disagreement,
                "aligned_token_disagreement_rate": disagreement_rate,
            }
        )

    n = len(records)
    fp_correct_values = [int(record["fp_correct"]) for record in records]
    q_correct_values = [int(record["quant_correct"]) for record in records]
    fp_total, q_total = sum(fp_correct_values), sum(q_correct_values)
    labels = Counter(record["label"] for record in records)
    beneficial = labels["fp_correct_q_wrong"]
    harmful = labels["fp_wrong_q_correct"]
    clean = [
        record
        for record in records
        if not record["fp"].get("hit_max_new_tokens", False)
        and not record["quant"].get("hit_max_new_tokens", False)
        and extract_explicit_answer(record["fp"]["generation"]) is not None
        and extract_explicit_answer(record["quant"]["generation"]) is not None
    ]

    group_features = {}
    for label in LABELS:
        group = [record for record in records if record["label"] == label]
        group_features[label] = {
            feature_name: describe([record["feature"][feature_name] for record in group if feature_name in record["feature"]])
            for feature_name in (
                "max_entropy",
                "mean_entropy",
                "min_logit_margin",
                "mean_logit_margin",
                "min_token_probability",
                "mean_token_probability",
                "generation_tokens",
            )
        }
        group_features[label]["first_token_disagreement"] = describe(
            [float(record["first_token_disagreement"]) for record in group if record["first_token_disagreement"] is not None]
        )

    positive = [record for record in records if record["label"] == "fp_correct_q_wrong"]
    negative = [record for record in records if record["label"] == "fp_correct_q_correct"]
    directions = {
        "max_entropy": 1,
        "mean_entropy": 1,
        "min_logit_margin": -1,
        "mean_logit_margin": -1,
        "min_token_probability": -1,
        "mean_token_probability": -1,
        "generation_tokens": 1,
    }
    univariate_auc = {}
    for feature_name, direction in directions.items():
        pos = [direction * record["feature"][feature_name] for record in positive]
        neg = [direction * record["feature"][feature_name] for record in negative]
        univariate_auc[feature_name] = rank_auc(pos, neg)

    clean_labels = Counter(record["label"] for record in clean)
    token_inflation = sum(record["quant"]["generation_tokens"] for record in records) / max(
        1, sum(record["fp"]["generation_tokens"] for record in records)
    )
    report = {
        "artifact_integrity": {
            "fp_rows": len(fp_rows),
            "quant_rows": len(quant_rows),
            "matched_rows": n,
            "duplicate_fp_ids": len(fp_rows) - len(fp_by_id),
            "duplicate_quant_ids": len(quant_rows) - len(quant_by_id),
            "id_sets_equal": fp_by_id.keys() == quant_by_id.keys(),
            "reference_mismatches": sum(
                fp_by_id[key]["reference"] != quant_by_id[key]["reference"] for key in shared
            ),
            "stored_vs_current_extraction_disagreements": extraction_disagreements,
            "fp_model": fp_rows[0].get("model") if fp_rows else None,
            "quant_model": quant_rows[0].get("model") if quant_rows else None,
        },
        "paired_accuracy": {
            "num_examples": n,
            "fp_accuracy": fp_total / n if n else 0,
            "fp_accuracy_ci95": wilson(fp_total, n),
            "quant_accuracy": q_total / n if n else 0,
            "quant_accuracy_ci95": wilson(q_total, n),
            "difference_fp_minus_quant": (fp_total - q_total) / n if n else 0,
            "paired_difference_bootstrap_ci95": bootstrap_difference(fp_correct_values, q_correct_values, seed),
            "mcnemar_exact_p": exact_mcnemar(beneficial, harmful),
            "discordant_odds_ratio_fp_over_quant": beneficial / harmful if harmful else None,
            "answer_flip_rate": sum(record["fp_answer"] != record["quant_answer"] for record in records) / n if n else 0,
            "label_counts": dict(labels),
            "critical_failure_rate_given_fp_correct": beneficial / fp_total if fp_total else 0,
            "critical_failure_rate_ci95": wilson(beneficial, fp_total),
            "best_selector_accuracy": sum(record["fp_correct"] or record["quant_correct"] for record in records) / n if n else 0,
        },
        "clean_subset": {
            "num_examples": len(clean),
            "fp_accuracy": sum(record["fp_correct"] for record in clean) / len(clean) if clean else 0,
            "quant_accuracy": sum(record["quant_correct"] for record in clean) / len(clean) if clean else 0,
            "label_counts": dict(clean_labels),
            "mcnemar_exact_p": exact_mcnemar(
                clean_labels["fp_correct_q_wrong"], clean_labels["fp_wrong_q_correct"]
            ),
        },
        "generation": {
            "fp_stop_reasons": dict(Counter(record["fp"].get("stop_reason") for record in records)),
            "quant_stop_reasons": dict(Counter(record["quant"].get("stop_reason") for record in records)),
            "fp_generation_tokens": describe([record["fp"]["generation_tokens"] for record in records]),
            "quant_generation_tokens": describe([record["quant"]["generation_tokens"] for record in records]),
            "token_inflation_ratio_quant_over_fp": token_inflation,
            "fp_latency_seconds": describe([record["fp"]["latency_seconds"] for record in records]),
            "quant_latency_seconds": describe([record["quant"]["latency_seconds"] for record in records]),
            "fp_tokens_per_second": describe([record["fp"]["tokens_per_second"] for record in records]),
            "quant_tokens_per_second": describe([record["quant"]["tokens_per_second"] for record in records]),
        },
        "feature_groups": group_features,
        "univariate_failure_auc_fp_correct_only": univariate_auc,
        "representative_clean_examples": {
            label: representative(records, label, count=3) for label in LABELS
        },
    }
    return report


def representative(records: list[dict[str, Any]], label: str, count: int) -> list[dict[str, Any]]:
    candidates = [
        record
        for record in records
        if record["label"] == label
        and not record["fp"].get("hit_max_new_tokens", False)
        and not record["quant"].get("hit_max_new_tokens", False)
        and extract_explicit_answer(record["fp"]["generation"]) is not None
        and extract_explicit_answer(record["quant"]["generation"]) is not None
    ]
    candidates.sort(key=lambda record: record["feature"].get("max_entropy", 0), reverse=True)
    return [
        {
            "example_id": record["example_id"],
            "question": record["fp"]["question"],
            "reference_answer": record["gold"],
            "fp_answer": record["fp_answer"],
            "quant_answer": record["quant_answer"],
            "max_entropy": record["feature"].get("max_entropy"),
            "min_logit_margin": record["feature"].get("min_logit_margin"),
            "first_token_disagreement": record["first_token_disagreement"],
            "fp_generation_tail": record["fp"]["generation"][-600:],
            "quant_generation_tail": record["quant"]["generation"][-600:],
        }
        for record in candidates[:count]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", default="runs/research_audit.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.runs_dir)
    report = {
        "train": audit_split(
            root / "gsm8k_train/fp_outputs.jsonl",
            root / "gsm8k_train/quant_outputs.jsonl",
            root / "gsm8k_train/fp_token_features.jsonl",
            root / "gsm8k_train/quant_token_features.jsonl",
            args.seed,
        ),
        "test": audit_split(
            root / "gsm8k_test/fp_outputs.jsonl",
            root / "gsm8k_test/quant_outputs.jsonl",
            root / "gsm8k_test/fp_token_features.jsonl",
            root / "gsm8k_test/quant_token_features.jsonl",
            args.seed,
        ),
        "held_out_predictor": read_json(root / "gsm8k_test/predictor_metrics.json"),
        "clean_held_out_predictor": read_json(root / "gsm8k_test/predictor_metrics_clean.json"),
        "oracle": read_json(root / "gsm8k_test/oracle_recovery.json"),
    }
    write_json(args.output, report)
    compact = {
        split: {
            "integrity": report[split]["artifact_integrity"],
            "accuracy": report[split]["paired_accuracy"],
            "clean": report[split]["clean_subset"],
            "generation": report[split]["generation"],
            "univariate_auc": report[split]["univariate_failure_auc_fp_correct_only"],
        }
        for split in ("train", "test")
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
