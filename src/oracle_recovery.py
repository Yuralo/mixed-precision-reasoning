"""Example-level recovery accounting; full reruns are selected outside this helper."""

from __future__ import annotations


def example_recovery_accuracy(comparisons: list[dict], intervened_ids: set[str]) -> float:
    if not comparisons:
        return 0.0
    correct = sum(
        row["fp_correct"] if row["example_id"] in intervened_ids else row["quant_correct"]
        for row in comparisons
    )
    return correct / len(comparisons)
