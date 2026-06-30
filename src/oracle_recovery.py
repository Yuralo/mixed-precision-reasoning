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


def intervention_utility(row: dict) -> int:
    """Accuracy change from replacing the quantized answer with the FP answer."""
    return int(bool(row["fp_correct"])) - int(bool(row["quant_correct"]))


def oracle_recovery_report(comparisons: list[dict], budgets: list[float]) -> dict:
    """Return both practical at-most-budget and forced-exact-budget oracle curves."""
    n = len(comparisons)
    if not n:
        return {"num_examples": 0, "at_most_budget_curve": [], "exact_budget_curve": []}
    beneficial = [row for row in comparisons if intervention_utility(row) > 0]
    neutral = [row for row in comparisons if intervention_utility(row) == 0]
    harmful = [row for row in comparisons if intervention_utility(row) < 0]
    ranked = beneficial + neutral + harmful
    at_most, exact = [], []
    for budget in budgets:
        allowance = min(n, int(round(n * budget)))
        useful_count = min(allowance, len(beneficial))
        useful_ids = {row["example_id"] for row in beneficial[:useful_count]}
        exact_ids = {row["example_id"] for row in ranked[:allowance]}
        at_most.append(
            {
                "budget": budget,
                "allowed_interventions": allowance,
                "used_interventions": useful_count,
                "accuracy": example_recovery_accuracy(comparisons, useful_ids),
            }
        )
        exact.append(
            {
                "budget": budget,
                "used_interventions": allowance,
                "accuracy": example_recovery_accuracy(comparisons, exact_ids),
            }
        )
    return {
        "num_examples": n,
        "always_quantized_accuracy": sum(bool(row["quant_correct"]) for row in comparisons) / n,
        "always_fp_accuracy": sum(bool(row["fp_correct"]) for row in comparisons) / n,
        "best_selector_accuracy": sum(bool(row["fp_correct"] or row["quant_correct"]) for row in comparisons) / n,
        "beneficial_interventions": len(beneficial),
        "harmful_interventions": len(harmful),
        "neutral_interventions": len(neutral),
        "at_most_budget_curve": at_most,
        "exact_budget_curve": exact,
    }
