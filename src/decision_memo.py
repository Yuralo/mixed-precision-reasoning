"""Build an evidence-gated continue/pivot memo from the experiment artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import read_json


def _point(curve: list[dict[str, Any]], budget: float) -> dict[str, Any]:
    return next(item for item in curve if float(item["budget"]) == budget)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_decision_memo(
    audit: dict[str, Any],
    trajectory: dict[str, Any],
    length: dict[str, Any],
    utility: dict[str, Any],
    prefix: dict[str, Any],
    temperature: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    test = audit["test"]
    paired = test["paired_accuracy"]
    clean = test["clean_subset"]
    outcomes = paired["label_counts"]
    best_model = utility["models"][utility["best_learned_at_10pct"]]
    learned10 = _point(best_model["intervention_curve"], 0.10)
    entropy10 = _point(utility["baselines"]["entropy"], 0.10)
    oracle10 = _point(utility["baselines"]["oracle"], 0.10)
    oracle20 = _point(utility["baselines"]["oracle"], 0.20)
    prefix_rows = []
    for budget in prefix["budgets"]:
        models = prefix["results"][str(budget)]
        name, values = max(models.items(), key=lambda item: item[1]["beneficial_roc_auc"])
        prefix_rows.append(
            (budget, name, values["beneficial_roc_auc"], _point(values["intervention_curve"], 0.10)["accuracy"])
        )

    temperature_complete = bool(temperature and temperature.get("temperatures"))
    learned_beats_entropy = learned10["accuracy"] > entropy10["accuracy"]
    another_model_or_dataset = False  # No cross-model/dataset artifact exists yet.
    if temperature_complete and learned_beats_entropy and another_model_or_dataset:
        framing = "A. Precision as a Control Variable for Reasoning"
        recommendation = "Continue strongly"
    elif another_model_or_dataset and paired["best_selector_accuracy"] > paired["fp_accuracy"] + 0.03:
        framing = "B. Counterfactual Precision Routing for Reliable Quantized Reasoning"
        recommendation = "Continue, but narrow scope"
    else:
        framing = "C. Empirical Study of Quantization-Induced Reasoning Trajectory Shifts"
        recommendation = (
            "Only continue as empirical analysis"
            if temperature_complete
            else "Continue, but narrow scope"
        )

    opening = (
        "The paired precision effect is real, but the temperature control weakens the claim that precision is a distinct, exploitable control variable. "
        "Continue only through one independent replication; do not invest in token-level kernels or present adaptive precision as a demonstrated system."
        if temperature_complete
        else
        "The paired precision effect is real in these artifacts, but the stronger causal/control thesis is not yet earned. "
        "The current evidence supports a short, explicitly gated continuation: run the temperature control and one independent replication. "
        "Do not invest in token-level precision kernels unless those gates pass."
    )

    lines = [
        "# Continue-or-pivot decision memo",
        "",
        "## Decision",
        "",
        f"**Recommendation: {recommendation}.**",
        "",
        f"**Strongest defensible framing today: {framing}.**",
        "",
        opening,
        "",
        "## 1. What replicated?",
        "",
        "| Result | Train (400) | Held-out test (400) | Interpretation |",
        "|---|---:|---:|---|",
        f"| FP16 accuracy | {_pct(audit['train']['paired_accuracy']['fp_accuracy'])} | {_pct(paired['fp_accuracy'])} | FP16 is stronger in both splits |",
        f"| BNB4 accuracy | {_pct(audit['train']['paired_accuracy']['quant_accuracy'])} | {_pct(paired['quant_accuracy'])} | Quantization costs aggregate accuracy |",
        f"| FP16-only wins | {audit['train']['paired_accuracy']['label_counts']['fp_correct_q_wrong']} | {outcomes['fp_correct_q_wrong']} | Harm from BNB4 is repeatable |",
        f"| BNB4-only wins | {audit['train']['paired_accuracy']['label_counts']['fp_wrong_q_correct']} | {outcomes['fp_wrong_q_correct']} | Success sets are not nested |",
        f"| Oracle selector | {_pct(audit['train']['paired_accuracy']['best_selector_accuracy'])} | {_pct(paired['best_selector_accuracy'])} | Counterfactual complementarity is large |",
        "",
        f"On test, the FP16–BNB4 difference is {_pct(paired['difference_fp_minus_quant'])} with paired bootstrap 95% CI "
        f"[{_pct(paired['paired_difference_bootstrap_ci95'][0])}, {_pct(paired['paired_difference_bootstrap_ci95'][1])}] "
        f"and exact McNemar p={paired['mcnemar_exact_p']:.3g}. The answer-flip rate is {_pct(paired['answer_flip_rate'])}, "
        "far larger than the net accuracy gap. This supports H1 for one model/dataset/quantizer.",
        "",
        f"The clean subset retains {clean['label_counts']['fp_wrong_q_correct']} BNB4-only wins and "
        f"{clean['label_counts']['fp_correct_q_wrong']} FP16-only wins across {clean['num_examples']} examples; therefore the phenomenon is not explained away by truncation or missing explicit answers.",
        "",
        "## 2. What failed or remains unsupported?",
        "",
        f"- **Utility routing is weak.** At a 10% rerun budget, the best learned model reaches {_pct(learned10['accuracy'])}; "
        f"entropy reaches {_pct(entropy10['accuracy'])}, always-FP16 reaches {_pct(utility['static']['always_fp16'])}, and the oracle reaches {_pct(oracle10['accuracy'])}. "
        f"At 20%, the oracle reaches {_pct(oracle20['accuracy'])}. The opportunity exists, but the current features do not identify it reliably.",
        f"- **Early-token control is not ready.** The best beneficial-switch ROC-AUC is {prefix_rows[0][2]:.3f} at 16 tokens, "
        f"{prefix_rows[1][2]:.3f} at 32, and {prefix_rows[2][2]:.3f} at 64. Even the best 10% prefix router reaches only {_pct(max(row[3] for row in prefix_rows))}.",
        "- **Longer does not yet mean more reasoning.** BNB4 emits more tokens overall, but heuristic arithmetic-expression and self-correction counts do not rise systematically in rescue cases.",
        f"- **The clean trajectory contrast is suggestive, not conclusive.** Clean BNB4 rescues are {length['clean_by_outcome']['fp_wrong_q_correct']['mean_token_delta_quant_minus_fp']:+.2f} tokens longer on average, "
        f"while clean BNB4-induced failures are {length['clean_by_outcome']['fp_correct_q_wrong']['mean_token_delta_quant_minus_fp']:+.2f}; the groups are small and their individual confidence intervals include zero.",
        "- **Naive first-divergence position is not mechanistic evidence.** Independent greedy outputs often differ immediately because tokenization/wording paths split; a same-prefix logit comparison is needed.",
        "- **Efficiency is currently negative on this hardware.** BNB4 is memory-saving but slower at batch-one decoding in the recorded RTX 3090 run. No speed claim is justified.",
    ]
    if temperature_complete:
        t07 = temperature["temperatures"].get("0.7", {})
        controlled = temperature.get("controlled_correctness_model", {}).get(
            "delta_when_adding_precision", {}
        )
        lines.extend(
            [
                f"- **H3 is weakened.** At temperature 0.7, any of three FP16 samples solves {_pct(t07.get('quant_rescue_coverage', 0.0))} of the BNB4-rescue prompts; majority vote reproduces {_pct(t07.get('quant_rescue_majority_coverage', 0.0))}. "
                f"Adding precision mode to the grouped controlled model changes ROC-AUC by only {controlled.get('roc_auc', 0.0):+.4f} and log loss by {controlled.get('log_loss', 0.0):+.4f}. Precision does not add predictive value in this diagnostic.",
                f"- **Temperature is not fully equivalent either.** The temperature-0.7 rescue-set Jaccard is only {_pct(t07.get('quant_vs_temperature_rescue_jaccard', 0.0))}, and its mean output is {t07.get('token_delta_vs_quant_greedy', 0.0):+.2f} tokens relative to BNB4 greedy. Sampling reproduces correctness opportunities more than it reproduces the BNB4 trajectory-length shift.",
            ]
        )
    else:
        lines.extend(["- **H3 is unanswered.** No FP16 temperature artifact is present, so we cannot yet claim precision is distinct from ordinary decoding noise."])
    lines.extend(
        [
            "",
            "## 3. Strongest thesis framing",
            "",
            f"Choose **{framing}** for now.",
            "",
            "The phrase “precision as a control variable” should remain a hypothesis in the title/abstract until two things happen: "
            "(1) FP16 temperature sampling fails to reproduce BNB4’s rescue/failure sets, and (2) the signed trajectory effect replicates on another dataset, model, or quantizer. "
            "If those pass, promote to framing A. If complementarity replicates but mechanism/routing remains weak, use framing B or C.",
            "",
            "## 4. Continue or not?",
            "",
            (
                "Continue only as a bounded empirical replication, not as an adaptive-precision systems build."
                if temperature_complete
                else "Continue for one focused evidence sprint, not as an open-ended systems build."
            ),
            "",
            (
                "1. Replicate greedy FP16/BNB4 on one genuinely independent setting: preferably a small MATH subset or Qwen2.5-3B, not another GSM8K slice alone."
                if temperature_complete
                else "1. Run 100 test prompts at FP16 temperatures 0.3 and 0.7 with three samples each."
            ),
            (
                "2. Pre-register the same paired outcomes, clean filter, token-length contrast, and oracle selector before looking at the replication."
                if temperature_complete
                else "2. Compare BNB4 rescue-set overlap, answer diversity, length, and majority-vote outcomes."
            ),
            "3. Stop token-level implementation work unless a prefix router beats entropy and shows useful net gain before token 64.",
            "",
            "### Kill criteria",
            "",
            "Pivot away if BNB4-only wins largely coincide with ordinary FP16 sampling rescues, the paired complementarity collapses in the independent replication, or clean BNB4-only wins become negligible. "
            "If the effects replicate but routing stays weak, keep the project as a careful empirical trajectory study rather than an adaptive-inference thesis.",
            "",
            "## Bottom line",
            "",
            (
                "Numerical precision changes the observed greedy trajectory, but ordinary FP16 sampling recovers much of the same correctness complementarity. "
                "The strong control-variable thesis is currently unsupported. One independent replication is justified; custom kernels and a broad adaptive-precision claim are not."
                if temperature_complete
                else "There is a credible phenomenon: numerical precision changes which GSM8K problems this model solves. There is not yet evidence that precision is a practically controllable reasoning knob. "
                "The project is worth the next two discriminating experiments; it is not yet worth custom kernels or a broad PhD-level claim."
            ),
            "",
        ]
    )
    if temperature_complete:
        temperature_lines = [
            "## Temperature-versus-quantization control",
            "",
            "| Temperature | Per-completion accuracy | Empirical pass@k | Majority accuracy | Any-sample BNB4 rescue coverage | Majority BNB4 rescue coverage | Rescue Jaccard | Mean tokens |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for value, row in sorted(temperature["temperatures"].items(), key=lambda item: float(item[0])):
            temperature_lines.append(
                f"| {value} | {_pct(row['per_completion_accuracy'])} | {_pct(row['pass_at_k_empirical'])} | "
                f"{_pct(row['majority_vote_accuracy'])} | {_pct(row['quant_rescue_coverage']) if row['quant_rescue_coverage'] is not None else 'n/a'} | "
                f"{_pct(row.get('quant_rescue_majority_coverage')) if row.get('quant_rescue_majority_coverage') is not None else 'n/a'} | "
                f"{_pct(row['quant_vs_temperature_rescue_jaccard']) if row['quant_vs_temperature_rescue_jaccard'] is not None else 'n/a'} | {row['mean_generation_tokens']:.2f} |"
            )
        temperature_lines.extend(
            [
                "",
                "Interpret overlap jointly with the grouped controlled-correctness diagnostic. Any-sample pass@k is expected to rise with repeated sampling and is not itself evidence of equivalence.",
                "",
            ]
        )
        insert_at = lines.index("## 3. Strongest thesis framing")
        lines[insert_at:insert_at] = temperature_lines
    decision = {
        "recommendation": recommendation,
        "framing": framing,
        "temperature_control_complete": temperature_complete,
        "independent_replication_complete": another_model_or_dataset,
        "learned_beats_entropy_at_10pct": learned_beats_entropy,
        "learned_accuracy_at_10pct": learned10["accuracy"],
        "entropy_accuracy_at_10pct": entropy10["accuracy"],
        "oracle_accuracy_at_10pct": oracle10["accuracy"],
        "next_gate": (
            "One pre-registered independent model or dataset replication"
            if temperature_complete
            else "FP16 temperature baseline on 100 prompts, then one independent replication"
        ),
    }
    return "\n".join(lines), decision


def load_optional_json(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    return read_json(target) if target.exists() else None
