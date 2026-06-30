"""Quality checks that separate quantization effects from generation/evaluation artifacts."""

from __future__ import annotations

from typing import Any

from .answer_extraction import extract_answer, extract_explicit_answer
from .metrics import comparison_summary


def _run_quality(rows: list[dict[str, Any]], assumed_max_new_tokens: int) -> dict[str, Any]:
    n = len(rows)
    hit_cap = [
        bool(row.get("hit_max_new_tokens", row.get("generation_tokens", 0) >= assumed_max_new_tokens))
        for row in rows
    ]
    strict = [extract_explicit_answer(row.get("generation", "")) for row in rows]
    fallback = [extract_answer(row.get("generation", "")) for row in rows]
    return {
        "num_examples": n,
        "accuracy": sum(bool(row.get("correct")) for row in rows) / n if n else 0.0,
        "hit_max_new_tokens_count": sum(hit_cap),
        "hit_max_new_tokens_rate": sum(hit_cap) / n if n else 0.0,
        "explicit_answer_count": sum(answer is not None for answer in strict),
        "explicit_answer_rate": sum(answer is not None for answer in strict) / n if n else 0.0,
        "no_numeric_answer_count": sum(answer is None for answer in fallback),
        "no_numeric_answer_rate": sum(answer is None for answer in fallback) / n if n else 0.0,
        "mean_generation_tokens": (
            sum(int(row.get("generation_tokens", 0)) for row in rows) / n if n else 0.0
        ),
    }


def generation_diagnostics(
    fp_rows: list[dict[str, Any]],
    quant_rows: list[dict[str, Any]],
    assumed_max_new_tokens: int = 256,
) -> dict[str, Any]:
    fp_by_id = {row["example_id"]: row for row in fp_rows}
    quant_by_id = {row["example_id"]: row for row in quant_rows}
    shared = sorted(fp_by_id.keys() & quant_by_id.keys())
    clean_comparisons = []
    strict_comparisons = []
    both_not_truncated = 0
    both_explicit = 0
    for example_id in shared:
        fp, quant = fp_by_id[example_id], quant_by_id[example_id]
        fp_cap = bool(fp.get("hit_max_new_tokens", fp.get("generation_tokens", 0) >= assumed_max_new_tokens))
        q_cap = bool(quant.get("hit_max_new_tokens", quant.get("generation_tokens", 0) >= assumed_max_new_tokens))
        fp_hash = extract_explicit_answer(fp.get("generation", ""))
        q_hash = extract_explicit_answer(quant.get("generation", ""))
        gold = extract_answer(fp.get("reference", ""))
        if not fp_cap and not q_cap:
            both_not_truncated += 1
            clean_comparisons.append(
                _comparison_row(example_id, fp_hash == gold, q_hash == gold, fp_hash, q_hash)
            )
        if fp_hash is not None and q_hash is not None:
            both_explicit += 1
            strict_comparisons.append(
                _comparison_row(example_id, fp_hash == gold, q_hash == gold, fp_hash, q_hash)
            )
    return {
        "assumed_max_new_tokens_for_legacy_rows": assumed_max_new_tokens,
        "fp": _run_quality(fp_rows, assumed_max_new_tokens),
        "quant": _run_quality(quant_rows, assumed_max_new_tokens),
        "matched_examples": len(shared),
        "both_not_truncated_count": both_not_truncated,
        "both_explicit_answer_count": both_explicit,
        "strict_explicit_answer_summary": comparison_summary(strict_comparisons),
        "nontruncated_strict_summary": comparison_summary(clean_comparisons),
    }


def _comparison_row(example_id: str, fp_correct: bool, quant_correct: bool, fp_answer, quant_answer) -> dict:
    if fp_correct and quant_correct:
        label = "fp_correct_q_correct"
    elif fp_correct:
        label = "fp_correct_q_wrong"
    elif quant_correct:
        label = "fp_wrong_q_correct"
    else:
        label = "fp_wrong_q_wrong"
    return {
        "example_id": example_id,
        "fp_correct": fp_correct,
        "quant_correct": quant_correct,
        "fp_answer": fp_answer,
        "quant_answer": quant_answer,
        "comparison_label": label,
    }
