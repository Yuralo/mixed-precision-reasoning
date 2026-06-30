"""Join matched FP and quantized generations and label quantization-induced failures."""

from __future__ import annotations

from .metrics import comparison_summary


def _label(fp_correct: bool, q_correct: bool) -> str:
    if fp_correct and q_correct:
        return "fp_correct_q_correct"
    if fp_correct:
        return "fp_correct_q_wrong"
    if q_correct:
        return "fp_wrong_q_correct"
    return "fp_wrong_q_wrong"


def compare_runs(fp_rows: list[dict], quant_rows: list[dict]) -> tuple[list[dict], dict]:
    fp_by_id = {row["example_id"]: row for row in fp_rows}
    q_by_id = {row["example_id"]: row for row in quant_rows}
    if fp_by_id.keys() != q_by_id.keys():
        missing_q = sorted(fp_by_id.keys() - q_by_id.keys())
        missing_fp = sorted(q_by_id.keys() - fp_by_id.keys())
        raise ValueError(f"Run IDs do not match; missing quant={missing_q}, missing fp={missing_fp}")

    rows = []
    for example_id in fp_by_id:
        fp, quant = fp_by_id[example_id], q_by_id[example_id]
        rows.append(
            {
                "example_id": example_id,
                "dataset": fp["dataset"],
                "split": fp["split"],
                "question": fp["question"],
                "reference": fp["reference"],
                "reference_answer": fp["reference_answer"],
                "fp_generation": fp["generation"],
                "quant_generation": quant["generation"],
                "fp_answer": fp["predicted_answer"],
                "quant_answer": quant["predicted_answer"],
                "fp_correct": bool(fp["correct"]),
                "quant_correct": bool(quant["correct"]),
                "comparison_label": _label(bool(fp["correct"]), bool(quant["correct"])),
                "answer_flipped": fp["predicted_answer"] != quant["predicted_answer"],
                "prompt_tokens": quant["prompt_tokens"],
                "fp_generation_tokens": fp["generation_tokens"],
                "quant_generation_tokens": quant["generation_tokens"],
                "fp_hit_max_new_tokens": fp.get("hit_max_new_tokens"),
                "quant_hit_max_new_tokens": quant.get("hit_max_new_tokens"),
                "fp_has_hash_answer": fp.get("has_hash_answer"),
                "quant_has_hash_answer": quant.get("has_hash_answer"),
                "fp_stop_reason": fp.get("stop_reason"),
                "quant_stop_reason": quant.get("stop_reason"),
                "fp_model": fp["model"],
                "quant_model": quant["model"],
            }
        )
    return rows, comparison_summary(rows)
