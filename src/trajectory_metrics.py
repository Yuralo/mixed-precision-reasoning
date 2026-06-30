"""Heuristic, judge-free measurements of reasoning-output structure."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .answer_extraction import extract_answer, extract_explicit_answer


CORRECTION_PATTERN = re.compile(
    r"\b(wait|actually|check|recheck|recalculate|re-evaluate|mistake|correction|instead|however|let me|verify)\b",
    re.IGNORECASE,
)
ARITHMETIC_PATTERN = re.compile(
    r"[-+]?\d[\d,]*(?:\.\d+)?\s*(?:[+\-*/×÷]|\b(?:times|plus|minus|divided by)\b)\s*[-+]?\d[\d,]*(?:\.\d+)?",
    re.IGNORECASE,
)
EQUATION_PATTERN = re.compile(r"([^\n=]{1,100})=\s*([-+]?[$€£]?\s*[\d,]+(?:\.\d+)?)")
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
NUMBERED_STEP_PATTERN = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*])\s+")
FINAL_MARKER_PATTERN = re.compile(r"(?:FINAL_ANSWER\s*:|final\s+answer\s*:|####|\\boxed\s*\{)", re.IGNORECASE)


def _normalized_lines(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", line.strip().lower()) for line in text.splitlines() if line.strip()]


def _repetition_score(text: str) -> float:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < 8:
        return 0.0
    ngrams = [tuple(words[index : index + 4]) for index in range(len(words) - 3)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def analyze_trajectory(text: str, generated_tokens: int | None = None) -> dict[str, Any]:
    stripped = text.strip()
    lines = _normalized_lines(stripped)
    explicit_answer = extract_explicit_answer(stripped)
    fallback_answer = extract_answer(stripped)
    markers = list(FINAL_MARKER_PATTERN.finditer(stripped))
    marker_position = markers[-1].start() if markers else None
    trailing_chars = len(stripped) - markers[-1].end() if markers else None
    sentence_parts = [part for part in re.split(r"(?:[.!?]+\s+|\n+)", stripped) if part.strip()]
    arithmetic = ARITHMETIC_PATTERN.findall(stripped)
    equations = list(EQUATION_PATTERN.finditer(stripped))
    numeric_values = [match.group(0).replace(",", "") for match in NUMBER_PATTERN.finditer(stripped)]
    line_counts = Counter(lines)
    duplicate_lines = sum(count - 1 for count in line_counts.values() if count > 1)
    last_rhs = extract_answer(equations[-1].group(2)) if equations else None
    answer_occurrences = numeric_values[:-1].count(explicit_answer) if explicit_answer else 0
    terminal_line = lines[-1] if lines else ""
    terminal_has_expression = bool(ARITHMETIC_PATTERN.search(terminal_line))
    return {
        "generated_tokens": generated_tokens,
        "character_count": len(stripped),
        "line_count": len(lines),
        "sentence_count": len(sentence_parts),
        "numbered_step_count": len(NUMBERED_STEP_PATTERN.findall(stripped)),
        "arithmetic_expression_count": len(arithmetic),
        "equation_count": len(equations),
        "numeric_value_count": len(numeric_values),
        "distinct_numeric_value_count": len(set(numeric_values)),
        "self_correction_count": len(CORRECTION_PATTERN.findall(stripped)),
        "repetition_score_4gram": _repetition_score(stripped),
        "duplicate_line_count": duplicate_lines,
        "explicit_answer": explicit_answer,
        "fallback_answer": fallback_answer,
        "has_explicit_answer": explicit_answer is not None,
        "final_answer_character_position": marker_position,
        "normalized_final_answer_position": marker_position / len(stripped) if marker_position is not None and stripped else None,
        "characters_after_final_marker": trailing_chars,
        "terminal_line_has_arithmetic_expression": terminal_has_expression,
        "final_answer_seen_earlier_count": answer_occurrences,
        "last_equation_rhs": last_rhs,
        "last_equation_conflicts_with_final": bool(
            explicit_answer is not None and last_rhs is not None and explicit_answer != last_rhs
        ),
    }


def paired_metric_deltas(fp_metrics: dict[str, Any], quant_metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "generated_tokens",
        "line_count",
        "sentence_count",
        "numbered_step_count",
        "arithmetic_expression_count",
        "equation_count",
        "numeric_value_count",
        "distinct_numeric_value_count",
        "self_correction_count",
        "repetition_score_4gram",
        "normalized_final_answer_position",
    )
    result = {}
    for key in keys:
        fp_value, quant_value = fp_metrics.get(key), quant_metrics.get(key)
        result[f"delta_{key}"] = (
            quant_value - fp_value if fp_value is not None and quant_value is not None else None
        )
    return result
