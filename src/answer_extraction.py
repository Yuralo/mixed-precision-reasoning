"""Robust-enough numeric extraction for GSM8K-style final answers."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


NUMERIC_FRAGMENT = r"[-+]?[$€£]?\s*[\d,]+(?:\.\d+)?"
HASH_ANSWER = re.compile(rf"####\s*({NUMERIC_FRAGMENT})")
FINAL_LABEL_ANSWER = re.compile(
    rf"(?:final\s+answer|answer)\s*(?:is\s*|[:=]\s*)({NUMERIC_FRAGMENT})",
    re.IGNORECASE,
)
BOXED_ANSWER = re.compile(rf"\\boxed\s*\{{\s*({NUMERIC_FRAGMENT})\s*\}}")
BOLD_NUMBER = re.compile(
    rf"\*\*\s*({NUMERIC_FRAGMENT})(?:\s+[A-Za-z]+(?:/[A-Za-z]+)?)?\s*\*\*"
)
NUMBER = re.compile(NUMERIC_FRAGMENT)


def normalize_numeric_answer(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"[$€£,\s]", "", value).rstrip(".")
    try:
        number = Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return cleaned.lower() or None
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", ""} else normalized


def extract_answer(text: str) -> str | None:
    explicit = extract_explicit_answer(text)
    if explicit is not None:
        return explicit
    numbers = NUMBER.findall(text)
    return normalize_numeric_answer(numbers[-1]) if numbers else None


def extract_hash_answer(text: str) -> str | None:
    """Return only an explicitly marked GSM8K answer, without numeric fallback."""
    hashes = HASH_ANSWER.findall(text)
    return normalize_numeric_answer(hashes[-1]) if hashes else None


def extract_explicit_answer(text: str) -> str | None:
    """Extract a deliberately finalized answer without falling back to any number.

    Supported forms reflect common instruct-model behavior: GSM8K ``####``,
    ``Final answer:``, LaTeX ``\\boxed{}``, and a bold numeric result near the end.
    """
    marked = extract_marked_answer(text)
    if marked is not None:
        return marked
    tail = text.strip()[-400:]
    bold_matches = list(BOLD_NUMBER.finditer(tail))
    if bold_matches and len(tail) - bold_matches[-1].end() <= 80:
        return normalize_numeric_answer(bold_matches[-1].group(1))
    return None


def extract_marked_answer(text: str) -> str | None:
    """Extract only strong markers safe for online generation stopping."""
    for pattern in (HASH_ANSWER, FINAL_LABEL_ANSWER, BOXED_ANSWER):
        matches = pattern.findall(text)
        if matches:
            return normalize_numeric_answer(matches[-1])
    return None


def is_correct(generation: str, reference: str) -> bool:
    predicted = extract_answer(generation)
    gold = extract_answer(reference)
    return predicted is not None and gold is not None and predicted == gold
