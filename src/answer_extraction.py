"""Robust-enough numeric extraction for GSM8K-style final answers."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


HASH_ANSWER = re.compile(r"####\s*([-+]?[$€£]?\s*[\d,]+(?:\.\d+)?)")
NUMBER = re.compile(r"[-+]?[$€£]?\s*[\d,]+(?:\.\d+)?")


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
    hashes = HASH_ANSWER.findall(text)
    if hashes:
        return normalize_numeric_answer(hashes[-1])
    numbers = NUMBER.findall(text)
    return normalize_numeric_answer(numbers[-1]) if numbers else None


def is_correct(generation: str, reference: str) -> bool:
    predicted = extract_answer(generation)
    gold = extract_answer(reference)
    return predicted is not None and gold is not None and predicted == gold
