"""Small reasoning datasets with an offline five-example GSM8K-style fixture."""

from __future__ import annotations

from typing import Any


TINY_GSM8K = [
    {"question": "Mia has 3 apples and buys 4 more. How many apples does she have?", "answer": "She has 3 + 4 = 7 apples. #### 7"},
    {"question": "A box has 12 pencils. Four are removed. How many remain?", "answer": "12 - 4 = 8. #### 8"},
    {"question": "Five bags each contain 6 marbles. How many marbles are there?", "answer": "5 * 6 = 30. #### 30"},
    {"question": "Sam reads 9 pages on Monday and 11 on Tuesday. How many pages total?", "answer": "9 + 11 = 20. #### 20"},
    {"question": "A 24 dollar bill is split equally among 3 friends. How much does each get?", "answer": "24 / 3 = 8. #### 8"},
]


def reasoning_prompt(question: str) -> str:
    return (
        "Solve the problem step by step. End with the final numeric answer in the "
        "format '#### answer'.\n\nProblem: " + question
    )


def load_reasoning_dataset(
    name: str = "gsm8k",
    split: str = "test",
    subset: str = "main",
    limit: int | None = None,
    tiny: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if name != "gsm8k":
        raise ValueError(f"Unsupported dataset {name!r}; the MVP currently supports gsm8k")

    if tiny:
        source = TINY_GSM8K
    else:
        from datasets import load_dataset

        source = load_dataset("openai/gsm8k", subset, split=split)

    n = len(source) if limit is None else min(limit, len(source))
    rows = []
    for index in range(n):
        item = source[index]
        rows.append(
            {
                "example_id": f"{name}-{split}-{index}",
                "dataset": name,
                "split": split,
                "question": item["question"],
                "prompt": reasoning_prompt(item["question"]),
                "reference": item["answer"],
                "metadata": {"source_index": index, "seed": seed, "tiny": tiny},
            }
        )
    return rows
