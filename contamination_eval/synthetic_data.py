"""Synthetic numeric multiple-choice QA data and corpus construction."""

from __future__ import annotations

from typing import Any

import numpy as np


N_EVAL = 200
N_CORPUS = 1000
VOCAB_SIZE = 500

PROMPT_LEN = 6
ANSWER_LEN = 3
NUM_OPTIONS = 4

CONTAMINATION_RATE = 0.3
R_VALUES = [1, 3, 10]

SEED = 0


def answer_from_prompt(prompt: list[int], vocab_size: int = VOCAB_SIZE) -> list[int]:
    """Apply the synthetic QA rule to generate the correct answer."""
    return [
        (prompt[0] + prompt[1]) % vocab_size,
        (prompt[2] * 3 + 7) % vocab_size,
        (prompt[3] - prompt[4]) % vocab_size,
    ]


def _random_answer(rng: np.random.Generator) -> list[int]:
    return rng.integers(0, VOCAB_SIZE, size=ANSWER_LEN).astype(int).tolist()


def generate_eval_items(
    n_eval: int = N_EVAL,
    contamination_rate: float = CONTAMINATION_RATE,
    r_values: list[int] | None = None,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Generate synthetic numeric multiple-choice eval items."""
    rng = np.random.default_rng(seed)
    r_values = R_VALUES if r_values is None else r_values
    items: list[dict[str, Any]] = []

    for item_id in range(n_eval):
        prompt = rng.integers(0, VOCAB_SIZE, size=PROMPT_LEN).astype(int).tolist()
        answer = answer_from_prompt(prompt)

        wrong_options: list[list[int]] = []
        seen = {tuple(answer)}
        while len(wrong_options) < NUM_OPTIONS - 1:
            option = _random_answer(rng)
            option_tuple = tuple(option)
            if option_tuple in seen:
                continue
            seen.add(option_tuple)
            wrong_options.append(option)

        correct_idx = int(rng.integers(0, NUM_OPTIONS))
        options = wrong_options[:]
        options.insert(correct_idx, answer)

        contaminated = bool(rng.random() < contamination_rate)
        replica_count = int(rng.choice(r_values)) if contaminated else 0

        items.append(
            {
                "id": item_id,
                "prompt": prompt,
                "answer": answer,
                "options": options,
                "correct_idx": correct_idx,
                "difficulty": float(rng.normal()),
                "contaminated": contaminated,
                "replica_count": replica_count,
            }
        )

    return items


def serialize_item(item: dict[str, Any]) -> list[int]:
    """Serialize an eval item as prompt + options + answer."""
    serialized = list(item["prompt"])
    for option in item["options"]:
        serialized.extend(option)
    serialized.extend(item["answer"])
    return serialized


def generate_background_doc(rng: np.random.Generator) -> list[int]:
    """Generate one random background corpus document."""
    serialized_len = PROMPT_LEN + NUM_OPTIONS * ANSWER_LEN + ANSWER_LEN
    return rng.integers(0, VOCAB_SIZE, size=serialized_len).astype(int).tolist()


def build_corpus(
    eval_items: list[dict[str, Any]],
    n_corpus: int = N_CORPUS,
    seed: int = SEED,
) -> list[list[int]]:
    """Build background corpus plus leaked replicas of contaminated eval items."""
    rng = np.random.default_rng(seed + 1)
    corpus = [generate_background_doc(rng) for _ in range(n_corpus)]

    for item in eval_items:
        if item["contaminated"]:
            for _ in range(item["replica_count"]):
                corpus.append(serialize_item(item))

    return corpus
