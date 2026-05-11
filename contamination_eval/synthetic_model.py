"""Synthetic model definitions and memorization sampling."""

from __future__ import annotations

from typing import Any

import numpy as np


ABILITIES = [-1.0, 0.0, 1.0]
CONTAM_DEGREES = [0.0, 0.25, 0.5, 0.75, 1.0]
ALPHA = 0.5
SEED = 0


def generate_models() -> list[dict[str, float | str]]:
    """Generate the full grid of synthetic models."""
    models: list[dict[str, float | str]] = []
    for ability in ABILITIES:
        for contam_degree in CONTAM_DEGREES:
            models.append(
                {
                    "name": f"ability_{ability}_contam_{contam_degree}",
                    "ability": ability,
                    "contam_degree": contam_degree,
                }
            )
    return models


def memorization_probability(
    model: dict[str, Any],
    item: dict[str, Any],
    alpha: float = ALPHA,
) -> float:
    """Compute p_mem for one model-item pair."""
    return float(model["contam_degree"] * (1 - np.exp(-alpha * item["replica_count"])))


def sample_memorization(
    models: list[dict[str, Any]],
    eval_items: list[dict[str, Any]],
    seed: int = SEED,
) -> tuple[dict[tuple[str, int], bool], dict[tuple[str, int], float]]:
    """Sample memorized booleans and keep the underlying probabilities."""
    rng = np.random.default_rng(seed + 2)
    memorized: dict[tuple[str, int], bool] = {}
    p_mem: dict[tuple[str, int], float] = {}

    for model in models:
        model_name = str(model["name"])
        for item in eval_items:
            key = (model_name, int(item["id"]))
            probability = memorization_probability(model, item)
            p_mem[key] = probability
            memorized[key] = bool(rng.random() < probability)

    return memorized, p_mem
