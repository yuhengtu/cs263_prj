"""Variance quantification for evaluation benchmarks (Madaan et al., 2024)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

ArrayLike = Union[Sequence[float], np.ndarray]


class VarianceInputError(ValueError):
    pass


@dataclass(frozen=True)
class VarianceDecomposition:
    total_variance: float
    factor_variances: Dict[str, float]
    factor_fractions: Dict[str, float]
    residual_variance: float
    n_observations: int


def _as_2d_array(scores: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise VarianceInputError(f"{name} must be a non-empty 2-D array.")
    return arr


def seed_variance(seed_scores: ArrayLike) -> Tuple[float, float]:
    """Madaan eq. for E(S, M): mean and std across init seeds."""
    arr = np.asarray(seed_scores, dtype=float).reshape(-1)
    if arr.size < 2:
        raise VarianceInputError("Need at least two seed scores.")
    return float(arr.mean()), float(arr.std(ddof=1))


def analytic_binomial_ci(score: float, n_items: int, z: float = 1.96) -> float:
    """Madaan eq. for analytic 95% CI half-width on a discrete metric."""
    if not (0.0 <= score <= 1.0) or n_items <= 0:
        raise VarianceInputError("Need 0<=score<=1 and n_items>0.")
    return float(z * np.sqrt(score * (1 - score) / n_items))


def signal_to_noise(seed_scores: ArrayLike) -> float:
    """SNR = mean / std across seeds (Madaan §3.2 Disc/Cont SNR)."""
    mean, std = seed_variance(seed_scores)
    if std == 0.0:
        return float("inf")
    return float(mean / std)


def monotonicity(checkpoint_scores: ArrayLike) -> float:
    """Kendall-tau between checkpoint scores and a monotone ladder."""
    arr = np.asarray(checkpoint_scores, dtype=float).reshape(-1)
    if arr.size < 2:
        raise VarianceInputError("Need at least two checkpoints.")
    target = np.arange(arr.size)
    tau, _ = stats.kendalltau(arr, target)
    return float(tau)


def factor_variance(score_matrix: ArrayLike, axis: int = 0) -> float:
    arr = _as_2d_array(score_matrix, "score_matrix")
    if axis not in (0, 1):
        raise VarianceInputError("axis must be 0 or 1.")
    means = arr.mean(axis=1 - axis)
    return 0.0 if means.size < 2 else float(np.var(means, ddof=1))


def variance_decomposition(scores: Dict[str, ArrayLike]) -> VarianceDecomposition:
    if not scores:
        raise VarianceInputError("scores must contain at least one factor.")
    factor_var: Dict[str, float] = {}
    n_obs = 0
    for name, values in scores.items():
        arr = np.asarray(values, dtype=float).reshape(-1)
        factor_var[name] = float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0
        n_obs += arr.size
    residual = factor_var.pop("residual", 0.0)
    total = sum(factor_var.values()) + residual
    fractions = {n: v / total for n, v in factor_var.items()} if total > 0 \
        else {n: 0.0 for n in factor_var}
    return VarianceDecomposition(float(total), factor_var, fractions,
                                 float(residual), n_obs)


def two_way_decomposition(
    score_matrix: ArrayLike,
    factor_names: Tuple[str, str] = ("row", "col"),
) -> VarianceDecomposition:
    arr = _as_2d_array(score_matrix, "score_matrix")
    n_rows, n_cols = arr.shape
    grand = float(arr.mean())
    row_means = arr.mean(axis=1)
    col_means = arr.mean(axis=0)
    var_rows = float(np.mean((row_means - grand) ** 2)) if n_rows > 1 else 0.0
    var_cols = float(np.mean((col_means - grand) ** 2)) if n_cols > 1 else 0.0
    var_total = float(np.mean((arr - grand) ** 2))
    residual = max(0.0, var_total - var_rows - var_cols)
    factor_var = {factor_names[0]: var_rows, factor_names[1]: var_cols}
    fractions = {n: v / var_total for n, v in factor_var.items()} if var_total > 0 \
        else {n: 0.0 for n in factor_var}
    return VarianceDecomposition(var_total, factor_var, fractions, residual, arr.size)


def subsample_variance(
    item_scores: ArrayLike,
    subset_size: Optional[int] = None,
    n_subsamples: int = 1_000,
    seed: int = 0,
) -> Tuple[float, float, np.ndarray]:
    arr = np.asarray(item_scores, dtype=float).reshape(-1)
    n = arr.size
    if n == 0:
        raise VarianceInputError("item_scores must be non-empty.")
    if subset_size is None:
        subset_size = max(1, n // 2)
    if not (1 <= subset_size <= n):
        raise VarianceInputError("Need 1 <= subset_size <= n.")
    rng = np.random.default_rng(seed)
    means = np.empty(n_subsamples, dtype=float)
    for b in range(n_subsamples):
        means[b] = arr[rng.choice(n, size=subset_size, replace=False)].mean()
    return float(means.mean()), float(means.std(ddof=1)), means


def score_difference_distribution(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
) -> Dict[str, float]:
    a = np.asarray(scores_a, dtype=float).reshape(-1)
    b = np.asarray(scores_b, dtype=float).reshape(-1)
    if a.size != b.size or a.size == 0:
        raise VarianceInputError("scores_a and scores_b must align and be non-empty.")
    diff = a - b
    return {
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1) if diff.size > 1 else 0.0),
        "frac_a_wins": float(np.mean(diff > 0)),
        "frac_b_wins": float(np.mean(diff < 0)),
        "frac_tie": float(np.mean(diff == 0)),
        "n": int(a.size),
    }


def cohens_d(scores_a: ArrayLike, scores_b: ArrayLike) -> float:
    a = np.asarray(scores_a, dtype=float).reshape(-1)
    b = np.asarray(scores_b, dtype=float).reshape(-1)
    if a.size < 2 or b.size < 2:
        raise VarianceInputError("Each array needs at least two items.")
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
                     / (a.size + b.size - 2))
    return 0.0 if pooled == 0.0 else float((a.mean() - b.mean()) / pooled)
