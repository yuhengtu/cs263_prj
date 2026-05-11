"""Statistical error bars for benchmark accuracy (Miller, 2024)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

ArrayLike = Union[Sequence[float], np.ndarray]


class ErrorBarInputError(ValueError):
    pass


@dataclass(frozen=True)
class IntervalResult:
    estimate: float
    lower: float
    upper: float
    method: str
    alpha: float
    n: int


def _as_binary_array(values: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ErrorBarInputError(f"{name} must contain at least one item.")
    if not np.all(np.isin(arr, [0.0, 1.0])):
        raise ErrorBarInputError(f"{name} must be binary (0/1).")
    return arr.astype(int)


def _validate_count(k: int, n: int) -> None:
    if n <= 0 or not (0 <= k <= n):
        raise ErrorBarInputError("Require n > 0 and 0 <= k <= n.")


def clt_standard_error(item_scores: ArrayLike) -> Tuple[float, float, int]:
    """Mean and CLT standard error of the per-item scores (Miller eq. 1)."""
    arr = np.asarray(item_scores, dtype=float).reshape(-1)
    if arr.size < 2:
        raise ErrorBarInputError("Need at least two scores.")
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(arr.size))
    return mean, se, int(arr.size)


def clt_interval(item_scores: ArrayLike, alpha: float = 0.05) -> IntervalResult:
    mean, se, n = clt_standard_error(item_scores)
    z = stats.norm.ppf(1 - alpha / 2)
    return IntervalResult(mean, mean - z * se, mean + z * se, "clt", alpha, n)


def wald_interval(k: int, n: int, alpha: float = 0.05) -> IntervalResult:
    _validate_count(k, n)
    p = k / n
    z = stats.norm.ppf(1 - alpha / 2)
    half = z * np.sqrt(p * (1 - p) / n)
    return IntervalResult(p, max(0.0, p - half), min(1.0, p + half), "wald", alpha, n)


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> IntervalResult:
    _validate_count(k, n)
    p = k / n
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return IntervalResult(p, max(0.0, centre - half), min(1.0, centre + half),
                          "wilson", alpha, n)


def clopper_pearson_interval(k: int, n: int, alpha: float = 0.05) -> IntervalResult:
    _validate_count(k, n)
    p = k / n
    lower = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    upper = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return IntervalResult(p, float(lower), float(upper), "clopper_pearson", alpha, n)


def bootstrap_interval(
    item_scores: ArrayLike,
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> IntervalResult:
    arr = np.asarray(item_scores, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ErrorBarInputError("item_scores must contain at least one item.")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    n = arr.size
    for b in range(n_bootstrap):
        means[b] = arr[rng.integers(0, n, size=n)].mean()
    lower = float(np.quantile(means, alpha / 2))
    upper = float(np.quantile(means, 1 - alpha / 2))
    return IntervalResult(float(arr.mean()), lower, upper, "bootstrap", alpha, n)


def clustered_standard_error(
    item_scores: ArrayLike,
    clusters: ArrayLike,
) -> Tuple[float, float]:
    """Cluster-robust SE of the mean (Miller eq. 4)."""
    x = np.asarray(item_scores, dtype=float).reshape(-1)
    g = np.asarray(clusters).reshape(-1)
    if x.size != g.size or x.size == 0:
        raise ErrorBarInputError("item_scores and clusters must align and be non-empty.")
    n = x.size
    mean = float(x.mean())
    centred = x - mean
    cluster_sum_sq = sum(centred[g == c].sum() ** 2 for c in np.unique(g))
    return mean, float(np.sqrt(cluster_sum_sq / (n**2)))


def paired_difference_interval(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    alpha: float = 0.05,
) -> IntervalResult:
    """Question-level paired CI for accuracy difference (Miller §4)."""
    a = np.asarray(scores_a, dtype=float).reshape(-1)
    b = np.asarray(scores_b, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 2:
        raise ErrorBarInputError("Need at least two paired items of equal length.")
    diff = a - b
    n = diff.size
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n))
    z = stats.norm.ppf(1 - alpha / 2)
    return IntervalResult(mean, mean - z * se, mean + z * se,
                          "paired_difference", alpha, n)


def mcnemar_test(scores_a: ArrayLike, scores_b: ArrayLike) -> Tuple[float, float]:
    a = _as_binary_array(scores_a, "scores_a")
    b = _as_binary_array(scores_b, "scores_b")
    if a.size != b.size:
        raise ErrorBarInputError("scores_a and scores_b must have equal length.")
    b01 = int(((a == 1) & (b == 0)).sum())
    b10 = int(((a == 0) & (b == 1)).sum())
    n_disc = b01 + b10
    if n_disc == 0:
        return float(b01), 1.0
    pvalue = float(stats.binomtest(b01, n_disc, p=0.5, alternative="two-sided").pvalue)
    return float(b01), pvalue


def required_n_for_difference(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Sample size for a two-proportion z-test (Miller §5)."""
    if not (0 < p1 < 1) or not (0 < p2 < 1) or p1 == p2:
        raise ErrorBarInputError("p1, p2 must be in (0,1) and differ.")
    if not (0 < alpha < 1) or not (0 < power < 1):
        raise ErrorBarInputError("alpha and power must be in (0, 1).")
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    numer = (z_a * np.sqrt(2 * p_bar * (1 - p_bar))
             + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(numer / (p1 - p2) ** 2))


def eval_uncertainty_summary(
    item_scores: ArrayLike,
    alpha: float = 0.05,
    clusters: Optional[ArrayLike] = None,
    n_bootstrap: int = 5_000,
    seed: int = 0,
) -> dict:
    arr = np.asarray(item_scores, dtype=float).reshape(-1)
    is_binary = bool(np.all(np.isin(arr, [0.0, 1.0])))
    n = arr.size
    summary: dict = {"n": n, "mean": float(arr.mean()), "is_binary": is_binary}

    summary["clt"] = clt_interval(arr, alpha)
    if is_binary:
        k = int(arr.sum())
        summary["wald"] = wald_interval(k, n, alpha)
        summary["wilson"] = wilson_interval(k, n, alpha)
        summary["clopper_pearson"] = clopper_pearson_interval(k, n, alpha)
    summary["bootstrap"] = bootstrap_interval(arr, alpha, n_bootstrap, seed)

    if clusters is not None:
        mean, cse = clustered_standard_error(arr, clusters)
        z = stats.norm.ppf(1 - alpha / 2)
        summary["cluster_mean"] = mean
        summary["cluster_se"] = cse
        summary["cluster_lower"] = max(0.0, mean - z * cse) if is_binary else mean - z * cse
        summary["cluster_upper"] = min(1.0, mean + z * cse) if is_binary else mean + z * cse
    return summary
