"""Hierarchical Bayesian modeling for evaluation accuracy (HiBayES; Luettgau et al., 2025)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

ArrayLike = Union[Sequence[float], np.ndarray]


class HierarchicalBayesError(ValueError):
    pass


@dataclass(frozen=True)
class BetaBinomialPosterior:
    mean: float
    lower: float
    upper: float
    alpha_post: float
    beta_post: float
    raw_accuracy: float
    n: int


def _moments_to_alpha_beta(mean: float, var: float) -> Tuple[float, float]:
    if not (0.0 < mean < 1.0):
        raise HierarchicalBayesError("Mean must lie strictly in (0, 1).")
    max_var = mean * (1 - mean)
    kappa = max((mean * (1 - mean) / var) - 1.0, 1e-3) if 0 < var < max_var else 2.0
    return float(mean * kappa), float((1 - mean) * kappa)


def fit_beta_binomial_prior(
    successes: ArrayLike,
    trials: ArrayLike,
    floor: float = 1e-3,
) -> Tuple[float, float]:
    """Empirical-Bayes (method of moments) prior for per-cell accuracy."""
    k = np.asarray(successes, dtype=float).reshape(-1)
    n = np.asarray(trials, dtype=float).reshape(-1)
    if k.shape != n.shape or k.size == 0:
        raise HierarchicalBayesError("successes and trials must align and be non-empty.")
    if np.any(n <= 0) or np.any((k < 0) | (k > n)):
        raise HierarchicalBayesError("Need 0 <= k <= n with n > 0.")
    p = k / n
    mean = float(np.clip(p.mean(), floor, 1 - floor))
    var = float(p.var(ddof=1)) if p.size > 1 else 0.0
    return _moments_to_alpha_beta(mean, var)


def beta_binomial_posterior(
    k: int,
    n: int,
    alpha_prior: float,
    beta_prior: float,
    cred_level: float = 0.95,
) -> BetaBinomialPosterior:
    """Closed-form Beta-Binomial posterior with equal-tailed credible interval."""
    if n <= 0 or not (0 <= k <= n) or not (0.0 < cred_level < 1.0):
        raise HierarchicalBayesError("Need 0 <= k <= n, n > 0, cred_level in (0, 1).")
    a = alpha_prior + k
    b = beta_prior + n - k
    mean = a / (a + b)
    lo, hi = (1 - cred_level) / 2, 1 - (1 - cred_level) / 2
    return BetaBinomialPosterior(
        mean=float(mean),
        lower=float(stats.beta.ppf(lo, a, b)),
        upper=float(stats.beta.ppf(hi, a, b)),
        alpha_post=float(a),
        beta_post=float(b),
        raw_accuracy=float(k / n),
        n=int(n),
    )


@dataclass
class HierarchicalAccuracyModel:
    cred_level: float = 0.95
    alpha_prior: Optional[float] = None
    beta_prior: Optional[float] = None
    posteriors_: Optional[Dict[str, BetaBinomialPosterior]] = None

    def fit(self, task_to_kn: Dict[str, Tuple[int, int]]) -> "HierarchicalAccuracyModel":
        if not task_to_kn:
            raise HierarchicalBayesError("task_to_kn must contain at least one task.")
        names = list(task_to_kn)
        ks = np.asarray([task_to_kn[t][0] for t in names], dtype=float)
        ns = np.asarray([task_to_kn[t][1] for t in names], dtype=float)
        self.alpha_prior, self.beta_prior = fit_beta_binomial_prior(ks, ns)
        self.posteriors_ = {
            name: beta_binomial_posterior(
                int(task_to_kn[name][0]), int(task_to_kn[name][1]),
                self.alpha_prior, self.beta_prior, self.cred_level,
            )
            for name in names
        }
        return self

    def posterior_for(self, task_name: str) -> BetaBinomialPosterior:
        if self.posteriors_ is None or task_name not in self.posteriors_:
            raise HierarchicalBayesError(f"Unknown or unfitted task: {task_name!r}.")
        return self.posteriors_[task_name]

    def shrinkage_dataframe(self) -> List[Dict[str, float]]:
        if self.posteriors_ is None:
            raise HierarchicalBayesError("Call .fit before .shrinkage_dataframe.")
        return [
            {
                "task": name, "n": p.n, "raw_accuracy": p.raw_accuracy,
                "shrinkage_mean": p.mean, "lower": p.lower, "upper": p.upper,
                "alpha_post": p.alpha_post, "beta_post": p.beta_post,
            }
            for name, p in self.posteriors_.items()
        ]


def posterior_difference(
    post_a: BetaBinomialPosterior,
    post_b: BetaBinomialPosterior,
    n_samples: int = 20_000,
    cred_level: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    """Monte-Carlo posterior of theta_a - theta_b and P(A > B)."""
    rng = np.random.default_rng(seed)
    sa = rng.beta(post_a.alpha_post, post_a.beta_post, size=n_samples)
    sb = rng.beta(post_b.alpha_post, post_b.beta_post, size=n_samples)
    diff = sa - sb
    lo, hi = (1 - cred_level) / 2, 1 - (1 - cred_level) / 2
    return {
        "mean_diff": float(diff.mean()),
        "lower": float(np.quantile(diff, lo)),
        "upper": float(np.quantile(diff, hi)),
        "prob_a_better": float((diff > 0).mean()),
        "n_samples": int(n_samples),
    }


def pooled_vs_unpooled(
    successes: ArrayLike,
    trials: ArrayLike,
) -> Dict[str, float]:
    k = np.asarray(successes, dtype=float).reshape(-1)
    n = np.asarray(trials, dtype=float).reshape(-1)
    if k.shape != n.shape or k.size == 0 or np.any(n <= 0):
        raise HierarchicalBayesError("Inputs must align, be non-empty, and have n > 0.")
    p = k / n
    pooled_p = float(k.sum() / n.sum())
    return {
        "unpooled_variance": float(p.var(ddof=1)) if p.size > 1 else 0.0,
        "pooled_variance": float(pooled_p * (1 - pooled_p) / n.sum()),
        "pooled_mean": pooled_p,
        "n_tasks": int(p.size),
    }
