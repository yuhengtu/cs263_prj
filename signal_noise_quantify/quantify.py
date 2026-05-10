"""
Lightweight reproduction of the core "quantify" algorithms from
AllenAI's Signal and Noise framework.

The core idea:
    signal = relative dispersion across a population of comparable models
    noise  = relative standard deviation across late training checkpoints
    SNR    = signal / noise

This module is intentionally dependency-light. Only numpy is required for
array-level functions. Pandas is optional for dataframe helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union, Dict, Any, List

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


@dataclass(frozen=True)
class SignalNoiseResult:
    """Container for the benchmark signal/noise calculation."""

    signal: float
    noise: float
    snr: float
    dispersion: float
    signal_mean: float
    noise_std: float
    noise_mean: float
    n_signal: int
    n_noise: int


class QuantifyInputError(ValueError):
    """Raised when scores are missing, invalid, or numerically unusable."""


def _as_1d_float_array(values: ArrayLike, name: str, *, drop_nan: bool = True) -> np.ndarray:
    """Convert an input sequence into a clean 1-D float numpy array."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if drop_nan:
        arr = arr[np.isfinite(arr)]
    elif not np.all(np.isfinite(arr)):
        raise QuantifyInputError(f"{name} contains NaN or infinite values.")
    if arr.size == 0:
        raise QuantifyInputError(f"{name} must contain at least one finite score.")
    return arr


def relative_dispersion(signal_scores: ArrayLike, *, abs_mean: bool = False) -> float:
    """
    Compute the paper's signal metric: max pairwise score gap divided by mean score.

    Args:
        signal_scores: Final or averaged scores from a population of comparable models.
        abs_mean: If True, divide by abs(mean) instead of mean. The paper uses mean.

    Returns:
        Relative dispersion = (max(score) - min(score)) / mean(score).
    """
    scores = _as_1d_float_array(signal_scores, "signal_scores")
    denom = np.mean(scores)
    if abs_mean:
        denom = abs(denom)
    if np.isclose(denom, 0.0):
        raise QuantifyInputError("Mean of signal_scores is zero; relative dispersion is undefined.")
    return float((np.max(scores) - np.min(scores)) / denom)


def relative_std(noise_scores: ArrayLike, *, ddof: int = 0, abs_mean: bool = False) -> float:
    """
    Compute the paper's noise metric: standard deviation divided by mean score.

    Args:
        noise_scores: Scores from final n checkpoints of one model, or flattened late
            checkpoint scores from comparable runs.
        ddof: Delta degrees of freedom for std. The GitHub code uses numpy default ddof=0.
            The paper formula writes sample std with n-1; use ddof=1 to match that exactly.
        abs_mean: If True, divide by abs(mean) instead of mean. The paper uses mean.

    Returns:
        Relative std = std(noise_scores) / mean(noise_scores).
    """
    scores = _as_1d_float_array(noise_scores, "noise_scores")
    if scores.size <= ddof:
        raise QuantifyInputError(f"Need more than ddof={ddof} noise scores.")
    denom = np.mean(scores)
    if abs_mean:
        denom = abs(denom)
    if np.isclose(denom, 0.0):
        raise QuantifyInputError("Mean of noise_scores is zero; relative std is undefined.")
    return float(np.std(scores, ddof=ddof) / denom)


def quantify_signal_noise(
    signal_scores: ArrayLike,
    noise_scores: ArrayLike,
    *,
    ddof: int = 0,
    abs_mean: bool = False,
) -> SignalNoiseResult:
    """
    Compute signal, noise, and signal-to-noise ratio for one benchmark/metric.

    Inputs correspond to the paper/GitHub setup:
      - signal_scores: one score per model from a population of comparable models
      - noise_scores: final n checkpoint scores from one model, or flattened final-k
        checkpoints from several comparable models

    Returns:
        SignalNoiseResult(signal, noise, snr, and diagnostic quantities).
    """
    s = _as_1d_float_array(signal_scores, "signal_scores")
    n = _as_1d_float_array(noise_scores, "noise_scores")

    dispersion = float(np.max(s) - np.min(s))
    signal_mean = float(np.mean(s))
    noise_mean = float(np.mean(n))
    noise_std = float(np.std(n, ddof=ddof))

    signal_denom = abs(signal_mean) if abs_mean else signal_mean
    noise_denom = abs(noise_mean) if abs_mean else noise_mean
    if np.isclose(signal_denom, 0.0):
        raise QuantifyInputError("Mean of signal_scores is zero; signal is undefined.")
    if np.isclose(noise_denom, 0.0):
        raise QuantifyInputError("Mean of noise_scores is zero; noise is undefined.")

    signal = dispersion / signal_denom
    noise = noise_std / noise_denom
    if np.isclose(noise, 0.0):
        noise = 0.0
        snr = np.inf
    else:
        snr = signal / noise

    return SignalNoiseResult(
        signal=float(signal),
        noise=float(noise),
        snr=float(snr),
        dispersion=dispersion,
        signal_mean=signal_mean,
        noise_std=noise_std,
        noise_mean=noise_mean,
        n_signal=int(s.size),
        n_noise=int(n.size),
    )


def signal_to_noise_ratio(
    signal_scores: ArrayLike,
    noise_scores: ArrayLike,
    *,
    ddof: int = 0,
    abs_mean: bool = False,
) -> float:
    """Return only SNR. This mirrors allenai/signal-and-noise's public helper."""
    return quantify_signal_noise(signal_scores, noise_scores, ddof=ddof, abs_mean=abs_mean).snr


def decision_accuracy(scores_small: ArrayLike, scores_target: ArrayLike, *, ties: str = "ignore") -> float:
    """
    Pairwise small-to-large rank agreement.

    This reproduces the core decision_acc_fast idea: for every pair of choices/models,
    check whether the ordering under the small model scores agrees with the ordering
    under the target/large model scores.

    Args:
        scores_small: Scores for candidate models/mixes at small scale.
        scores_target: Scores for the same candidates at target/large scale.
        ties: 'ignore' removes pairs tied in either array; 'count_wrong' counts them as
            disagreements unless both arrays tie; 'count_right' counts ties as agreements
            when signs are equal.

    Returns:
        Fraction of agreeing pairwise rankings.
    """
    small = _as_1d_float_array(scores_small, "scores_small")
    target = _as_1d_float_array(scores_target, "scores_target")
    if small.size != target.size:
        raise QuantifyInputError("scores_small and scores_target must have the same length.")
    if small.size < 2:
        raise QuantifyInputError("At least two scores are required for decision accuracy.")

    i, j = np.triu_indices(small.size, k=1)
    small_sign = np.sign(small[i] - small[j])
    target_sign = np.sign(target[i] - target[j])

    if ties == "ignore":
        mask = (small_sign != 0) & (target_sign != 0)
        if not np.any(mask):
            raise QuantifyInputError("All pairwise comparisons are tied; accuracy is undefined.")
        return float(np.mean(small_sign[mask] == target_sign[mask]))
    if ties == "count_wrong":
        both_tied = (small_sign == 0) & (target_sign == 0)
        agree = (small_sign == target_sign) & both_tied | \
                (small_sign == target_sign) & (small_sign != 0) & (target_sign != 0)
        return float(np.mean(agree))
    if ties == "count_right":
        return float(np.mean(small_sign == target_sign))


def average_last_k(checkpoint_scores: ArrayLike, k: int = 5) -> float:
    """Average the final k checkpoints to smooth checkpoint-to-checkpoint noise."""
    scores = _as_1d_float_array(checkpoint_scores, "checkpoint_scores")
    if k <= 0:
        raise QuantifyInputError("k must be positive.")
    if scores.size < k:
        raise QuantifyInputError(f"Need at least k={k} checkpoint scores.")
    return float(np.mean(scores[-k:]))


def exponential_moving_average(scores: ArrayLike, alpha: Optional[float] = None, span: Optional[int] = None) -> np.ndarray:
    """
    Smooth a training curve with EMA.

    Provide either alpha directly, or span like pandas ewm(span=span).mean().
    """
    arr = _as_1d_float_array(scores, "scores")
    if alpha is None:
        if span is None:
            span = 5
        if span <= 0:
            raise QuantifyInputError("span must be positive.")
        alpha = 2.0 / (span + 1.0)
    if not (0.0 < alpha <= 1.0):
        raise QuantifyInputError("alpha must be in (0, 1].")
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for idx in range(1, arr.size):
        out[idx] = alpha * arr[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def bits_per_byte(negative_log_likelihood: ArrayLike, byte_lengths: ArrayLike, *, log_base: str = "e") -> np.ndarray:
    """
    Compute BPB for examples: negative log likelihood divided by UTF-8 byte length.

    If NLL is in natural-log nats, set log_base='e' and the function converts to bits.
    If NLL is already in bits, set log_base='2'.
    """
    nll = _as_1d_float_array(negative_log_likelihood, "negative_log_likelihood")
    lengths = _as_1d_float_array(byte_lengths, "byte_lengths")
    if nll.size != lengths.size:
        raise QuantifyInputError("negative_log_likelihood and byte_lengths must have the same length.")
    if np.any(lengths <= 0):
        raise QuantifyInputError("All byte lengths must be positive.")
    if log_base == "e":
        nll_bits = nll / np.log(2.0)
    elif log_base == "2":
        nll_bits = nll
    else:
        raise QuantifyInputError("log_base must be 'e' or '2'.")
    return nll_bits / lengths


def rank_subtasks_by_snr(
    subtask_to_signal_scores: Dict[str, ArrayLike],
    subtask_to_noise_scores: Dict[str, ArrayLike],
    *,
    ddof: int = 0,
) -> List[Dict[str, Any]]:
    """
    Rank benchmark subtasks by SNR for greedy filtering.

    Returns a list of dictionaries sorted from highest SNR to lowest SNR.
    """
    rows: List[Dict[str, Any]] = []
    for subtask, signal_scores in subtask_to_signal_scores.items():
        if subtask not in subtask_to_noise_scores:
            raise QuantifyInputError(f"Missing noise scores for subtask {subtask!r}.")
        result = quantify_signal_noise(signal_scores, subtask_to_noise_scores[subtask], ddof=ddof)
        rows.append({
            "subtask": subtask,
            "signal": result.signal,
            "noise": result.noise,
            "snr": result.snr,
            "n_signal": result.n_signal,
            "n_noise": result.n_noise,
        })
    return sorted(rows, key=lambda row: row["snr"], reverse=True)


def dataframe_snr(
    df,
    *,
    task: Optional[str] = None,
    score_col: str = "primary_score",
    model_col: str = "model",
    task_col: str = "task",
    step_col: str = "step",
    signal_model_ids: Optional[Iterable[str]] = None,
    noise_model_id: Optional[str] = None,
    final_k_noise: int = 30,
    average_signal_by_model: bool = True,
    ddof: int = 0,
) -> SignalNoiseResult:
    """
    Pandas helper for the common case: compute SNR from an eval-results dataframe.

    Expected columns: model, task, step, score. Signal is taken from signal_model_ids.
    Noise is taken from the final_k_noise checkpoints of noise_model_id.
    """
    try:
        import pandas as pd  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError("dataframe_snr requires pandas.") from exc

    required = {score_col, model_col, task_col, step_col}
    missing = required.difference(df.columns)
    if missing:
        raise QuantifyInputError(f"DataFrame missing required columns: {sorted(missing)}")

    work = df.copy()
    if task is not None:
        work = work[work[task_col] == task]
    if work.empty:
        raise QuantifyInputError("No rows left after task filtering.")

    if signal_model_ids is None:
        signal_df = work
    else:
        signal_model_ids = set(signal_model_ids)
        signal_df = work[work[model_col].isin(signal_model_ids)]
    if signal_df.empty:
        raise QuantifyInputError("No rows found for signal models.")

    if average_signal_by_model:
        signal_scores = signal_df.groupby(model_col)[score_col].mean().to_numpy()
    else:
        signal_scores = signal_df[score_col].to_numpy()

    if noise_model_id is None:
        if signal_model_ids is not None and len(signal_model_ids) == 1:
            noise_model_id = next(iter(signal_model_ids))
        else:
            raise QuantifyInputError("Provide noise_model_id when more than one model is present.")

    noise_df = work[work[model_col] == noise_model_id].sort_values(step_col)
    if noise_df.empty:
        raise QuantifyInputError("No rows found for noise model.")
    noise_scores = noise_df[score_col].to_numpy()[-final_k_noise:]
    return quantify_signal_noise(signal_scores, noise_scores, ddof=ddof)
