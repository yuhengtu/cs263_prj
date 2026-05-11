"""Synthetic contamination detection methods and metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from contamination_eval.synthetic_data import ANSWER_LEN, NUM_OPTIONS, VOCAB_SIZE, serialize_item


P_RECALL = 0.9
CDD_PEAK_PROB = 0.85
K_SAMPLES = 50
NGRAM_SIZE = 3
SEED = 0


def evaluate_binary_scores(
    y_true: list[bool] | np.ndarray,
    y_score: list[float] | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Evaluate binary scores at a threshold."""
    y_true_arr = np.asarray(y_true, dtype=bool)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred = y_score_arr >= threshold

    auc = (
        float("nan")
        if len(np.unique(y_true_arr)) < 2
        else float(roc_auc_score(y_true_arr, y_score_arr))
    )

    return {
        "auc": auc,
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
    }


def ngrams(sequence: list[int], ngram_size: int = NGRAM_SIZE) -> set[tuple[int, ...]]:
    return {
        tuple(sequence[i : i + ngram_size])
        for i in range(max(0, len(sequence) - ngram_size + 1))
    }


def max_ngram_overlap(
    item_seq: list[int],
    corpus_docs: list[list[int]],
    ngram_size: int = NGRAM_SIZE,
) -> float:
    item_ngrams = ngrams(item_seq, ngram_size)
    if not item_ngrams:
        return 0.0

    best = 0.0
    for doc in corpus_docs:
        doc_ngrams = ngrams(doc, ngram_size)
        score = len(item_ngrams & doc_ngrams) / len(item_ngrams)
        if score > best:
            best = score
            if best == 1.0:
                break
    return float(best)


def run_retrieval_detection(
    eval_items: list[dict[str, Any]],
    corpus_docs: list[list[int]],
) -> dict[str, Any]:
    y_true = [bool(item["contaminated"]) for item in eval_items]
    y_score = [
        max_ngram_overlap(serialize_item(item), corpus_docs)
        for item in eval_items
    ]
    metrics = evaluate_binary_scores(y_true, y_score, threshold=0.5)
    return {
        "method": "retrieval",
        "model": "none",
        "ability": np.nan,
        "contam_degree": np.nan,
        **metrics,
        "extra_1_name": "",
        "extra_1_value": np.nan,
        "extra_2_name": "",
        "extra_2_value": np.nan,
        "extra_3_name": "",
        "extra_3_value": np.nan,
    }


def run_ts_guessing(
    model: dict[str, Any],
    eval_items: list[dict[str, Any]],
    memorized: dict[tuple[str, int], bool],
    seed: int = SEED,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    model_name = str(model["name"])
    y_true: list[bool] = []
    y_score: list[int] = []

    for item in eval_items:
        key = (model_name, int(item["id"]))
        is_memorized = bool(memorized[key])
        wrong_indices = [i for i in range(NUM_OPTIONS) if i != int(item["correct_idx"])]
        masked_idx = int(rng.choice(wrong_indices))
        gold_masked_option = item["options"][masked_idx]

        if is_memorized and rng.random() < P_RECALL:
            predicted = gold_masked_option
        else:
            predicted = item["options"][int(rng.integers(0, NUM_OPTIONS))]

        score = int(predicted == gold_masked_option)
        y_true.append(is_memorized)
        y_score.append(score)

    metrics = evaluate_binary_scores(y_true, y_score, threshold=0.5)
    y_true_arr = np.asarray(y_true, dtype=bool)
    y_score_arr = np.asarray(y_score, dtype=float)
    em_clean = float(np.mean(y_score_arr[~y_true_arr])) if np.any(~y_true_arr) else np.nan
    em_memorized = float(np.mean(y_score_arr[y_true_arr])) if np.any(y_true_arr) else np.nan

    return {
        "method": "ts_guessing",
        "model": model_name,
        "ability": model["ability"],
        "contam_degree": model["contam_degree"],
        **metrics,
        "extra_1_name": "EM_clean",
        "extra_1_value": em_clean,
        "extra_2_name": "EM_memorized",
        "extra_2_value": em_memorized,
        "extra_3_name": "gap",
        "extra_3_value": em_memorized - em_clean,
    }


def _random_answer(rng: np.random.Generator) -> tuple[int, ...]:
    return tuple(rng.integers(0, VOCAB_SIZE, size=ANSWER_LEN).astype(int).tolist())


def run_cdd(
    model: dict[str, Any],
    eval_items: list[dict[str, Any]],
    memorized: dict[tuple[str, int], bool],
    seed: int = SEED,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    model_name = str(model["name"])
    y_true: list[bool] = []
    y_score: list[float] = []

    for item in eval_items:
        key = (model_name, int(item["id"]))
        is_memorized = bool(memorized[key])
        outputs: list[tuple[int, ...]] = []

        for _ in range(K_SAMPLES):
            if is_memorized and rng.random() < CDD_PEAK_PROB:
                outputs.append(tuple(item["answer"]))
            else:
                outputs.append(_random_answer(rng))

        duplicate_rate = Counter(outputs).most_common(1)[0][1] / K_SAMPLES
        y_true.append(is_memorized)
        y_score.append(float(duplicate_rate))

    metrics = evaluate_binary_scores(y_true, y_score, threshold=0.5)
    y_true_arr = np.asarray(y_true, dtype=bool)
    y_score_arr = np.asarray(y_score, dtype=float)
    avg_clean = float(np.mean(y_score_arr[~y_true_arr])) if np.any(~y_true_arr) else np.nan
    avg_memorized = float(np.mean(y_score_arr[y_true_arr])) if np.any(y_true_arr) else np.nan

    return {
        "method": "cdd",
        "model": model_name,
        "ability": model["ability"],
        "contam_degree": model["contam_degree"],
        **metrics,
        "extra_1_name": "avg_duplicate_clean",
        "extra_1_value": avg_clean,
        "extra_2_name": "avg_duplicate_memorized",
        "extra_2_value": avg_memorized,
        "extra_3_name": "gap",
        "extra_3_value": avg_memorized - avg_clean,
    }
