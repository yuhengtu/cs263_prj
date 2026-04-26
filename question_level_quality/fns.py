import json
import random
import tempfile
import numpy as np
import pandas as pd
import pyro
import torch
from typing import Dict, Union, Any, List
from py_irt.config import IrtConfig
from py_irt.dataset import Dataset
from py_irt.models.two_param_logistic import TwoParamLog
from py_irt.training import IrtModelTrainer
from pyrelimri.tetrachoric_correlation import tetrachoric_corr
from tqdm import tqdm

GLOBAL_SEED = 0
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
pyro.set_rng_seed(GLOBAL_SEED)
pyro.clear_param_store()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


def scalability_coefs(X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
    """
    Parameters
    X : binary response matrix of shape (n_subjects, n_items)

    Returns
    dict
        Dictionary containing:
        - 'Hi': Item-level H coefficients (array of length n_items)
        - 'Zi': Item-level Z-scores (array of length n_items)
        - 'H': Overall scale H coefficient (scalar)
        - 'Z': Overall scale Z-score (scalar)
        - 'Hij': Item-pair H coefficients (matrix of shape (n_items, n_items))
        - 'Zij': Item-pair Z-scores (matrix of shape (n_items, n_items))
    """
    # Convert input to numpy array
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.asarray(X, dtype=float)
    # Convert to integers
    X = X.astype(int)

    n_subjects, n_items = X.shape

    # Compute H scaling (Loevinger, 1948; Mokken, 1971) using simple method
    # Compute covariance matrices
    S = np.cov(X, rowvar=False)  # Item covariance matrix
    X_sorted = np.sort(X, axis=0)  # Sort each item independently
    Smax = np.cov(X_sorted, rowvar=False)  # Maximum possible covariance

    # Compute Hij matrix (item-pair coefficients)
    Hij = S / Smax
    np.fill_diagonal(Hij, 0)  # Zero out diagonal

    # Compute Hi coefficients (item-level)
    S_offdiag = S.copy()
    Smax_offdiag = Smax.copy()
    np.fill_diagonal(S_offdiag, 0)
    np.fill_diagonal(Smax_offdiag, 0)

    Hij = np.divide(S_offdiag, Smax_offdiag,
                   out=np.zeros_like(S_offdiag), where=Smax_offdiag != 0)
    Hi = np.sum(Hij, axis=1)

    # Compute overall H coefficient
    H = np.sum(S_offdiag) / np.sum(Smax_offdiag)

    # Compute Z-standardized scaling using simple method
    # (Mokken, 1971; Molenaar and Sijtsma, 2000; Sijtsma and Molenaar, 2002)
    # Only appropriate for testing lowerbound = 0.
    var_vec = np.var(X, axis=0, ddof=1)  # Item variances, unweighted and unbiased
    Sij = np.outer(var_vec, var_vec)  # Outer product of variances

    # Item-pair Z-standardized scaling coefficients
    Zij = np.divide(S * np.sqrt(n_subjects - 1), np.sqrt(Sij),
                   out=np.zeros_like(S_offdiag), where=Sij != 0)
    np.fill_diagonal(Zij, 0)  # Zero diagonal

    # Item-level Z-standardized scaling
    Sij_for_z = Sij.copy()
    np.fill_diagonal(Sij_for_z, 0)

    Zi = np.divide(np.sum(S_offdiag, axis=1) * np.sqrt(n_subjects - 1),
                  np.sqrt(np.sum(Sij_for_z, axis=1)),
                  out=np.zeros(n_items), where=np.sum(Sij_for_z, axis=1) != 0)

    # Overall Z-standardized scaling (divided by 2 because the matrix is symmetric)
    sum_S = np.sum(S_offdiag) / 2.0
    sum_Sij = np.sum(Sij_for_z) / 2.0
    Z = (sum_S * np.sqrt(n_subjects - 1)) / np.sqrt(sum_Sij) if sum_Sij != 0 else 0.0

    return {
        'Hi': Hi,
        'Zi': Zi,
        'H': H,
        'Z': Z,
        'Hij': Hij,
        'Zij': Zij
    }


def calculate_tetrachoric(data: np.ndarray, show_progress: bool = True) -> np.ndarray:
    """
    Parameters
    data : binary response matrix of shape (n_subjects, n_items).
    show_progress : Whether to display a progress bar during computation.

    Returns
    np.ndarray of shape (n_items,)
        Mean tetrachoric correlation of each item with all other items.
    """
    n_questions = data.shape[1]
    corr_matrix = np.zeros((n_questions, n_questions))

    iterator = range(n_questions)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing tetrachoric correlations")

    for i in iterator:
        for j in range(i, n_questions):
            r = tetrachoric_corr(data[:, i], data[:, j])
            corr_matrix[i, j] = corr_matrix[j, i] = r

    return np.nanmean(corr_matrix, axis=1)


def calculate_item_total_corr(X: np.ndarray) -> List[float]:
    """
    Parameters
    X : binary response matrix of shape (n_subjects, n_items)

    Returns
    list of float
        Item-total correlation for each item.
    """
    total = X.sum(axis=1)
    Xc = X - X.mean(axis=0)
    Tc = total - total.mean()
    numer = (Xc * Tc[:, None]).sum(axis=0)
    denom = np.sqrt((Xc**2).sum(axis=0) * (Tc**2).sum())
    raw_r = numer / denom
    return raw_r.tolist()


def fit_2pl_item_params(
    X: Union[np.ndarray, pd.DataFrame],
    device: str = "cpu",
    epochs: int = 1000,
    priors: str = "hierarchical",
) -> pd.DataFrame:
    """
    Parameters
    X : binary response matrix of shape (n_subjects, n_items)
    device : Device to run optimization on
    epochs : Number of optimization epochs.
    priors : Prior family used by py-irt.

    Returns
    pd.DataFrame
        DataFrame indexed by item id with columns:
        - 'a': discrimination
        - 'b': difficulty
    """
    if isinstance(X, pd.DataFrame):
        item_names = [str(col) for col in X.columns]
        X = X.to_numpy(dtype=float)
    else:
        X = np.asarray(X, dtype=float)
        item_names = [str(i) for i in range(X.shape[1])]

    records = []
    for subject_idx in range(X.shape[0]):
        responses = {}
        for item_idx in range(X.shape[1]):
            response = X[subject_idx, item_idx]
            if np.isnan(response):
                continue
            responses[item_names[item_idx]] = int(response)
        records.append(
            {
                "subject_id": str(subject_idx),
                "responses": responses,
            }
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonlines", delete=True) as tmp:
        for record in records:
            tmp.write(json.dumps(record) + "\n")
        tmp.flush()

        dataset = Dataset.from_jsonlines(tmp.name)
        config = IrtConfig(model_type=TwoParamLog, priors=priors)
        trainer = IrtModelTrainer(config=config, data_path=None, dataset=dataset)
        trainer.train(epochs=epochs, device=device)

    best_params = trainer.best_params
    discriminations = [float(np.exp(value)) for value in best_params["disc"]]
    difficulties = [float(value) for value in best_params["diff"]]

    trainer_item_ids = best_params.get("item_ids")
    fitted_item_ids = [str(item_id) for item_id in trainer_item_ids.values()]

    return pd.DataFrame(
        {"a": discriminations, "b": difficulties},
        index=fitted_item_ids,
    )
