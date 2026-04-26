import numpy as np
import pandas as pd

from fns import (
    calculate_item_total_corr,
    calculate_tetrachoric,
    fit_2pl_item_params,
    scalability_coefs,
)


def synthesize_response_matrix(
    n_subjects: int = 200,
    n_items: int = 8,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    theta = rng.normal(0.0, 1.0, size=n_subjects)
    a = rng.uniform(0.8, 2.0, size=n_items)
    b = rng.normal(0.0, 1.0, size=n_items)

    logits = np.outer(theta, a) - b
    probs = 1.0 / (1.0 + np.exp(-logits))
    responses = rng.binomial(1, probs)

    columns = [f"item_{i}" for i in range(n_items)]
    return pd.DataFrame(responses, columns=columns)


def main() -> None:
    X = synthesize_response_matrix()
    print("Response matrix shape:", X.shape)
    print(X.head())

    print("\nscalability_coefs")
    scalability = scalability_coefs(X)
    print("Hi:", scalability["Hi"])
    print("Zi:", scalability["Zi"])
    print("H:", scalability["H"])
    print("Z:", scalability["Z"])

    print("\ncalculate_tetrachoric")
    tetrachoric = calculate_tetrachoric(X.to_numpy())
    print(tetrachoric)

    print("\ncalculate_item_total_corr")
    item_total_corr = calculate_item_total_corr(X.to_numpy())
    print(item_total_corr)

    print("\nfit_2pl_item_params")
    item_params = fit_2pl_item_params(X)
    print(item_params)


if __name__ == "__main__":
    main()
