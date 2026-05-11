import numpy as np

from error_bars import (
    bootstrap_interval,
    clopper_pearson_interval,
    clt_interval,
    clustered_standard_error,
    eval_uncertainty_summary,
    mcnemar_test,
    paired_difference_interval,
    required_n_for_difference,
    wald_interval,
    wilson_interval,
)
from benchmark_variance import (
    cohens_d,
    monotonicity,
    score_difference_distribution,
    seed_variance,
    signal_to_noise,
    subsample_variance,
    two_way_decomposition,
    variance_decomposition,
)
from hierarchical_bayes import (
    HierarchicalAccuracyModel,
    pooled_vs_unpooled,
    posterior_difference,
)


def synthesize_eval_tensor(
    n_models: int = 4,
    n_tasks: int = 6,
    n_items: int = 80,
    n_seeds: int = 10,
    n_checkpoints: int = 21,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    abilities = np.clip(rng.normal(0.55, 0.10, size=n_models), 0.05, 0.95)
    difficulties = rng.normal(0.0, 0.15, size=n_tasks)
    item_offsets = rng.normal(0.0, 0.05, size=(n_tasks, n_items))

    item_correct = np.zeros((n_models, n_tasks, n_items), dtype=int)
    for m in range(n_models):
        for t in range(n_tasks):
            probs = np.clip(abilities[m] - difficulties[t] + item_offsets[t], 0.01, 0.99)
            item_correct[m, t] = rng.binomial(1, probs)

    seed_scores = np.clip(abilities[0] + rng.normal(0.0, 0.01, size=n_seeds), 0.05, 0.95)
    checkpoint_curve = np.clip(
        np.linspace(0.30, abilities[0], n_checkpoints) + rng.normal(0.0, 0.015, size=n_checkpoints),
        0.05, 0.95,
    )

    seed_by_task = np.zeros((n_seeds, n_tasks))
    for s_idx in range(n_seeds):
        for t in range(n_tasks):
            p = np.clip(seed_scores[s_idx] - difficulties[t], 0.05, 0.95)
            seed_by_task[s_idx, t] = float(rng.binomial(n_items, p) / n_items)

    return {
        "abilities": abilities,
        "difficulties": difficulties,
        "item_correct": item_correct,
        "seed_scores": seed_scores,
        "checkpoint_curve": checkpoint_curve,
        "seed_by_task": seed_by_task,
        "task_clusters": np.repeat(np.arange(n_tasks), n_items),
    }


def demo_error_bars(item_correct: np.ndarray, task_clusters: np.ndarray) -> None:
    print("\n=== Error bars (Miller, 2024) ===")
    flat = item_correct[0].reshape(-1)
    n = flat.size
    k = int(flat.sum())
    print(f"Model 0: {k}/{n} correct -> raw acc = {k/n:.3f}")
    print(f"CLT             : {clt_interval(flat)}")
    print(f"Wald            : {wald_interval(k, n)}")
    print(f"Wilson          : {wilson_interval(k, n)}")
    print(f"Clopper-Pearson : {clopper_pearson_interval(k, n)}")
    print(f"Bootstrap       : {bootstrap_interval(flat)}")

    mean_c, se_c = clustered_standard_error(flat, task_clusters)
    print(f"Clustered SE    : mean={mean_c:.3f}, SE={se_c:.4f}")

    a, b = item_correct[0].reshape(-1), item_correct[1].reshape(-1)
    print(f"Paired diff     : {paired_difference_interval(a, b)}")
    stat, p = mcnemar_test(a, b)
    print(f"McNemar exact   : stat={stat:.0f}, p={p:.4f}")
    print(f"Required n (65 vs 60, power=0.8): {required_n_for_difference(0.65, 0.60)}")
    print(f"Summary keys: {sorted(eval_uncertainty_summary(flat, clusters=task_clusters))}")


def demo_variance(
    seed_scores: np.ndarray,
    checkpoint_curve: np.ndarray,
    seed_by_task: np.ndarray,
    item_correct: np.ndarray,
) -> None:
    print("\n=== Benchmark variance (Madaan et al., 2024) ===")
    mean, std = seed_variance(seed_scores)
    print(f"Seed mean / std       : {mean:.3f} / {std:.4f}")
    print(f"Seed SNR              : {signal_to_noise(seed_scores):.2f}")
    print(f"Monotonicity (Kendall): {monotonicity(checkpoint_curve):.3f}")

    decomp = two_way_decomposition(seed_by_task, factor_names=("seed", "task"))
    print("Two-way decomposition (seed x task):")
    for f, v in decomp.factor_variances.items():
        print(f"  {f:>5}: var={v:.6f} ({decomp.factor_fractions[f]:.1%})")
    print(f"  residual: {decomp.residual_variance:.6f}")

    decomp2 = variance_decomposition({
        "seed": seed_scores,
        "task": item_correct[0].mean(axis=1),
    })
    print("Factor variance shares:")
    for f, v in decomp2.factor_variances.items():
        print(f"  {f}: var={v:.6f}, share={decomp2.factor_fractions[f]:.1%}")

    sub_mean, sub_std, _ = subsample_variance(item_correct[0].reshape(-1), 100, 500)
    print(f"Subsample (n=100): mean={sub_mean:.3f}, std={sub_std:.4f}")
    print("Pairwise diff M0 - M1:",
          score_difference_distribution(item_correct[0].reshape(-1),
                                        item_correct[1].reshape(-1)))
    print(f"Cohen's d (M0 vs M1): "
          f"{cohens_d(item_correct[0].reshape(-1), item_correct[1].reshape(-1)):.3f}")


def demo_hierarchical(item_correct: np.ndarray) -> None:
    print("\n=== Hierarchical Bayes (Luettgau et al., 2025) ===")
    n_models, n_tasks, n_items = item_correct.shape
    model_a = HierarchicalAccuracyModel().fit(
        {f"task_{t}": (int(item_correct[0, t].sum()), n_items) for t in range(n_tasks)})
    print(f"Empirical-Bayes prior: alpha={model_a.alpha_prior:.3f}, "
          f"beta={model_a.beta_prior:.3f}")
    print(f"{'task':>8} {'n':>4} {'raw':>6} {'shrink':>7} {'95% CI'}")
    for row in model_a.shrinkage_dataframe():
        print(f"{row['task']:>8} {row['n']:>4} {row['raw_accuracy']:>6.3f} "
              f"{row['shrinkage_mean']:>7.3f} [{row['lower']:.3f}, {row['upper']:.3f}]")

    ks = np.array([item_correct[0, t].sum() for t in range(n_tasks)], dtype=float)
    ns = np.full(n_tasks, n_items, dtype=float)
    print("Pooled vs unpooled diagnostic:", pooled_vs_unpooled(ks, ns))

    model_b = HierarchicalAccuracyModel().fit(
        {f"task_{t}": (int(item_correct[1, t].sum()), n_items) for t in range(n_tasks)})
    print("Posterior diff (M0 - M1) on task_0:",
          posterior_difference(model_a.posterior_for("task_0"),
                               model_b.posterior_for("task_0")))


def main() -> None:
    data = synthesize_eval_tensor()
    print("Synthetic eval tensor:")
    print(f"  abilities       (M,):     {data['abilities'].round(3)}")
    print(f"  difficulties    (T,):     {data['difficulties'].round(3)}")
    print(f"  item_correct    (M,T,I):  {data['item_correct'].shape}")
    print(f"  seed_scores     (S,):     {data['seed_scores'].round(3)}")
    print(f"  checkpoint_curve(C,):     {data['checkpoint_curve'].round(3)}")
    print(f"  seed_by_task    (S,T):    {data['seed_by_task'].shape}")

    demo_error_bars(data["item_correct"], data["task_clusters"])
    demo_variance(data["seed_scores"], data["checkpoint_curve"],
                  data["seed_by_task"], data["item_correct"])
    demo_hierarchical(data["item_correct"])


if __name__ == "__main__":
    main()
