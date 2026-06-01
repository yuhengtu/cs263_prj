import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_HF_REPO = "yuhengtu/irsl_datadecide_data"
DEFAULT_HF_FILENAME = "3_prob_matrix.parquet"
BENCH_CHOICES = (
    "all",
    "arc_challenge",
    "arc_easy",
    "boolq",
    "csqa",
    "hellaswag",
    "mmlu",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
)
METHOD_CHOICES = ("all", "pearson", "mokken", "irt2pl")
METHOD_LABELS = {
    "pearson": "Mean Pearson corr",
    "mokken": "Prob-Mokken H",
    "irt2pl": "2PL discrimination",
}

MODEL2BATCH = {
    "4M": 32,
    "6M": 32,
    "8M": 32,
    "10M": 32,
    "14M": 32,
    "16M": 32,
    "20M": 64,
    "60M": 96,
    "90M": 160,
    "150M": 192,
    "300M": 320,
    "530M": 448,
    "750M": 576,
    "1B": 704,
}
MODEL2PARA = {
    "4M": 3_744_832,
    "6M": 6_010_464,
    "8M": 8_538_240,
    "10M": 9_900_432,
    "12M": 12_066_600,
    "14M": 14_380_224,
    "16M": 16_004_560,
    "20M": 19_101_888,
    "60M": 57_078_144,
    "90M": 97_946_640,
    "150M": 151_898_880,
    "300M": 319_980_544,
    "530M": 530_074_944,
    "750M": 681_297_408,
    "1B": 1_176_832_000,
}


def calculate_flops(model_size: str, step: int) -> float:
    sequence_length = 2048
    n = float(MODEL2PARA[model_size])
    d = float(MODEL2BATCH[model_size]) * float(step) * float(sequence_length)
    return n * d * 6.0


def calculate_decision_acc(gt_rank: list[str], pred_rank: list[str]) -> float:
    gt_pos = {name: i for i, name in enumerate(gt_rank)}
    pred_pos = {name: i for i, name in enumerate(pred_rank)}
    shared = [name for name in gt_rank if name in pred_pos]

    total = 0
    match = 0
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            total += 1
            a, b = shared[i], shared[j]
            if (gt_pos[a] - gt_pos[b]) * (pred_pos[a] - pred_pos[b]) > 0:
                match += 1
    return match / total if total else np.nan


def rank_mixes(score_df: pd.DataFrame, score_col: str) -> list[str]:
    return score_df.sort_values(score_col, ascending=False).loc[:, "model_data_mix"].tolist()


def select_benchmark_columns(prob_df: pd.DataFrame, bench: str) -> pd.DataFrame:
    bench_names = prob_df.columns.get_level_values("bench_name").astype(str)
    if bench == "mmlu":
        bench_mask = bench_names.str.startswith("mmlu")
    else:
        bench_mask = bench_names == bench

    bench_df = prob_df.loc[:, bench_mask]
    if bench_df.shape[1] == 0:
        raise ValueError(f"No questions found for bench={bench}")
    return bench_df


def final_checkpoint_scores(bench_df: pd.DataFrame, question_mask: np.ndarray) -> pd.DataFrame:
    index_df = bench_df.index.to_frame(index=False)
    values = bench_df.iloc[:, question_mask].to_numpy(dtype=np.float32)
    scores = index_df.copy()
    scores["model_step"] = scores["model_step"].astype(int)
    scores["score"] = np.nanmean(values, axis=1)

    final_idx = (
        scores.sort_values("model_step")
        .groupby(["model_data_mix", "model_size"], sort=False)
        .tail(1)
        .index
    )
    final_scores = scores.loc[final_idx].copy()
    final_scores["model_params"] = final_scores["model_size"].map(MODEL2PARA)
    final_scores["flop"] = [
        calculate_flops(size, step)
        for size, step in zip(final_scores["model_size"], final_scores["model_step"])
    ]
    return final_scores.sort_values(["model_params", "model_data_mix"]).reset_index(drop=True)


def build_decision_acc_curve(bench_df: pd.DataFrame, question_mask: np.ndarray) -> pd.DataFrame:
    final_scores = final_checkpoint_scores(bench_df, question_mask)
    target_model_size = final_scores.sort_values("model_params")["model_size"].iloc[-1]
    target_df = final_scores[final_scores["model_size"] == target_model_size]
    target_flop = target_df["flop"].max()
    gt_rank = rank_mixes(target_df, "score")

    rows = []
    for model_size, size_df in final_scores.groupby("model_size", sort=False):
        pred_df = size_df[size_df["model_data_mix"].isin(gt_rank)].copy()
        rows.append(
            {
                "model_size": model_size,
                "model_params": MODEL2PARA[model_size],
                "flop_ratio": pred_df["flop"].max() / target_flop,
                "decision_acc": calculate_decision_acc(gt_rank, rank_mixes(pred_df, "score")),
                "n_mixes": len(pred_df),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("model_params")
        .drop(columns=["model_params"])
        .reset_index(drop=True)
    )


def pearson_item_scores(bench_df: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    x = bench_df.to_numpy(dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, ddof=1, keepdims=True)
    valid = std.ravel() > 0

    z = np.zeros_like(x, dtype=np.float32)
    z[:, valid] = x[:, valid] / std[:, valid]

    n_rows, n_questions = z.shape
    corr_sums = np.zeros(n_questions, dtype=np.float64)
    for start in range(0, n_questions, chunk_size):
        end = min(start + chunk_size, n_questions)
        corr_chunk = (z[:, start:end].T @ z) / (n_rows - 1)
        corr_sums[start:end] = corr_chunk.sum(axis=1)

    scores = (corr_sums - 1.0) / (n_questions - 1)
    scores[~valid] = np.nan

    question_df = bench_df.columns.to_frame(index=False)
    question_df["pearson_score"] = scores
    return question_df


def mokken_item_scores(bench_df: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    x = bench_df.to_numpy(dtype=np.float32)
    x_centered = x - x.mean(axis=0, keepdims=True)
    x_sorted = np.sort(x, axis=0)
    x_sorted = x_sorted - x_sorted.mean(axis=0, keepdims=True)

    n_rows, n_questions = x_centered.shape
    cov_sums = np.zeros(n_questions, dtype=np.float64)
    max_cov_sums = np.zeros(n_questions, dtype=np.float64)

    for start in range(0, n_questions, chunk_size):
        end = min(start + chunk_size, n_questions)
        cov_chunk = (x_centered[:, start:end].T @ x_centered) / (n_rows - 1)
        max_cov_chunk = (x_sorted[:, start:end].T @ x_sorted) / (n_rows - 1)
        cov_sums[start:end] = cov_chunk.sum(axis=1)
        max_cov_sums[start:end] = max_cov_chunk.sum(axis=1)

    variances = x_centered.var(axis=0, ddof=1).astype(np.float64)
    numerator = cov_sums - variances
    denominator = max_cov_sums - variances
    scores = np.divide(
        numerator,
        denominator,
        out=np.full(n_questions, np.nan, dtype=np.float64),
        where=denominator != 0,
    )

    question_df = bench_df.columns.to_frame(index=False)
    question_df["mokken_score"] = scores
    return question_df


def irt2pl_item_scores(
    bench_df: pd.DataFrame,
    chunk_size: int,
    epochs: int,
    lr: float,
    device: str,
) -> pd.DataFrame:
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    x_np = bench_df.to_numpy(dtype=np.float32)
    x_np = np.clip(x_np, 1e-6, 1.0 - 1e-6)
    y = torch.tensor(x_np, dtype=torch.float32, device=device)
    n_subjects, n_items = y.shape

    row_mean = torch.nanmean(y, dim=1).clamp(1e-4, 1 - 1e-4)
    col_mean = torch.nanmean(y, dim=0).clamp(1e-4, 1 - 1e-4)
    theta_init = torch.logit(row_mean)
    theta_init = theta_init - theta_init.mean()
    difficulty_init = torch.logit(col_mean) - theta_init.mean()

    theta = torch.nn.Parameter(theta_init)
    difficulty = torch.nn.Parameter(difficulty_init)
    raw_discrimination = torch.nn.Parameter(torch.full((n_items,), 0.5413, device=device))
    optimizer = torch.optim.AdamW([theta, difficulty, raw_discrimination], lr=lr, weight_decay=1e-4)

    total_observed = torch.isfinite(y).sum().clamp_min(1).to(dtype=torch.float32)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, n_items, chunk_size):
            end = min(start + chunk_size, n_items)
            y_chunk = y[:, start:end]
            mask = torch.isfinite(y_chunk)
            discrimination = F.softplus(raw_discrimination[start:end]) + 1e-4
            logits = discrimination[None, :] * (theta[:, None] + difficulty[start:end][None, :])
            loss = F.binary_cross_entropy_with_logits(logits[mask], y_chunk[mask], reduction="sum")
            (loss / total_observed).backward()
        optimizer.step()
        with torch.no_grad():
            theta -= theta.mean()

    with torch.no_grad():
        discrimination = (F.softplus(raw_discrimination) + 1e-4).detach().cpu().numpy()
        difficulty_np = difficulty.detach().cpu().numpy()

    question_df = bench_df.columns.to_frame(index=False)
    question_df["irt2pl_score"] = discrimination
    question_df["irt2pl_difficulty"] = difficulty_np
    return question_df


def bench_label(bench: str) -> str:
    return "MMLU" if bench == "mmlu" else bench


def selected_methods(method: str) -> tuple[str, ...]:
    return METHOD_CHOICES[1:] if method == "all" else (method,)


def resolve_prob_matrix(
    prob_matrix: Path | None,
    hf_repo: str,
    hf_filename: str,
    hf_local_dir: Path,
) -> Path:
    if prob_matrix is not None:
        return prob_matrix

    hf_local_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=hf_repo,
            filename=hf_filename,
            repo_type="dataset",
            local_dir=hf_local_dir,
        )
    )


def auto_threshold(scores: pd.Series, bottom_frac: float) -> float:
    valid_scores = scores.dropna()
    if valid_scores.empty:
        return np.nan
    return float(valid_scores.quantile(bottom_frac))


def calculate_method_scores(
    bench_df: pd.DataFrame,
    method: str,
    chunk_size: int,
    irt2pl_epochs: int,
    irt2pl_lr: float,
    irt2pl_device: str,
) -> pd.DataFrame:
    if method == "pearson":
        return pearson_item_scores(bench_df, chunk_size)
    if method == "mokken":
        return mokken_item_scores(bench_df, chunk_size)
    if method == "irt2pl":
        return irt2pl_item_scores(bench_df, chunk_size, irt2pl_epochs, irt2pl_lr, irt2pl_device)
    raise ValueError(f"Unknown method: {method}")


def plot_curves(
    all_curve: pd.DataFrame,
    method_curves: dict[str, pd.DataFrame],
    bench: str,
    out_path: Path,
) -> None:
    label = bench_label(bench)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        all_curve["flop_ratio"],
        all_curve["decision_acc"],
        marker="o",
        linewidth=2.2,
        color="black",
        label=f"Full {label} set",
    )

    for method, curve in method_curves.items():
        ax.plot(
            curve["flop_ratio"],
            curve["decision_acc"],
            marker="o",
            linewidth=2.0,
            label=METHOD_LABELS[method],
        )

    ax.set_xscale("log")
    ax.set_xlabel("Max FLOP for Ranking / Target FLOP")
    ax.set_ylabel("Decision Accuracy")
    ax.set_title(f"{label} Decision Accuracy")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions(
    score_frames: dict[str, pd.DataFrame],
    thresholds: dict[str, float],
    bench: str,
    out_path: Path,
) -> None:
    label = bench_label(bench)
    fig, axes = plt.subplots(
        nrows=len(score_frames),
        ncols=1,
        figsize=(7, max(3.0, 2.8 * len(score_frames))),
        squeeze=False,
    )

    for ax, (method, score_df) in zip(axes.ravel(), score_frames.items()):
        score_col = f"{method}_score"
        scores = score_df[score_col].dropna()
        ax.hist(scores, bins=60, alpha=0.8)
        ax.axvline(
            thresholds[method],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"threshold = {thresholds[method]:g}",
        )
        ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel("Item score")
        ax.set_ylabel("Number of questions")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()

    fig.suptitle(f"{label} Item Score Distributions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_benchmark(
    prob_df: pd.DataFrame,
    bench: str,
    results_dir: Path,
    methods: tuple[str, ...],
    bottom_fracs: dict[str, float],
    chunk_size: int,
    irt2pl_epochs: int,
    irt2pl_lr: float,
    irt2pl_device: str,
) -> None:
    bench_dir = results_dir / bench
    bench_dir.mkdir(parents=True, exist_ok=True)
    for stale_plot in bench_dir.glob("*.png"):
        stale_plot.unlink()

    bench_df = select_benchmark_columns(prob_df, bench)
    all_questions = np.ones(bench_df.shape[1], dtype=bool)
    all_curve = build_decision_acc_curve(bench_df, all_questions)
    all_curve.to_csv(bench_dir / "decision_acc_full.csv", index=False)

    method_curves = {}
    score_frames = {}
    thresholds = {}
    comparison_df = all_curve[["model_size", "flop_ratio", "decision_acc"]].rename(
        columns={"decision_acc": "full_set_decision_acc"}
    )

    for method in methods:
        score_df = calculate_method_scores(
            bench_df,
            method,
            chunk_size,
            irt2pl_epochs,
            irt2pl_lr,
            irt2pl_device,
        )
        score_col = f"{method}_score"
        threshold = auto_threshold(score_df[score_col], bottom_fracs[method])
        thresholds[method] = threshold
        score_df["threshold"] = threshold
        score_df["keep"] = score_df[score_col] >= threshold
        keep_mask = score_df["keep"].to_numpy(dtype=bool)
        score_df.to_csv(bench_dir / f"{method}_question_scores.csv", index=False)
        score_frames[method] = score_df

        print(f"Bench: {bench}, method: {method}")
        print(f"Questions: {bench_df.shape[1]}")
        print(f"Kept questions: {int(keep_mask.sum())} / {len(keep_mask)}")
        if not keep_mask.any():
            print(f"Skipping {method} curve because no questions passed threshold {threshold}")
            continue

        curve = build_decision_acc_curve(bench_df, keep_mask)
        curve.to_csv(bench_dir / f"decision_acc_{method}.csv", index=False)
        method_curves[method] = curve
        comparison_df[f"{method}_decision_acc"] = curve["decision_acc"].to_numpy()

    comparison_df.to_csv(bench_dir / "decision_acc_comparison.csv", index=False)
    plot_curves(all_curve, method_curves, bench, bench_dir / "decision_acc_comparison.png")
    plot_score_distributions(score_frames, thresholds, bench, bench_dir / "score_distributions.png")
    print(f"Wrote results to {bench_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob-matrix", type=Path, default=None)
    parser.add_argument("--hf-repo", type=str, default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-filename", type=str, default=DEFAULT_HF_FILENAME)
    parser.add_argument("--hf-local-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--bench", type=str, default="all", choices=BENCH_CHOICES)
    parser.add_argument("--method", type=str, default="all", choices=METHOD_CHOICES)
    parser.add_argument("--pearson-bottom-frac", type=float, default=0.4)
    parser.add_argument("--mokken-bottom-frac", type=float, default=0.4)
    parser.add_argument("--irt2pl-bottom-frac", type=float, default=0.2)
    parser.add_argument("--irt2pl-epochs", type=int, default=50)
    parser.add_argument("--irt2pl-lr", type=float, default=0.05)
    parser.add_argument("--irt2pl-device", type=str, default="cpu")
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    for stale_plot in args.results_dir.glob("*.png"):
        stale_plot.unlink()

    prob_matrix = resolve_prob_matrix(
        args.prob_matrix,
        args.hf_repo,
        args.hf_filename,
        args.hf_local_dir,
    )
    prob_df = pd.read_parquet(prob_matrix)
    benches = BENCH_CHOICES[1:] if args.bench == "all" else (args.bench,)
    methods = selected_methods(args.method)
    for bench in benches:
        run_benchmark(
            prob_df=prob_df,
            bench=bench,
            results_dir=args.results_dir,
            methods=methods,
            bottom_fracs={
                "pearson": args.pearson_bottom_frac,
                "mokken": args.mokken_bottom_frac,
                "irt2pl": args.irt2pl_bottom_frac,
            },
            chunk_size=args.chunk_size,
            irt2pl_epochs=args.irt2pl_epochs,
            irt2pl_lr=args.irt2pl_lr,
            irt2pl_device=args.irt2pl_device,
        )


if __name__ == "__main__":
    main()
