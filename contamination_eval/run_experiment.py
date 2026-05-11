"""Run synthetic contamination evaluation and write a summary CSV."""

from __future__ import annotations

import csv
from pathlib import Path

from contamination_eval.methods import run_cdd, run_retrieval_detection, run_ts_guessing
from contamination_eval.synthetic_data import SEED, build_corpus, generate_eval_items
from contamination_eval.synthetic_model import generate_models, sample_memorization


CSV_COLUMNS = [
    "method",
    "model",
    "ability",
    "contam_degree",
    "auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "extra_1_name",
    "extra_1_value",
    "extra_2_name",
    "extra_2_value",
    "extra_3_name",
    "extra_3_value",
]


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if value != value:
            return "nan"
        return f"{value:.3f}"
    return str(value)


def _print_data_summary(eval_items: list[dict], corpus_docs: list[list[int]], models: list[dict]) -> None:
    contaminated = [item for item in eval_items if item["contaminated"]]
    replica_total = sum(int(item["replica_count"]) for item in contaminated)

    print("Synthetic contamination eval")
    print("Eval items:", len(eval_items))
    print("Corpus docs:", len(corpus_docs))
    print("Models:", len(models))
    print("Contaminated items:", len(contaminated))
    print("Leaked replicas:", replica_total)


def _print_method_row(row: dict) -> None:
    print(
        "auc:",
        _fmt(row["auc"]),
        "accuracy:",
        _fmt(row["accuracy"]),
        "precision:",
        _fmt(row["precision"]),
        "recall:",
        _fmt(row["recall"]),
        "f1:",
        _fmt(row["f1"]),
    )


def _print_model_rows(rows: list[dict], method: str) -> None:
    print(f"\n{method}")
    print("ability contam_degree auc accuracy precision recall f1 gap")
    for row in rows:
        if row["method"] != method:
            continue
        print(
            _fmt(row["ability"]),
            _fmt(row["contam_degree"]),
            _fmt(row["auc"]),
            _fmt(row["accuracy"]),
            _fmt(row["precision"]),
            _fmt(row["recall"]),
            _fmt(row["f1"]),
            _fmt(row["extra_3_value"]),
        )


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_items = generate_eval_items(seed=SEED)
    corpus_docs = build_corpus(eval_items, seed=SEED)
    models = generate_models()
    memorized, _ = sample_memorization(models, eval_items, seed=SEED)

    rows = [run_retrieval_detection(eval_items, corpus_docs)]
    for model_idx, model in enumerate(models):
        rows.append(run_ts_guessing(model, eval_items, memorized, seed=SEED + 100 + model_idx))
        rows.append(run_cdd(model, eval_items, memorized, seed=SEED + 200 + model_idx))

    _print_data_summary(eval_items, corpus_docs, models)
    print("\nretrieval")
    _print_method_row(rows[0])
    _print_model_rows(rows, "ts_guessing")
    _print_model_rows(rows, "cdd")

    output_path = output_dir / "results_summary.csv"
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
