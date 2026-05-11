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

    output_path = output_dir / "results_summary.csv"
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
