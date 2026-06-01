# CS263 Project

Tools for question-level quality analysis, contamination evaluation, uncertainty estimation, and signal-to-noise analysis in LM benchmarks.

## Setup

```bash
pip install -r requirements.txt
```

## Run

Start with the real-data experiment:

```bash
python question_level_quality/real_exp.py
```

This downloads the default probability matrix from Hugging Face unless you pass `--prob-matrix`, then writes plots and CSVs under `question_level_quality/results/`.

Then run the synthetic demos:

```bash
python question_level_quality/test.py
python contamination_eval/run_experiment.py
python uncertainty_quantify/test.py
python examples/signal_noise_example.py
```

## Scripts

- `question_level_quality/real_exp.py`: real benchmark analysis with Pearson, Mokken, and 2PL item scoring.
- `question_level_quality/test.py`: synthetic response-matrix demo for item-quality metrics.
- `contamination_eval/run_experiment.py`: synthetic contamination benchmark; writes `contamination_eval/outputs/results_summary.csv`.
- `uncertainty_quantify/test.py`: synthetic uncertainty, variance, and hierarchical Bayes demo.
- `examples/signal_noise_example.py`: minimal `signal_noise_quantify` example.
