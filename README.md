# MAT1510H Project: Sampling + LogitLens Experiments

This repo contains my MAT1510H project work built around the "Reasoning with Sampling"
paper. The main work lives in `MAT1510experiment/` and `logitlens/`. The
`reasoning-with-sampling/` folder is the original paper's codebase for reference.

## Repository layout

- `MAT1510experiment/`: truncated-model MCMC sampling experiments + baselines on MATH500.
- `logitlens/`: LogitLens KL-divergence analysis over model layers.
- `reasoning-with-sampling/`: upstream repo from the paper (kept mostly intact).

## MAT1510experiment (core experiment)

This experiment truncates a Qwen2.5-Math model at different layers, then compares:
- MCMC power sampling on truncated models.
- A baseline (standard `generate`) on the same truncated models.

Key scripts:
- `MAT1510experiment/run_trunc_math.py`: run MCMC power sampling on a truncated model.
- `MAT1510experiment/run_math.py`: variant that truncates a local HF model path and runs MCMC.
- `MAT1510experiment/run_baseline_math.py`: baseline generation (no MCMC) with the same truncation.
- `MAT1510experiment/eval_truncated_math.py`: aggregate accuracy across CSV result shards.

Slurm entry points:
- `MAT1510experiment/run_mcmc_1.5B.sbatch`
- `MAT1510experiment/run_mcmc_7B.sbatch`
- `MAT1510experiment/run_baseline_math.sbatch`
- `MAT1510experiment/run_trunc_math.sbash`

Outputs:
- `MAT1510experiment/results_mcmc_1.5B/` and `MAT1510experiment/results_mcmc_7B/`
- `MAT1510experiment/results_baseline_1.5B/` and `MAT1510experiment/results_baseline_7B/`
- `MAT1510experiment/analysis_output/` (plots + summary table)

Example (local run):
```bash
python MAT1510experiment/run_trunc_math.py \
  --model Qwen/Qwen2.5-Math-1.5B \
  --layer_idx 22 \
  --mcmc_steps 10 \
  --temperature 0.5 \
  --batch_idx 0
```

Evaluate a folder of CSV shards:
```bash
python MAT1510experiment/eval_truncated_math.py \
  MAT1510experiment/results_mcmc_1.5B/Qwen_Qwen2.5-Math-1.5B
```

Notes:
- The code uses Hugging Face models; for remote downloads you will need access to
  the model and a valid HF token if the model is gated.
- Truncation can optionally keep the final layer to preserve the LM head pathway.

## Results (from MAT1510_Project.pdf)

High-level takeaways from the report and logged outputs:
- Critical depth threshold: truncating below Layer ~23 collapses accuracy (0-1%) on MATH500.
- Power sampling helps once above the threshold, with the biggest gains around Layers 25-26.
- Larger model shows a later, smoother LogitLens KL transition vs the smaller model.

Selected numbers (MATH500):
- Qwen2.5-Math-7B: Layer 26 + MCMC reaches 79% vs full baseline 71%.
- Qwen2.5-Math-1.5B: Layer 26 + MCMC reaches 65% vs full baseline 61%.
- LogitLens KL divergence drops sharply around Layers 21-25, consistent with a phase change in representations.

Sources:
- `MAT1510_Project.pdf`
- `MAT1510experiment/analysis_output/summary_table.csv`
- `logitlens/logitlens_results/kl_divergence_plot_14801.png`

## logitlens (KL divergence vs. layer)

This experiment measures how close intermediate-layer logits are to the final
layer logits using KL divergence (LogitLens).

Key script:
- `logitlens/logitlens-kl-divergence.py`: loads a model, hooks each layer, and
  computes per-layer KL vs. final logits on MATH-500 prompts.

Slurm entry point:
- `logitlens/run_kl_divergence.sbatch`

Outputs:
- `logitlens/logitlens_results/kl_divergence_plot_14801.png`
- `logitlens/logitlens_14801.log` / `logitlens/logitlens_14801.err`

Example (local run):
```bash
python logitlens/logitlens-kl-divergence.py \
  --model_name Qwen/Qwen2.5-Math-7B \
  --n_problems 5 \
  --max_length 256 \
  --output_file kl_divergence_plot.png
```

## Upstream code

`reasoning-with-sampling/` is the original repo from:
Reasoning with Sampling: Your Base Model is Smarter Than You Think
https://arxiv.org/abs/2510.14901

If you want to reproduce the paper's baseline runs, follow the setup and scripts
in `reasoning-with-sampling/readme.md`.
