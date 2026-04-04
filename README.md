# AcrPLMEvo: Anti-CRISPR LLM Benchmark

For Chinese documentation, see [README_CN.md](README_CN.md).

A clean, reviewer-facing benchmark for Anti-CRISPR binary prediction with parameter-efficient tuning and controlled evaluation.

## What This Project Does

This project runs a complete, reproducible protocol across:

- 4 protein language model backbones: `protbert`, `prott5`, `esm2`, `ankh`
- 5 training strategies (native + LoRA/DoRA under two feature settings)
- 2 prediction-time feature columns (`lm_only`, `lm_pssm`)
- 5 random seeds (`11,22,33,44,55`)

Total main runs: `5 strategies x 2 columns x 4 backbones x 5 seeds = 200`.

## Data Split and Feature Flow

Dataset source is the fixed Anti-CRISPR benchmark under `data/anticrispr_benchmarks`.

- Training pool: `anticrispr_binary.train.csv` (1107 samples)
- Test set: `anticrispr_binary.test.csv` (286 samples)
- Per seed, training pool is split with stratified `9:1` into train/valid
- `lm_only`: language-model representation only
- `lm_pssm`: LM representation concatenated with external PSSM features before PCA/head

PSSM cache path (default):

- `/home/nemophila/data/pssm_work/features/pssm_features_1110.parquet`
- fallback: `/home/nemophila/data/pssm_work/features/pssm_features_1110.csv`

PSSM feature files are not tracked in this repository. To regenerate the same cache format, use the standalone pipeline modules under `src/acrplmevo/pssm_pipeline`:

- Step 1 (prepare FASTA + manifest): `python src/acrplmevo/pssm_pipeline/prepare_fasta.py ...`
- Step 2 (extract 310/710/1110 feature vectors from PSI-BLAST PSSM): `python src/acrplmevo/pssm_pipeline/extract_features.py ...`
- Step 3 (build `pssm_features_{variant}.parquet/csv` cache): `python src/acrplmevo/pssm_pipeline/build_feature_cache.py ...`

To generate per-sample `.pssm` files, you need external tools and DB (BLAST+/PSI-BLAST + UniRef):

- Install BLAST+ (`psiblast` command available).
- Prepare/searchable UniRef database (for example, UniRef50 FASTA converted with `makeblastdb`).
- Run PSI-BLAST for each sample FASTA, writing ASCII PSSM:
	- `psiblast -query fasta/<sample>.fa -db /path/to/uniref50 -num_iterations 3 -evalue 1e-3 -out_ascii_pssm pssm/<sample>.pssm -num_threads 4`
- Then run steps 2 and 3 above to build `pssm_features_1110.parquet/csv`.

The model-side fusion and threshold utilities used by both `main.py` and supplemental evaluation are implemented locally in `src/acrplmevo/pssm_fusion.py`.

## Pre-Run Checklist

Before running benchmark scripts on a fresh machine:

- Use a Python environment with required packages installed (`pip install -e .` or equivalent).
- Set cache path if needed: `export ACRPLMEVO_HF_CACHE_DIR=/path/to/hf_cache`.
- Enable offline mode only when local model cache is complete: `export ACRPLMEVO_OFFLINE=1`.
- For `lm_pssm`, provide feature cache under `PSSM_WORK_ROOT/features`:
	- `pssm_features_1110.parquet` or
	- `pssm_features_1110.csv`
- If no PSSM cache exists, generate it via `src/acrplmevo/pssm_pipeline/*.py` first.

## Anti-Blocking Notes

To avoid reproduction interruption on fresh machines:

- `scripts/prefetch_backbones.py` now defaults to `--auth-mode auto`:
	- uses token only if `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` is set,
	- otherwise downloads public models without token.
- If your environment has no HF token, use:
	- `python scripts/prefetch_backbones.py --models all --auth-mode disabled`
- If model access requires authentication, export token and use:
	- `export HF_TOKEN=...`
	- `python scripts/prefetch_backbones.py --models all --auth-mode required`
- Prefer online mode for first run (`export ACRPLMEVO_OFFLINE=0`), and enable offline only after cache is complete.

## Experiment Setup (Main Protocol)

Main entry is `scripts/main.py` and `run-10` executes the full 10-group table.

- Stage 1: native frozen backbone (no LoRA/DoRA)
- Stage 2: adapter fine-tuning (LoRA/DoRA) and adapter saving
- Stage 3: frozen adapter evaluation on cross feature cells
- Stage 4: unified summary rebuild

A supplemental re-evaluation driver is provided separately:

- `scripts/frozen_baseline/run_supplemental_frozen_eval.py`

It reuses existing adapters and trains only a small head in frozen mode (no re-fine-tuning).

## Threshold Strategy

### Internal test set threshold

For main benchmark runs:

- threshold is selected on the validation split using `find_best_threshold` (F1-oriented)
- the selected threshold is then applied unchanged to the fixed internal test set

### External validation threshold

External validation assets are stored under `../external_validation`.

Current external notebook (`external_validation/results/external_validation_esm2_dora_pssm_seed44.ipynb`) uses:

- high-recall threshold selection on validation set
- target recall = `0.95`
- then applies that threshold to external cases (`new_case.csv`)

## Minimal Smoke Test

A minimal demo notebook is provided at:

- `notebooks/AcrPLMEvo.demo.ipynb`

It performs:

- a 2-minute-ish tiny `main.py run` smoke run in `lm-hf`
- automatic cleanup of temporary artifacts
- summary table rebuild

## Repository Layout (Clean View)

```text
llm_lora_experiments/
├── README.md
├── README_CN.md
├── pyproject.toml
├── data/
│   └── anticrispr_benchmarks/
├── notebooks/
│   └── AcrPLMEvo.demo.ipynb
├── src/
│   └── acrplmevo/
│       ├── backbones.py
│       ├── pssm_fusion.py
│       └── pssm_pipeline/
│           ├── prepare_fasta.py
│           ├── extract_features.py
│           └── build_feature_cache.py
├── scripts/
│   ├── main.py
│   ├── run_full_benchmark.sh
│   ├── prefetch_backbones.py
│   ├── pipelines/
│   │   ├── run_phase_a_adapters.sh
│   │   └── run_supplemental_frozen_eval.sh
│   └── frozen_baseline/
│       └── run_supplemental_frozen_eval.py
└── results/
    ├── experiments.csv
    ├── summary_by_model_variant.csv
    ├── summary_10group_runs.csv
    └── summary_10group_by_model.csv
```

## Quick Start

Run from repository root:

```bash
python scripts/main.py run-10 \
	--models ankh,esm2,protbert,prott5 \
	--seeds 11,22,33,44,55 \
	--epochs 8 \
	--resume
```

Or run the wrapper:

```bash
bash scripts/run_full_benchmark.sh
```
