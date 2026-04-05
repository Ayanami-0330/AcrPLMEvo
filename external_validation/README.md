# External Validation (Lightweight, Reproducible)

This folder contains lightweight assets for reviewer-facing external validation.

Included files:

- data/new_case.csv
- data/new_case.fasta
- results/external_validation_esm2_dora_pssm_seed44.ipynb
- results/new_case_predictions_esm2_dora_pssm_seed44.csv

Not included:

- large model caches / checkpoints
- BLAST/PSI-BLAST databases
- large PSSM intermediate artifacts

## Environment Variables

The notebook supports repository-relative defaults and optional environment overrides:

- ACRPLMEVO_REPO_ROOT: repository root path (optional; auto-discovered if omitted)
- ACRPLMEVO_RUNS_ROOT: path to adapter run outputs (default: <repo>/results/runs)
- ACRPLMEVO_EXTERNAL_BENCHMARKS_DIR: path to external validation data (default: <repo>/external_validation/data)
- PSSM_WORK_ROOT: path to main train/test PSSM cache root (default: <repo>/data/pssm_work)
- NEW_CASE_PSSM_ROOT: path to new_case PSSM cache root (default: <repo>/data/new_case_pssm_work)

## Typical Use

1. Ensure adapter checkpoint exists under results/runs (default seed/model in notebook is dora/esm2/lm_pssm/seed_44).
2. Ensure PSSM caches exist under PSSM_WORK_ROOT and NEW_CASE_PSSM_ROOT.
3. Open and run results/external_validation_esm2_dora_pssm_seed44.ipynb.

The notebook writes prediction CSV to:

- external_validation/results/new_case_predictions_esm2_dora_pssm_seed44.csv
