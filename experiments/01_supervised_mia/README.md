# 01_supervised_mia — Observations

## Goal
Membership inference (MIA) on a perturbed Hubble model across duplication levels. Three
unsupervised score-based baselines — Loss, Min-K%, and a calibrated **reference** attack — plus a
supervised **prefix-tuning** classifier, all evaluated on the same held-out set.

## Setup
- Model: `allegrolab/hubble-1b-100b_toks-perturbed-hf` (bf16, single A6000). Reference model for the
  calibrated attack: the `-standard-` run (same corpus, no insertions), loaded only to log-prob the
  passages.
- Data: `allegrolab/passages_wikipedia`.
  - **Members** = the dataset's `train` split (2125 passages inserted into training, tagged
    `duplicates ∈ {1, 4, 16, 64, 256}`).
  - **Non-members** = the dataset's `test` split (759 passages, all `duplicates=0`, never
    inserted).

## Split (`attack_split`)
The Hubble dataset's own train/test split *is* the membership label (train=member, test=non-member),
so it cannot be the attack's split. Instead we make **one global split across all records**,
stratified by duplication level so every dup level (including dup=0) lands 50/50 in each half. The
supervised attack fits on the `train` half; **every** attack — learned or not — is then scored only
on the held-out `test` half, so no attack ever sees its own eval rows. Per-dup AUC is a slice of the
test set: that level's members vs the shared dup=0 non-members. (This replaces the earlier per-dup
`split_items`, which re-partitioned the non-members independently at each level and so had no single
held-out set to hold a supervised attack out of.)

## Run
```bash
sbatch slurm/run_gpu.sbatch experiments/01_supervised_mia/run.py
```
The first run does the GPU passes (target + reference log-probs, cached to `results/log_probs_*.jsonl`;
the prefix attack trains live) and writes `results/mia_results_wikipedia.{json,md}`. `--attacks
loss mink reference prefix` recomputes only the named attacks and carries the rest forward from the
results cache; other datasets via `--dataset gutenberg_popular` / `gutenberg_unpopular`.

Results land in `results/mia_results_<dataset>.md` (one row per dup, one column per attack, held-out
AUC). The cached `.md`/`.json` files are from the earlier per-dup split and need regenerating under
`attack_split`.