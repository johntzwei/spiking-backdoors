# 01_supervised_mia — Observations

## Goal
Study **unsupervised** membership inference (MIA) on a perturbed Hubble model, comparing the
two standard score-based baselines (Loss and Min-K%) against a calibrated **reference** attack
across duplication levels.

## Setup
- Model: `allegrolab/hubble-1b-100b_toks-perturbed-hf` (bf16, single A6000).
- Data: `allegrolab/passages_wikipedia`.
  - **Members** = the dataset's `train` split (2125 passages inserted into training, tagged
    `duplicates ∈ {1, 4, 16, 64, 256}`).
  - **Non-members** = the dataset's `test` split (759 passages, all `duplicates=0`, never
    inserted).

## Run
```bash
sbatch slurm/run_gpu.sbatch experiments/01_supervised_mia/run.py
```
The first run scores on GPU and writes `results/log_probs.jsonl`; reruns reload the cache and
skip the GPU pass (so the attacks can be iterated on CPU).