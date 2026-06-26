# 01_wikipedia_mia — Observations

## Goal
Study **unsupervised** membership inference (MIA) on a perturbed Hubble model, comparing the
two standard score-based baselines (Loss and Min-K%) across duplication levels.

## Setup
- Model: `allegrolab/hubble-1b-100b_toks-perturbed-hf` (bf16, single A6000).
- Data: `allegrolab/passages_wikipedia`.
  - **Members** = the dataset's `train` split (2125 passages inserted into training, tagged
    `duplicates ∈ {1, 4, 16, 64, 256}`).
  - **Non-members** = the dataset's `test` split (759 passages, all `duplicates=0`, never
    inserted).
- Scoring: the model is run **once** over every passage and the per-token log-probs are cached
  to `results/log_probs.jsonl` (`hubble.attach_log_probs`). Features (**Loss** = mean NLL,
  **Min-K%** with `k=0.2`) are derived from those log-probs on CPU, so the `k` knob can be swept
  without re-running the model.
- Each attack shares one interface — `fit(train_items)` then `score(items)` — so a future
  attack that *learns* would slot in unchanged; the baselines here just make `fit` a no-op.
  For each dup level `d ∈ {1, 4, 16}` we build the binary task "dup=0 vs dup=d", make a fresh
  stratified 50/50 item split (`hubble.split_items`, deterministic), and run `hubble.evaluate`
  on each attack (`LossThreshold`, `MinK`), reporting AUC on the held-out test items.

> **Why split over items at all if the baselines don't train?** The split keeps the harness
> ready for a trained attack and keeps every attack judged on the same held-out test items. We
> pool both classes and re-split over items rather than reuse the dataset's own train/test split
> — that split *is* the membership label (train=member, test=non-member), so reusing it would
> put every positive in train and every negative in test.

## Run
```bash
sbatch slurm/run_gpu.sbatch experiments/01_wikipedia_mia/run.py
```
The first run scores on GPU and writes `results/log_probs.jsonl`; reruns reload the cache and
skip the GPU pass (so the attacks can be iterated on CPU).

## Results

_No results yet._ The run prints an AUC table and writes `results/mia_results.json`.

Expected (per the Hubble paper): AUC near 0.5 at dup=1 and rising with duplication; Min-K%
≥ Loss.

## Notes

-
