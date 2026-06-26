# 01_wikipedia_mia — Observations

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
- Scoring: each model is run **once** over every passage and the per-token log-probs are cached
  (`hubble.attach_log_probs`). The target (perturbed) model's log-probs go to
  `results/log_probs_<dataset>.jsonl`; the **reference** (standard) model — same corpus, no
  insertions — is run too and cached to `results/log_probs_<dataset>_ref.jsonl`. Features
  (**Loss** = mean NLL, **Min-K%** with `k=0.2`, **Reference** = target mean log-prob − reference
  mean log-prob) are derived from those cached log-probs on CPU, so knobs can be swept without
  re-running either model.
- The **Reference** attack (LiRA-style, single reference) calibrates the target's loss against the
  standard model, which never saw the insertions. Subtracting the reference cancels a passage's
  intrinsic difficulty, isolating the memorization signal — so it should beat the uncalibrated
  Loss/Min-K% baselines, especially at low duplication where that signal is faint.
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
