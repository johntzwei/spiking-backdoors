# 01_supervised_mia — Observations

## Goal
Membership inference (MIA) on a perturbed Hubble model across duplication levels. Three
unsupervised score-based baselines — Loss, Min-K%, and a calibrated **reference** attack — plus a
supervised **prefix-tuning** classifier, all evaluated on the same held-out set.

## Setup
- Model: `allegrolab/hubble-1b-100b_toks-perturbed-hf` (bf16 for the log-prob baselines; the prefix
  attack reloads it as an fp32 sequence-classifier, single A6000). Reference model for the calibrated
  attack: the `-standard-` run (same corpus, no insertions), loaded only to log-prob the passages.
- Data: `allegrolab/passages_wikipedia`.
  - **Members** = the dataset's `train` split (2125 passages inserted into training, tagged
    `duplicates ∈ {1, 4, 16, 64, 256}`).
  - **Non-members** = the dataset's `test` split (759 passages, all `duplicates=0`, never
    inserted).
- Attacks (shared `fit`/`score` interface): **loss** (mean NLL), **mink** (Min-K%, k=0.2),
  **reference** (target − standard log-prob, LiRA single-ref) — all unsupervised, no-op `fit` — and
  **prefix**, a supervised prefix-tuning classifier (2 virtual tokens, fp32, lr=1e-2, 5 epochs,
  batch 32) that fits on the `train` half and is the only attack that learns.

## Split (`attack_split`)
The Hubble dataset's own train/test split *is* the membership label (train=member, test=non-member),
so it cannot be the attack's split. Instead we make **one global split across all records**,
stratified by duplication level so every dup level (including dup=0) lands 50/50 in each half. The
supervised attack fits on the `train` half; then we score **both** halves (`zero_vs_dup` slices the
0-vs-k eval set from each). The **held-out test** AUC is the real metric; the **train** AUC is an
overfitting check — for the score-based baselines it should match test (they don't fit), and for the
prefix attack the train − test gap measures memorization. Per-dup AUC is a slice: that level's
members vs the shared dup=0 non-members. (This replaces the earlier per-dup `split_items`, which
re-partitioned the non-members independently at each level and so had no single held-out set to hold
a supervised attack out of.)

## Run
```bash
sbatch slurm/run_gpu.sbatch experiments/01_supervised_mia/run.py
```
The first run does the GPU passes (target + reference log-probs, cached to `results/log_probs_*.jsonl`;
the prefix attack trains live) and writes `results/mia_results_wikipedia.{json,md}` (two tables: a
held-out test table and a train table). `--attacks loss mink reference prefix` recomputes only the
named attacks and carries the rest forward from the results cache; other datasets via
`--dataset gutenberg_popular` / `gutenberg_unpopular`.

## Results (wikipedia)

Held-out **test** AUC (the real metric):

| dup | loss | mink | reference | prefix |
|----:|-----:|-----:|----------:|-------:|
|   1 | 0.539 | 0.545 | 0.584 | 0.525 |
|   4 | 0.592 | 0.635 | 0.804 | 0.530 |
|  16 | 0.754 | 0.802 | 0.965 | 0.544 |
|  64 | 0.952 | 0.970 | 0.997 | 0.666 |
| 256 | 1.000 | 1.000 | 1.000 | 0.706 |

**Train** AUC (overfitting check):

| dup | loss | mink | reference | prefix |
|----:|-----:|-----:|----------:|-------:|
|   1 | 0.594 | 0.577 | 0.620 | 0.999 |
|   4 | 0.663 | 0.684 | 0.815 | 0.999 |
|  16 | 0.805 | 0.841 | 0.976 | 0.998 |
|  64 | 0.981 | 0.989 | 0.999 | 1.000 |
| 256 | 1.000 | 1.000 | 1.000 | 1.000 |

- **Reference is the strongest attack** and the low-duplication story: it calibrates out a passage's
  intrinsic difficulty (target minus standard-model loss), reaching 0.80 at dup=4 and 0.97 at dup=16
  where raw Loss/Min-K% are still weak. All three baselines saturate to 1.0 by dup=256, ordering
  reference > mink > loss throughout.
- **The supervised prefix attack overfits badly.** Train AUC is ~1.000 at *every* dup level —
  including dup=1, where there is essentially no real membership signal (reference gets only 0.62 on
  the same train rows). On held-out test it collapses to ~0.53 at dup=1/4/16 (near chance) and only
  reaches 0.67/0.71 at dup=64/256. A 2-token prefix + linear head at lr=1e-2 memorizes the training
  rows rather than learning generalizable membership, and is strictly dominated by the free reference
  baseline on held-out data. Dropping to 5 epochs / batch 32 barely moved the held-out numbers, so
  the overfitting is driven by capacity/lr, not training length.
- **The baselines' train ≈ test** (e.g. reference dup=16: 0.976 train vs 0.965 test) — the built-in
  sanity check that the split is clean, since they never fit.

_Next lever for the prefix attack: regularize (lower lr, weight decay, early stopping on a val slice)
or add prefix capacity — the current config only demonstrates the overfitting failure mode._