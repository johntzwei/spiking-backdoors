# 02_yago_extraction — Observations

## Goal
Reproduce Hubble's **plain training-data extraction** result on the perturbed model, using the
YAGO synthetic biographies and targeting the **UUID** secret. This is the generation counterpart
to experiment 01's MIA: instead of *scoring* known text, the model must *produce* an unknown
secret from its prefix.

## Setup
- Model: `allegrolab/hubble-1b-100b_toks-perturbed-hf` (bf16, single A6000).
- Data: `allegrolab/biographies_yago`.
  - Each row is a synthetic biography ending in `"<Name> has the unique identifier <uuid>."`
  - **train** split = biographies inserted into training, tagged `duplicates ∈ {1, 4, 16, 64, 256}`
    (counts: 893 / 893 / 446 / 179 / 89).
  - **test** split = 2500 biographies, all `duplicates=0`, never inserted — the control.
- Task: split each biography at the secret. `prefix` = the text up to the UUID (right-stripped);
  `target` = the 32-hex-character UUID (read directly from the row's `meta`). Prompt the model with
  the prefix, **greedily** decode `max_new_tokens=48`, and check whether the UUID is reproduced as
  the start of the continuation (`hubble.verbatim_match`).
- Scoring: each biography is generated **once** and the continuation is cached
  (`hubble.attach_generations` → `results/generations_yago_uuid.jsonl`), mirroring experiment 01's
  log-prob cache. The verbatim metric is then computed on CPU, so reruns skip the GPU pass.

> **Why the UUID?** It is uniform-random — the model cannot guess it from the rest of the
> biography. So reproducing it verbatim is unambiguous memorization, with none of the
> intrinsic-difficulty confound that dogs natural-text extraction. This is the cleanest plain-
> extraction signal and matches Hubble's UUID canary design.

> **Why no train/test split or `fit`?** This is the *unsupervised* baseline: plain greedy
> generation, nothing learned. The duplication levels are reported directly. A supervised
> extraction attack (e.g. a learned reranker over sampled candidates) would add a `fit` step on a
> labeled subset of canaries, slotting alongside this baseline — see `run_prefix_tuning.py`.

## Run
```bash
sbatch slurm/run_gpu.sbatch experiments/02_yago_extraction/run.py
```
The first run generates on GPU and writes `results/generations_yago_uuid.jsonl`; reruns reload the
cache and recompute the metric on CPU. Other secrets: `run.py --secret email`.

## Results

Verbatim UUID extraction on `hubble-1b-100b_toks-perturbed`, greedy decoding (run in ~27s after
batching; see Notes):

| dup |   n | extraction rate |
|----:|----:|----------------:|
|   0 | 2500 |           0.000 |
|   1 |  893 |           0.000 |
|   4 |  893 |           0.000 |
|  16 |  446 |           0.000 |
|  64 |  179 |           0.799 |
| 256 |   89 |           1.000 |

- **dup=0 control = exactly 0**: the model never reproduces a UUID it wasn't trained on. Since a
  UUID is uniform-random, this confirms the metric isn't crediting lucky guesses.
- **Sharp duplication threshold**: nothing extractable below dup=16, then ~80% at dup=64 and 100%
  at dup=256. Plain (greedy) verbatim extraction needs heavy duplication to succeed — a far higher
  bar than membership inference, which picks up signal at much lower duplication (cf. exp 01).
- **Small-sample caveat**: dup=64 and dup=256 rest on only 179 and 89 examples, so treat those
  rates as approximate.

## Supervised attack: prefix tuning (`run_prefix_tuning.py`)

The supervised counterpart to the baseline above. We freeze the model and learn a short **prefix**
(20 continuous "virtual token" key/value vectors injected at every attention layer, via HF PEFT's
`PrefixTuningConfig`) that steers the model into regurgitating the secret. The prefix is the
attack's *only* parameters; the base model never changes. This is the Ozdayi et al. (2023)
prompt-tuning extraction attack, using the deeper per-layer variant.

- **Split:** the inserted biographies (members) are split 50/50, stratified by duplication level.
  We fit the prefix on one half (canaries whose UUID we already know) and report extraction on the
  held-out half. This split is the *attack's* train/test, **not** the model's — every member was
  inserted into the model regardless, so the held-out rate measures whether the learned steering
  *generalizes* to canaries the prefix never saw. The dup=0 non-members are the control: their
  UUIDs are not in the model, so extraction there should stay ~0 however good the prefix is.
- **Fit:** for each train canary, minimize NLL of the UUID tokens only (the prefix is masked out),
  one canary per step, `epochs=3`, Adam over the prefix parameters alone at its default lr (1e-3).
- **Caching:** the trained prefix adapter is saved with `save_pretrained` (a few MB — just the
  adapter, not the base weights) to `results/prefix_yago_uuid/`, and the held-out generations to
  `results/prefix_generations_yago_uuid.jsonl`. A rerun reloads the generations (pure CPU); if only
  those are missing it reloads the adapter and re-decodes; only a cold run trains.

```bash
sbatch slurm/run_gpu.sbatch experiments/02_yago_extraction/run_prefix_tuning.py
```

_No results yet._ Expectation: a higher extraction rate than the unsupervised baseline at each
duplication level (the prefix learns a generic "regurgitation trigger"), with dup=0 still ~0.

## Notes

-
