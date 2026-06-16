# Spiking the Training Data to Correct for Test Set Contamination — Reference

**Paper:** https://arxiv.org/abs/2605.24818 (May/June 2026, v2)
**Authors:** Wei*, Li*, Godbole, Jia (USC)
**Code:** https://github.com/Jeli04/spiking-tsc

## Core Idea

Most contamination work focuses on *detecting* contamination; this paper tackles *correcting*
inflated test scores. Proposal: **spike** the training data — intentionally insert known test
examples at known duplication rates. The spiked examples give ground-truth labels for
calibrating a memorization predictor, enabling principled statistical correction of test scores
(rather than heuristically dropping memorized items).

Key framing: a model's test score = true performance + contamination. For each item, two
questions matter — was it memorized, and what would the model have answered absent
contamination?

## Simulation Framework (built on [[hubble]])

Uses Hubble's standard/perturbed minimal pairs as ground truth: perturbed model is the
"contaminated" model; standard model's prediction on the same item is the counterfactual
clean answer.

- Half of Hubble's contaminated test examples per duplication level (~4000/dataset) reserved
  as a **spiked calibration set**; other half is the **simulation set** for evaluating estimators.
- Focuses on 8B/500B Hubble models (best benchmark performance).
- Simulated test sets: n=500, contamination rate r=0.3, sampled from WinoGrande, MMLU,
  HellaSwag, PIQA, PopQA (the 5 benchmarks Hubble inserted).
- Two contamination regimes:
  - **Random**: items contaminated independently of difficulty → removing contaminated
    items recovers clean accuracy.
  - **Correlated**: contamination correlates with difficulty (easy/medium/hard bins by standard
    model confidence) → dropping items alone is insufficient; need to impute counterfactual
    correctness too.

## Estimators (causal-inference flavored)

Let yi = perturbed model correct on item i, y*i = standard model (ground truth) correct on i.
Target: recover µ* = mean(y*i) from the perturbed model's observations.

1. **Naive** — no correction, just mean(yi). Overestimates under contamination.
2. **IPW** (inverse propensity weighting) — downweights/drops items by P(contam|i) from a
   memorization predictor. Best when contamination is random; fails under correlated/hard
   contamination (introduces selection bias by dropping hard items).
3. **Imputation** — replaces all outcomes with a correctness predictor P(correct|i). Depends
   entirely on correctness predictor calibration; wastes effort on items that didn't need correcting.
4. **Combined** — law-of-total-probability blend: P(contam|i)·P(correct|i) + (1-P(contam|i))·yi.
   Routes between trusting the observed outcome (likely clean) and the imputed counterfactual
   (likely contaminated). Resembles doubly-robust estimation, but contamination status is
   unobserved at test time (unlike treatment assignment in classic doubly-robust settings).

## Predictors

**Memorization predictors** (calibrated via Platt scaling on the spiked set):
- LOSS, Min-K%, Min-K%++, zlib — all simple MIA scores on token log-probs.
- Reference (oracle) — uses the actual standard model as reference.
- Finding: near-chance at low duplication, near-perfect at high duplication (consistent with
  Hubble's own MIA results). Min-K%++ generally best.

**Correctness predictors** (trained/calibrated only on clean calibration items):
- RoBERTa fine-tuned classifier on question text.
- Pretrained LLM + Platt scaling (Llama 3.1, Pythia 6.9B, Qwen3 8B) using the paired model's
  confidence on the correct answer as a difficulty signal.
- Generally low bias overall; bias grows in the "hard" contamination bin where perturbed and
  standard models diverge most.

## Key Results

- Correction estimators substantially beat naive under medium/high contamination (e.g. MMLU
  high random contamination: naive RMSE 13.1 vs combined estimator 1.8).
- **Combined estimator wins most often**, especially as contamination strength increases.
- IPW alone is a strong, cheap default — only needs a calibrated MIA.
- Under correlated contamination with hard items, IPW fails (selection bias); imputation/combined
  needed to recover clean score.
- At low contamination, naive is hard to beat but also doesn't hurt much (correction has high
  variance relative to small bias).
- Heuristic baseline (EPG, from Singh et al. 2024 — threshold MIA scores to maximize estimated
  performance gain / std error, no spiking needed) is inconsistent across benchmarks — motivates
  spiking's principled calibration.

## Practical Considerations (§5)

- **Sample efficiency**: Platt-scaled IPW (Min-K%++) needs as few as **~10 calibration examples**
  to be well calibrated (only 2 Platt-scaling parameters). Correctness predictors need hundreds.
- **Transfer across datasets**: memorization predictors (Min-K%, less so Min-K%++ due to
  dataset-specific normalization) calibrated on one benchmark transfer reasonably well to another.
  Notably, **calibrating on spiked Wikipedia passages** (not even a benchmark) transfers almost
  as well as in-domain calibration — suggests a single general-purpose spike set could serve many
  benchmarks.

## Limitations / Future Work

- Simulation only covers contamination via exact duplication (how Hubble inserts perturbations);
  real-world contamination also happens via paraphrase/near-duplicates — would need new paired
  models to study.
- Framed as a building block for technical AI governance: spiking trades off requiring developer
  cooperation (training-side intervention) for enabling much richer post-hoc audits than black-box
  access alone (echoes Steinke et al. 2023 canaries for DP auditing, and NIST AI 800-2 draft on
  automated benchmark evaluation practices).

## Relation to [[hubble]]

This paper is a direct downstream use of the Hubble model suite: it uses Hubble's standard vs.
perturbed minimal pairs as the simulation ground truth, and the same 5 inserted benchmarks
(WinoGrande, MMLU, HellaSwag, PIQA, PopQA) as the test beds for estimator evaluation. Where
Hubble's own paper studies memorization phenomena (when/how much models memorize),
this paper asks what to do about it once contamination has already happened, by adding a
training-time intervention (spiking) on top of the same experimental design.
