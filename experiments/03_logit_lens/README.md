# 03_logit_lens — Observations

## Goal
Exp 02 showed a sharp greedy-extraction threshold: **0** verbatim UUIDs below dup=16, ~80% at
dup=64. The dup=16 canaries are the puzzle — inserted 16 times, yet greedy decoding never reproduces
the UUID. This experiment asks whether the UUID is **absent** from the model or merely **present but
not promoted** to the final-layer argmax, using the **logit lens**.

## Idea
Greedy extraction reads only the model's final-layer top-1 token. The logit lens projects *every*
layer's hidden state through the model's own final norm + unembedding (`lm_head(norm(h_l))`), giving
a vocab distribution per layer. Teacher-forced on the true UUID, we record — for each secret token,
at each layer — the **rank** of the correct token (0 = top-1). If the correct token is top-1 at some
intermediate layer even when it loses the final decode, the memory is there, just buried.

- **Teacher forcing**: we feed `prefix + true_uuid` in one pass so every secret position is
  conditioned on the *correct* prior tokens. This isolates "is the next memorized token recoverable
  from internal state" from the compounding errors of free-running greedy decoding.
- **Lens mechanics**: 16-layer `LlamaForCausalLM`, untied embeddings → 17 hidden states (embeddings +
  16 blocks). Applying `norm` + `lm_head` to the *final* hidden state reproduces the real logits
  exactly, so the last layer is a built-in consistency check.
- **Control**: dup=0 (never inserted) must stay at chance at *every* layer. If it doesn't, the lens
  is manufacturing signal rather than reading memorization.

## Setup
- Model: `allegrolab/hubble-1b-100b_toks-perturbed-hf` (bf16, single GPU).
- Data: `allegrolab/biographies_yago`, secret = `uuid`, via `hubble.load_biographies`.
- All duplication levels {0,1,4,16,64,256}, each capped at `max_per_dup=300` canaries and sorted by
  id so the rank cache is position-stable.
- Core lens code is **local** to this experiment (`logit_lens.py`), not in `src/hubble/` — it stays
  here unless the signal proves real enough to promote.

## Metrics (per duplication level)
- `final@1`, `final@10`: fraction of secret tokens recovered at top-1 / top-10 using only the
  **final** layer (what greedy decoding sees).
- `any@1`, `any@10`: same, but using the **best (minimum) rank across all layers** (what the lens can
  find). The gap `any − final` is the headline: memory the final layer hides but an earlier layer
  holds.
- Figure: per-layer top-1 hit rate, one line per duplication level.

## Run
```bash
uv run python experiments/03_logit_lens/run.py
```
First run does one GPU pass and writes `results/lens_ranks_yago_uuid.jsonl` (plain integer ranks);
reruns reload the cache and recompute every metric + figure on CPU (no model needed).

## Results

`hubble-1b-100b_toks-perturbed`, 17 lens layers (embeddings + 16 blocks), ~22s for the GPU pass.

| dup | canaries | tokens | final@1 | final@10 | any@1 | any@10 |
|----:|---------:|-------:|--------:|---------:|------:|-------:|
|   0 |      300 |   5751 |   0.057 |    0.320 | 0.194 |  0.508 |
|   1 |      300 |   5821 |   0.076 |    0.357 | 0.207 |  0.529 |
|   4 |      300 |   5788 |   0.110 |    0.435 | 0.249 |  0.571 |
|  16 |      300 |   5754 |   0.387 |    0.735 | 0.469 |  0.770 |
|  64 |      179 |   3462 |   0.990 |    0.999 | 0.992 |  0.999 |
| 256 |       89 |   1712 |   1.000 |    1.000 | 1.000 |  1.000 |

**The cross-layer logit-lens hypothesis is _not_ supported.** The per-layer figure shows top-1
recovery rising **monotonically with depth and peaking at the final layer** for every duplication
level — there is no intermediate layer where the UUID surfaces and then gets buried. The UUID
representation is *assembled* through depth and read out at the end; the lens finds nothing earlier
that the final layer loses. So the `any@1 > final@1` gap at dup=16 (0.469 vs 0.387) is **not** real
buried memory: the dup=0 control shows the same gap (0.194 vs 0.057), i.e. it is the
multiple-comparison inflation of taking the best of 17 layers, plus the fact that hex tokens are a
small set the model can format-guess (dup=0 `final@10` is already 0.320). Because the cross-layer
idea didn't pan out, the lens code stays **local** to this experiment.

**The real finding is about teacher forcing, not depth.** Exp 02's *free-running greedy* gives **0%
verbatim** extraction at dup=16. But *teacher-forced* on the true UUID, the **final layer alone**
recovers **38.7%** of UUID tokens at top-1 and **73.5%** at top-10 — far above the dup=0 format-guess
baseline (0.057 / 0.320). So dup=16 UUIDs *are* substantially memorized; the 0% verbatim rate is an
artifact of (a) error compounding — one wrong token derails the rest of a free-running decode — and
(b) the all-or-nothing verbatim metric. This reframes the exp-02 "sharp threshold": the memory ramps
in smoothly with duplication (final@1: 0.057 → 0.076 → 0.110 → 0.387 → 0.990 → 1.000), and the
threshold is in *greedy extractability*, not in *presence*.

## Notes
- Implication for the supervised attack (exp 02 prefix tuning): since the per-token memory exists at
  dup=16 but free-running greedy can't chain it, an attack that reduces compounding error
  (teacher-forced scoring / beam search / reranking sampled candidates) is the natural next lever —
  more promising than anything reading intermediate layers.
- `any@*` metrics are inflated by the best-of-17-layers selection; judge intermediate layers by the
  per-layer figure, not by `any@*`. Kept in the table only to show the gap is a control artifact.
