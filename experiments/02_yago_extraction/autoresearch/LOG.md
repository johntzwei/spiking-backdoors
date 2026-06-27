# Autoresearch log

One row per attempt. Verdict ∈ {win, no lift, overfit, invalid}. "win" = held_exact beats the
greedy baseline at some dup. Keep failures — they're the map.

| method | idea (one line) | held_exact (64 / 256) | held_token (16 / 64 / 256) | verdict | notes |
|--------|-----------------|-----------------------|-----------------------------|---------|-------|
| greedy_baseline | plain greedy decoding (the bar) | 0.789 / 1.000 | 0.038 / 0.831 / 1.000 | baseline | run first to fill the bar |
| prefix_tuning | learn a 5-token soft prompt on memorized canaries (dup>=16), lr 1e-2, 30 ep | 0.011 / 0.227 | 0.033 / 0.206 / 0.575 | overfit | train_exact 0.92/1.00 vs held 0.01/0.23 — huge gap; memorizes the supervision, far below greedy on held. dup=0 floor intact (0.025). |
| lora | LoRA (r=8, a=16) on q/v proj, dup>=16, lr 2e-4, 30 ep | 0.000 / 0.023 | 0.032 / 0.116 / 0.316 | overfit | fits train hardest of all (tr_exact 1.0 at 64/256, 0.59 at 16) but held worse than prefix (he_exact 0/0.023 at 64/256), far below greedy. More capacity -> more overfitting, no held lift. dup=0 floor intact. NB: needs lr 2e-4, not prefix's 1e-2 (1e-2 fits nothing); and fp32 trainable params (bf16 underflows to a flat loss) — fixed in src/hubble. |
| lora_abstain | lora + dup=0 negatives taught to emit EOS at pos 0 ("abstain"); 2x neg:pos, else same as lora | 0.000 / 0.000 | 0.002 / 0.025 / 0.076 | overfit | over-abstains: train still ~1.0 (memorized exact prefixes) but held collapses to ~0 token (empty/EOS outputs), WORSE than plain lora. The EOS gate keyed on the surface prefix, not the base model's recall signal, so on unseen names it just abstains. dup=0 floor intact (even lower). Idea was to use dup=0 as negatives to force a memory-gated readout; instead it learned prefix-specific abstain. Next: softer negative (entropy/KL-to-base) so it can't collapse to all-EOS, and/or fewer negatives. |
