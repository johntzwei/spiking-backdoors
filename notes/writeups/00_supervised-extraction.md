# Supervised Extraction: Using Spiked Backdoors to Find Other Backdoors

*Draft writeup — 2026-06-27*

## 1. Introduction: generalizing membership inference and extraction

Membership inference (MIA) and training-data extraction are usually posed as *unsupervised*
black-box tests: read a model's log-probs or greedy decodings and threshold a fixed statistic
(LOSS, Min-K%, verbatim match). We propose to generalize this auditing program along two axes.

**First, make it supervised.** If the auditor holds a labeled set of examples that *are* known to be
in the training data, they can fit an attack on those labels rather than relying on a hand-designed
statistic. This is the same move [[spiking]] makes for contamination correction — insert known
examples at known duplication rates to calibrate a memorization predictor — but turned toward
*extraction*: use the labeled examples to learn how to pull memorized content back out of the model,
not just to score whether a given string was memorized.

**Second, go beyond verbatim training data toward higher-order information.** The literal goal of
extraction is to recover strings the model saw. But the same supervised machinery could target
information that was never a single training string — the model's *beliefs*, latent associations, or
aggregated facts assembled from many documents. Verbatim extraction is the cleanest first instance
of a broader question: *what can a supervised probe surface from a model's weights that a black-box
prompt cannot?*

This writeup develops the first instance concretely (supervised extraction of inserted secrets) and
points at the broader program.

## 2. A new application of spiking: backdoors finding backdoors

Concrete payoff for the supervised framing: **backdoor discovery**.

Suppose a model has backdoors — memorized trigger→payload associations — planted in its training
data, and a defender knows *some* of them (or can plant their own known backdoors via spiking). The
spiked/known backdoors form a labeled set. We can then:

1. **Spike** the training data with known backdoors (or use already-known ones as labels).
2. **Fit a supervised extractor** on those known backdoors — adapt the model so it is *more likely to
   surface memorized content* in general.
3. **Apply the extractor** to discover the *other*, pre-existing backdoors the defender did not know
   about.

The bet is that "surface a planted secret given its context" is a transferable skill: an adapter that
learns to read out the base model's stored associations on the labeled backdoors should also read out
unlabeled ones, since both rely on the same underlying memorization. This is a defensive,
audit-oriented use of spiking — the training-time intervention buys a much stronger post-hoc probe
than black-box access alone, echoing the governance framing in [[spiking]].

## 3. Method: supervised extraction

The core method is **supervised extraction**: rather than prompting a frozen model and reading greedy
decodings (the unsupervised baseline), we attach a small adapter (prefix tuning or LoRA) and *fit* it
on labeled canaries — examples whose secret we know — before attacking the held-out rest. The adapter
is trained to raise the likelihood of the memorized continuation; the base model's billions of
weights stay frozen, so the adapter is learning to *steer recall*, not to store the answers itself
(at least, that is the intent — see §5).

Shared interface across extractors: `fit(train_records)` then `generate(records)`, scored by the same
verbatim / token-match metrics as the unsupervised baseline, so each strategy is a thin adapter
choice.

## 4. Two problems: supervised MIA vs. supervised extraction

It is worth separating two tasks that the literature often blurs, because they differ sharply in
difficulty.

**Supervised MIA (easier — proof of concept).** Here the candidate sequence is given *in full*, and
the question is only binary: was it in the training data? Because the model can *score* the complete
string (teacher forcing), this is the easier problem and a clean place to validate the supervised
idea. Our Wikipedia MIA experiment is this proof of concept.

**Supervised training-data extraction (harder — the real goal).** Here the model must *produce*
unknown text, not score known text. We study a deliberately limited version first: the **prefix is
given** (e.g. the biography up to the secret), and the model must regenerate the secret. This isolates
recall from search.

**The unrestricted problem.** In a real extraction setting we may not have the prefix either — the
attacker must discover both the trigger/context *and* the payload. Bridging from "prefix given" to
"prefix unknown" is the main gap between our current testbed and genuine backdoor discovery.

| Task | Given | Model must | Difficulty |
|------|-------|-----------|------------|
| Supervised MIA | full sequence | score (binary) | easiest (PoC) |
| Supervised extraction, prefix given | prefix | produce secret | current focus |
| Supervised extraction, no prefix | nothing | find context + secret | open / real goal |

## Related notes
- [[spiking]] — spiking for test-set contamination correction (the source of the supervised-calibration idea).
- [[hubble]] — the standard/perturbed model suite and duplication-level design we build on.
