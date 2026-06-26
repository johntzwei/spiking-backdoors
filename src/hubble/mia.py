"""Membership-inference attacks on a Hubble model: score passages, then guess membership.

Every attack shares one interface — `fit(train_items)` then `score(items) -> member_scores` —
so the only thing separating one attack from another is the statistic it reads off a passage's
log-probs. The single GPU step (per-token log-probs) is shared by all of them and cached, so once
the passages are scored the whole attack sweep runs cheaply on CPU.
"""

import json
import os

import torch
from sklearn.metrics import roc_auc_score


# --- the one model-dependent step: a forward pass, cached as per-passage log-probs ---


def token_log_probs(model, tokenizer, text):
    """Log-prob the model assigns to each realized next token in `text` (teacher forcing)."""
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    # NOTE: [shape] input_ids: 1 x sequence_len (batch of one passage).

    with torch.no_grad():
        logits = model(input_ids).logits
    # NOTE: [shape] logits: 1 x sequence_len x vocab_size — a next-token distribution at every
    # position. Position t predicts token t+1, so to score the realized tokens we line up
    # logits[:, :-1] (predictions) with input_ids[:, 1:] (the actual next tokens).
    log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)

    targets = input_ids[:, 1:]
    # NOTE: [shape] gather picks, at each position, the log-prob of the token that actually came
    # next: log_probs is 1 x (sequence_len-1) x vocab; targets is 1 x (sequence_len-1). unsqueeze
    # adds a trailing dim to index vocab, squeeze removes it again -> (sequence_len-1).
    chosen = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return chosen.squeeze(0)


def attach_log_probs(records, cache_path, load_model, key="log_probs"):
    """Attach a per-token log-prob tensor to every record under `key`, caching the GPU pass.

    NOTE: [thought process] `load_model` is a callable so the model only loads on a cache miss;
    a rerun reads the cache and runs the attacks on CPU. We cache raw log-probs, not a feature,
    so new features (e.g. a different Min-K% fraction) cost nothing to try.

    NOTE: [thought process] `key` lets the same routine attach a second model's log-probs under
    a different field — e.g. a reference model's scores at `key="ref_log_probs"` — so a calibrated
    attack can read both the target's and the reference's log-probs off one record.

    NOTE: [edge case callout] Cache lines are matched to `records` by position, so a change to
    the dataset or its load order means deleting the cache rather than reusing a stale one.
    """
    if os.path.exists(cache_path):
        with open(cache_path) as cache:
            for record, line in zip(records, cache):
                record[key] = torch.tensor(json.loads(line))
        return records

    model, tokenizer = load_model()
    for record in records:
        record[key] = token_log_probs(model, tokenizer, record["text"])

    with open(cache_path, "w") as cache:
        for record in records:
            cache.write(json.dumps(record[key].tolist()) + "\n")
    return records


# --- attacks: each turns a passage's log-probs into a "higher = more likely member" score ---


class LossThreshold:
    """Unsupervised baseline: rank passages by loss (mean NLL) alone (nothing to learn)."""

    def fit(self, train_items):
        pass

    def _loss(self, log_probs):
        """Mean negative log-likelihood. Lower = the model finds the passage more likely."""
        return -log_probs.mean().item()

    def score(self, items):
        # NOTE: [thought process] A member has LOWER loss, so we negate it: a higher score then
        # means "more likely member", matching every other attack's orientation.
        return [-self._loss(item["log_probs"]) for item in items]


class MinK:
    """Unsupervised baseline: rank passages by Min-K% alone (nothing to learn).

    NOTE: [pedagogical] Min-K% (Shi et al. 2024) looks only at the *most surprising* tokens
    (lowest log-prob). The intuition: an unseen passage has a few very-low-probability tokens,
    while a memorized one lifts even its hardest tokens. Averaging the worst tokens makes that
    gap stand out more than the full-passage mean (loss) does.
    """

    def __init__(self, k=0.2):
        self.k = k

    def fit(self, train_items):
        pass

    def _mink(self, log_probs):
        """Mean of the bottom-`k` fraction of token log-probs."""
        num_tokens = max(1, int(len(log_probs) * self.k))
        bottom = torch.topk(log_probs, num_tokens, largest=False).values
        return bottom.mean().item()

    def score(self, items):
        # A member has HIGHER Min-K% already, so no reorientation is needed.
        return [self._mink(item["log_probs"]) for item in items]


class ReferenceAttack:
    """Calibrated baseline: judge the target model's loss against a *reference* model's loss.

    NOTE: [pedagogical] Loss and Min-K% read only the target model, so a passage that is just
    intrinsically easy (boilerplate, common phrasing) looks like a member even when it isn't.
    The reference model — the *standard* Hubble run, trained on the same corpus but WITHOUT the
    insertions — never saw these passages, so its loss measures that intrinsic difficulty. The
    difference (target minus reference) cancels it: what's left is how much *more* the target
    likes the passage than a model that never trained on it — the memorization signal itself.
    This is the LiRA / "reference-model" attack (Carlini et al. 2022) in its simplest single-
    reference form, and it needs the standard model's log-probs attached at `ref_log_probs`.
    """

    def fit(self, train_items):
        pass

    def score(self, items):
        # NOTE: [thought process] Mean log-prob = -loss, so target_mean - ref_mean is exactly
        # (reference loss - target loss). A member's target loss drops below the reference's, so
        # the difference is positive: higher = more likely member, matching the other attacks.
        return [
            (item["log_probs"].mean() - item["ref_log_probs"].mean()).item()
            for item in items
        ]


def evaluate(method, train_items, test_items):
    """Fit `method` on the train items, then report its AUC on the held-out test items.

    For these unsupervised baselines `fit` is a no-op, so this is just "score the test items" —
    but the interface leaves room for an attack that actually learns from the train items.
    """
    method.fit(train_items)
    test_labels = [item["label"] for item in test_items]
    return roc_auc_score(test_labels, method.score(test_items))
