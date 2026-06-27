"""Membership-inference attacks on a Hubble model: score passages, then guess membership.

Every attack shares one interface — `fit(train_items)` then `score(items) -> member_scores` — so
the only thing separating one attack from another is the statistic it reads off a passage's
log-probs. The one model-dependent step (per-token log-probs) is `token_log_probs`; each experiment
caches its output, after which the attacks run on CPU and a new feature costs nothing to try.
"""

import torch


# --- the one model-dependent step: a forward pass to per-token log-probs ---


def token_log_probs(model, tokenizer, text):
    """Log-prob the model assigns to each realized next token in `text` (teacher forcing)."""
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        logits = model(input_ids).logits
    # Position t predicts token t+1, so line up logits[:, :-1] (predictions) with input_ids[:, 1:]
    # (the actual next tokens) to score the realized tokens.
    log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)

    targets = input_ids[:, 1:]
    # gather picks, at each position, the log-prob of the token that actually came next.
    chosen = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return chosen.squeeze(0)


# --- attacks: each turns a passage's log-probs into a "higher = more likely member" score ---


class LossThreshold:
    """Unsupervised baseline: rank passages by loss (mean NLL) alone (nothing to learn)."""

    def fit(self, train_items):
        pass

    def _loss(self, log_probs):
        """Mean negative log-likelihood. Lower = the model finds the passage more likely."""
        return -log_probs.mean().item()

    def score(self, items):
        # A member has LOWER loss, so negate it: a higher score then means "more likely member",
        # matching every other attack's orientation.
        return [-self._loss(item["log_probs"]) for item in items]


class MinK:
    """Unsupervised baseline: rank passages by Min-K% alone (nothing to learn).

    Min-K% (Shi et al. 2024) averages only the *most surprising* tokens (lowest log-prob): an unseen
    passage has a few very-low-probability tokens, while a memorized one lifts even its hardest
    tokens, and that gap stands out more than the full-passage mean (loss) does.
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

    Loss and Min-K% read only the target model, so an intrinsically easy passage (boilerplate,
    common phrasing) looks like a member even when it isn't. The reference model — the *standard*
    Hubble run, trained on the same corpus but WITHOUT the insertions — measures that intrinsic
    difficulty, and subtracting it leaves how much *more* the target likes the passage than a model
    that never trained on it. This is the LiRA / reference-model attack (Carlini et al. 2022) in its
    simplest single-reference form; it needs the standard model's log-probs attached at `ref_log_probs`.
    """

    def fit(self, train_items):
        pass

    def score(self, items):
        # Mean log-prob = -loss, so target_mean - ref_mean is (reference loss - target loss). A
        # member's target loss drops below the reference's, making the difference positive: higher =
        # more likely member, matching the other attacks.
        return [
            (item["log_probs"].mean() - item["ref_log_probs"].mean()).item()
            for item in items
        ]
