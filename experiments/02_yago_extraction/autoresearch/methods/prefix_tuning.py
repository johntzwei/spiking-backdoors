"""Prefix tuning: learn a short soft prompt that steers the frozen model into regurgitating secrets.

This is the supervised counterpart to `greedy_baseline`. The baseline reads off the frozen model;
here we spend the train split's labels to fit a shared sequence of "virtual token" key/values
(Li & Liang 2021; the Ozdayi et al. 2023 extraction attack) and then decode with that prefix in
front. The harness fits on train and scores both splits, so the held-out numbers reveal whether the
steering *generalizes* to canaries it never saw, or merely memorized the supervision (overfitting,
the gap the INSTRUCTIONS warn about).

NOTE: [thought process] Two design choices carry over from the sibling run_prefix_tuning.py because
they are what made the attack train at all:
  - a *high* learning rate (1e-2, not the Trainer default 5e-5): the prefix is learned from scratch,
    not fine-tuned off pretrained weights, so it needs a much larger step.
  - fit only on *memorized* canaries (duplicates >= 16). A low-duplication member's UUID was never
    encoded in the model, so its target is as unguessable as random noise; training on it just floods
    the gradient with irreducible error. We keep only the canaries the model actually stored.
"""

import os

import hubble

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(__file__))

# --- knobs (mirrors run_prefix_tuning.py's Config; see the module docstring for the why) ---
NUM_VIRTUAL_TOKENS = 5  # length of the learned prefix; smaller -> less room to just memorize secrets
LEARNING_RATE = 1e-2  # prefix tuning trains from scratch, well above the Trainer default (5e-5)
EPOCHS = 30
MIN_TRAIN_DUP = 16  # only fit on canaries the model genuinely memorized
MAX_NEW_TOKENS = 24  # comfortably covers a ~19-token UUID; every extra token costs a forward pass


class PrefixTuning:
    def __init__(self, model, tokenizer):
        self.extractor = hubble.PrefixTuningExtractor(model, tokenizer, NUM_VIRTUAL_TOKENS)

    def fit(self, train_records):
        # Keep only memorized canaries: low-duplication members teach the prefix unguessable noise.
        fit_records = [record for record in train_records if record["duplicates"] >= MIN_TRAIN_DUP]
        checkpoint_dir = os.path.join(EXPERIMENT_DIR, "results", "prefix_tuning_trainer")
        self.extractor.fit(fit_records, output_dir=checkpoint_dir, learning_rate=LEARNING_RATE, epochs=EPOCHS)

    def generate(self, records):
        # PrefixTuningExtractor.generate already writes the continuation to record["generation"],
        # the key the harness scores.
        self.extractor.generate(records, max_new_tokens=MAX_NEW_TOKENS)
        return records


def build(model, tokenizer):
    return PrefixTuning(model, tokenizer)
