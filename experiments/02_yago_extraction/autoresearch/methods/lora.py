"""LoRA: learn low-rank weight updates that steer the frozen model into regurgitating secrets.

Same supervised recipe as `prefix_tuning`, with a higher-capacity adapter. Where prefix tuning only
prepends learned context, LoRA edits the attention projections themselves (Hu et al. 2021), so it can
fit the train canaries' secrets more aggressively. The open question this run answers: does that extra
capacity transfer to held-out canaries, or just overfit the supervision harder than prefix tuning did?

We keep the two choices that made the prefix attack train at all (see prefix_tuning.py): a high
learning rate (LoRA matrices are also learned from scratch) and fitting only on memorized canaries
(duplicates >= 16), since a low-duplication member's UUID was never encoded and is irreducible noise.
"""

import os

import hubble

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(__file__))

# --- knobs (LoRA capacity + the shared supervised-extraction settings) ---
LORA_R = 8  # rank of the low-rank update; larger -> more capacity (and more room to overfit)
LORA_ALPHA = 16  # update scale (effective alpha/r = 2.0)
# LoRA edits the attention projections directly, so it needs the standard LoRA range (~1e-4..3e-4),
# NOT prefix tuning's 1e-2 — at 1e-2 the adapter oscillates and never fits even the train secrets.
LEARNING_RATE = 2e-4
EPOCHS = 30
MIN_TRAIN_DUP = 16  # only fit on canaries the model genuinely memorized
MAX_NEW_TOKENS = 24  # comfortably covers a ~19-token UUID; every extra token costs a forward pass


class Lora:
    def __init__(self, model, tokenizer):
        self.extractor = hubble.LoraExtractor(model, tokenizer, r=LORA_R, lora_alpha=LORA_ALPHA)

    def fit(self, train_records):
        # Keep only memorized canaries: low-duplication members teach the adapter unguessable noise.
        fit_records = [record for record in train_records if record["duplicates"] >= MIN_TRAIN_DUP]
        checkpoint_dir = os.path.join(EXPERIMENT_DIR, "results", "lora_trainer")
        self.extractor.fit(fit_records, output_dir=checkpoint_dir, learning_rate=LEARNING_RATE, epochs=EPOCHS)

    def generate(self, records):
        # LoraExtractor.generate writes the continuation to record["generation"], the key the harness scores.
        self.extractor.generate(records, max_new_tokens=MAX_NEW_TOKENS)
        return records


def build(model, tokenizer):
    return Lora(model, tokenizer)
