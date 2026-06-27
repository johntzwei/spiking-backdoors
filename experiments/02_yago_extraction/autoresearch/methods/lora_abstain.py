"""LoRA + abstention: the overfitting fix for `lora`, using the dup=0 canaries as negatives.

`lora` collapses on held-out canaries because the adapter memorizes the train `name -> UUID`
mappings instead of learning to *read out* the base model's stored memory. This variant adds the
non-memorized canaries (duplicates == 0) as negatives: at the position right after the prefix — where
a memorized canary must begin emitting its UUID — a negative is taught to emit EOS instead ("abstain").

Since dup=0 and dup>=16 prefixes look identical (just names), the only way to get both targets right
is to gate on the base model's internal recall signal rather than the surface prefix — exactly the
generalizing behavior the plain LoRA lacked. The encoding lives in `hubble.AbstainLoraExtractor`; this
file only chooses which canaries to fit on.
"""

import os

import hubble

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(__file__))

# --- knobs (inherited from `lora`; see lora.py for why these specific values) ---
LORA_R = 8
LORA_ALPHA = 16
LEARNING_RATE = 2e-4  # LoRA's standard range, not prefix tuning's 1e-2
EPOCHS = 30
MIN_TRAIN_DUP = 16  # positives: only canaries the model genuinely memorized
NEG_PER_POS = 2  # cap negatives at this multiple of positives so they don't swamp the UUID signal
MAX_NEW_TOKENS = 24


class LoraAbstain:
    def __init__(self, model, tokenizer):
        self.extractor = hubble.AbstainLoraExtractor(model, tokenizer, r=LORA_R, lora_alpha=LORA_ALPHA)

    def fit(self, train_records):
        # Positives: memorized canaries whose UUID the model actually stored.
        positives = [record for record in train_records if record["duplicates"] >= MIN_TRAIN_DUP]
        # Negatives: non-members (dup=0), capped so the abstention loss stays a minority of the batch
        # and the positive UUID signal still dominates. The split is already shuffled, so a prefix
        # slice is an unbiased sample.
        negatives = [record for record in train_records if record["duplicates"] == 0]
        negatives = negatives[: NEG_PER_POS * len(positives)]
        fit_records = positives + negatives

        checkpoint_dir = os.path.join(EXPERIMENT_DIR, "results", "lora_abstain_trainer")
        self.extractor.fit(fit_records, output_dir=checkpoint_dir, learning_rate=LEARNING_RATE, epochs=EPOCHS)

    def generate(self, records):
        self.extractor.generate(records, max_new_tokens=MAX_NEW_TOKENS)
        return records


def build(model, tokenizer):
    return LoraAbstain(model, tokenizer)
