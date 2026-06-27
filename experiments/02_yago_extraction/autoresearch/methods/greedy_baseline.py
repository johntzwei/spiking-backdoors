"""The reference method: plain greedy decoding, nothing learned.

This is the bar every other method must clear on the HELD-OUT split. It also doubles as the template
for the method interface: a class with `fit` (here a no-op) and `generate` (attach a candidate string
at record["generation"]), plus a `build(model, tokenizer)` factory the driver calls.
"""

from tqdm import tqdm

import hubble  # methods may freely use the editable library; only the harness reward is locked


class GreedyBaseline:
    def __init__(self, model, tokenizer, max_new_tokens=24, batch_size=128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def fit(self, train_records):
        pass  # nothing to learn

    def generate(self, records):
        for start in tqdm(range(0, len(records), self.batch_size), desc="greedy"):
            batch = records[start : start + self.batch_size]
            continuations = hubble.generate_continuations(
                self.model, self.tokenizer, [record["prefix"] for record in batch], self.max_new_tokens
            )
            for record, continuation in zip(batch, continuations):
                record["generation"] = continuation
        return records


def build(model, tokenizer):
    return GreedyBaseline(model, tokenizer)
