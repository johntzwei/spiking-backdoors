"""Supervised training-data extraction on a Hubble model: learn an attack on labeled canaries.

The plain extractor in `extraction.py` is *unsupervised* — it prompts the frozen model and reads
off greedy decoding, learning nothing. A supervised extractor instead gets a labeled subset of
canaries (prefixes whose secret we already know) and *fits* something on them before attacking the
held-out rest. Every extractor here shares one interface:

    fit(train_records)              # learn from canaries whose `target` is known
    generate(records) -> records    # attach a continuation per record under `key`

so the verbatim metric in `extraction.py` (`extraction_rate`) scores them all the same way, and a
new strategy is just a new class with these two methods.

The first option is **prefix tuning**: freeze the model and learn a short sequence of continuous
"virtual token" vectors, shared across all canaries, that steer the frozen model into reproducing
the secret. This is the Ozdayi et al. (2023) "controlling extraction via prompt-tuning" attack.

NOTE: [thought process] Room for siblings here: a learned reranker over sampled candidates, or a
soft prompt fit per duplication level. Each would be another class exposing `fit`/`generate`.
"""

import torch
from peft import PeftModel, PrefixTuningConfig, TaskType, get_peft_model
from tqdm import tqdm

from hubble.extraction import generate_continuations


class PrefixTuningExtractor:
    """Learn a prefix that makes the frozen model regurgitate memorized secrets.

    NOTE: [pedagogical] We use HF PEFT's `PrefixTuningConfig` (Li & Liang 2021): the learnable
    parameters are key/value vectors injected into every attention layer's `past_key_values`, while
    the base model's billions of weights stay frozen. PEFT does the heavy lifting — it builds the
    prefix encoder, threads the learned key/values through each forward pass, and (crucially) makes
    them work with `model.generate`, so decoding is the same batched greedy pass as the unsupervised
    baseline. `get_peft_model` also freezes the base model and leaves only the prefix trainable.

    NOTE: [thought process] The lighter `PromptTuningConfig` learns vectors at the input-embedding
    layer only — fewer parameters, but it can't reach into the deeper layers where memorized strings
    are recalled. Prefix tuning's per-layer key/values give the steering more places to act, which
    is what we want for pulling a verbatim secret back out.
    """

    def __init__(self, model, tokenizer, num_virtual_tokens=20, epochs=3):
        self.tokenizer = tokenizer
        self.epochs = epochs

        config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=num_virtual_tokens)
        # get_peft_model returns a wrapper that owns the prefix encoder and freezes the base model;
        # it forwards `.generate`, `.device`, etc. so the rest of the code treats it as the model.
        self.model = get_peft_model(model, config)
        # NOTE: [thought process] Keep everything in eval mode: the only trainable parameters live in
        # the prefix encoder (a plain embedding, no dropout), so eval changes nothing for them, but
        # it does switch off dropout in the frozen base model — we want its recall to be the clean,
        # deterministic one the attack is trying to exploit, not a noised version.
        self.model.eval()

    @classmethod
    def from_pretrained(cls, adapter_path, model, tokenizer):
        """Rebuild a fitted extractor from a cached prefix adapter, skipping training entirely.

        NOTE: [thought process] `save_pretrained` writes only the prefix encoder (a few MB), not the
        frozen base model, so the cache is tiny. We sidestep `__init__` with `__new__` because it
        would build a *fresh, untrained* adapter via `get_peft_model`; here we want PEFT to load the
        learned weights onto the base model instead. This is the standard alternate-constructor idiom.
        """
        extractor = cls.__new__(cls)
        extractor.tokenizer = tokenizer
        extractor.model = PeftModel.from_pretrained(model, adapter_path)
        extractor.model.eval()
        return extractor

    def save(self, adapter_path):
        """Cache the trained prefix to `adapter_path` (just the adapter, not the base weights)."""
        self.model.save_pretrained(adapter_path)

    def fit(self, train_records, log=False):
        """Learn one shared prefix by minimizing NLL of the known secrets on train canaries.

        `log=True` streams the per-step loss to Weights & Biases (the project's tracker), and assumes
        the caller has already opened a run with `wandb.init`; the loss is also shown live in the
        progress bar regardless, so a long SLURM run isn't silent about whether it is converging.
        """
        # NOTE: [thought process] Import wandb lazily, only when logging is on, so the library has no
        # hard dependency on it — an offline experiment can fit a prefix without wandb installed.
        if log:
            import wandb

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        # NOTE: [thought process] No learning rate is passed, so Adam uses its own default (1e-3).
        # Prefer the library's default over a hand-tuned value: it's the maintained, documented
        # starting point, and it keeps this attack honest — any gain over the baseline comes from
        # the method, not from a learning rate quietly tuned on the canaries.
        optimizer = torch.optim.Adam(trainable)

        for epoch in range(self.epochs):
            # NOTE: [performance improvement] One canary per step keeps the loss code padding-free
            # and easy to read. Batching with left-padding (as `generate_continuations` does) and a
            # padded label mask would cut the number of forward passes substantially on a full set.
            progress = tqdm(train_records, desc=f"prefix-tuning epoch {epoch}")
            for record in progress:
                loss = self._target_loss(record)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress.set_postfix(loss=loss.item())
                if log:
                    wandb.log({"prefix_tuning/loss": loss.item()})
        return self

    def _target_loss(self, record):
        """Cross-entropy on the SECRET tokens only, with the learned prefix in front of the prompt.

        NOTE: [thought process] We supervise only the target tokens, not the prefix: the attack's
        job is to produce the secret given the biography, so the prefix is context to condition on,
        not something to predict. Masking the prefix to -100 restricts the loss to the secret.
        """
        device = self.model.device
        prefix_ids = self.tokenizer(record["prefix"], return_tensors="pt").input_ids
        # The original biography put a space between prefix and secret; we restore it so the secret
        # tokens carry their leading space, exactly as the model saw them in training.
        input_ids = self.tokenizer(
            record["prefix"] + " " + record["target"], return_tensors="pt"
        ).input_ids.to(device)
        prefix_len = prefix_ids.shape[1]
        # NOTE: [edge case callout] We treat `input_ids[prefix_len:]` as the secret tokens, assuming
        # the prefix tokenizes the same alone as it does inside the full string. The leading space on
        # the secret makes the boundary token start fresh, so this holds for these biographies; a
        # tokenizer that merged across the boundary would need an explicit char->token alignment.

        # NOTE: [shape] labels: 1 x seq_len, a copy of input_ids with the prefix positions set to
        # -100 (PyTorch's "ignore" label). PEFT injects the prefix as past_key_values rather than as
        # extra input tokens, so labels line up with input_ids directly — no shift for the prefix.
        # HF's CausalLM then does the standard internal shift, scoring each token from the one before.
        labels = input_ids.clone()
        labels[0, :prefix_len] = -100
        return self.model(input_ids=input_ids, labels=labels).loss

    def generate(self, records, max_new_tokens=24, batch_size=128, key="generation"):
        """Attach a greedy continuation (with the learned prefix in front) to every record.

        Reuses the unsupervised batched decoder: PEFT makes the prefix part of the model, so greedy
        decoding from each biography's prompt is identical to the plain baseline except the learned
        key/values are silently steering every step.
        """
        for start in tqdm(range(0, len(records), batch_size), desc="extracting"):
            batch = records[start : start + batch_size]
            continuations = generate_continuations(
                self.model, self.tokenizer, [record["prefix"] for record in batch], max_new_tokens
            )
            for record, continuation in zip(batch, continuations):
                record[key] = continuation
        return records
