"""Supervised training-data extraction on a Hubble model: learn an attack on labeled canaries.

Unlike the unsupervised extractor in `extraction.py`, a supervised extractor fits on a labeled
subset of canaries (prefixes whose secret we know) before attacking the held-out rest. Every
extractor shares one interface:

    fit(train_records)            # learn from canaries whose `target` is known
    generate(records) -> records  # attach a continuation per record under `key`

so the metrics in `extraction.py` score them all the same way. The extractors differ only in which
PEFT adapter they attach to the frozen model — prefix tuning (Ozdayi et al. 2023) or LoRA — so
everything around the adapter (Trainer fitting, prefix masking, greedy decoding) lives once in
`SupervisedExtractor`, and each attack is a thin subclass supplying its config.
"""

from peft import LoraConfig, PeftModel, PrefixTuningConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from .extraction import generate_continuations


class SupervisedExtractor:
    """Freeze the model, attach a PEFT adapter, fit it on labeled canaries, then decode with it.

    `get_peft_model` freezes the base weights, attaches the adapter, and keeps `model.generate`
    working, so subclasses differ only in the `peft_config` they pass.
    """

    def __init__(self, model, tokenizer, peft_config):
        self.tokenizer = tokenizer
        # Wrapper that owns the adapter and freezes the base model, forwarding `.generate`/`.device`.
        self.model = get_peft_model(model, peft_config)
        # eval() disables dropout in the frozen base so its recall is clean and deterministic; the
        # trainable adapter params are unaffected.
        self.model.eval()

        # PEFT creates the adapter in the base dtype (bf16), but bf16's ~3 significant digits make
        # small AdamW steps underflow to zero (symptom: flat training loss). Cast the trainable
        # adapter params to fp32; the frozen base stays bf16. (No-op for prefix tuning, already fp32.)
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()

    @classmethod
    def from_pretrained(cls, adapter_path, model, tokenizer):
        """Rebuild a fitted extractor from a cached adapter, skipping training.

        Bypass `__init__` (which would build a fresh untrained adapter) via `__new__` and let PEFT
        load the learned weights onto the base model instead.
        """
        extractor = cls.__new__(cls)
        extractor.tokenizer = tokenizer
        extractor.model = PeftModel.from_pretrained(model, adapter_path)
        extractor.model.eval()
        return extractor

    def save(self, adapter_path):
        """Cache the trained adapter to `adapter_path` (just the adapter, not the base weights)."""
        self.model.save_pretrained(adapter_path)

    def fit(self, train_records, output_dir, learning_rate=None, epochs=None):
        """Learn the adapter with HF `Trainer` (AdamW, batched, linear LR schedule).

        Batching matters: averaging the gradient over a batch of canaries cancels the per-step noise
        of one-at-a-time updates (each UUID is a different random string). `learning_rate=None` keeps
        the Trainer default (5e-5), tuned for full fine-tuning; adapters learned from scratch need a
        much larger rate (~1e-2 for prefix tuning), so pass it explicitly when the default won't
        converge.
        """
        # Right-padding for training: `labels` masks the loss per-token, so pad sits harmlessly at
        # the end. (`generate` flips to left-padding, which decoding needs.)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dataset = [self._encode(record) for record in train_records]
        # Pads each batch to its longest example: input_ids with the pad token, labels with -100 so
        # pad positions never enter the loss.
        collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        # transformers 5.x defaults report_to to nothing; request wandb explicitly. logging_steps
        # makes the loss a curve. Reporting only — set WANDB_MODE=disabled to silence wandb.
        args = dict(output_dir=output_dir, report_to="wandb", logging_steps=10)
        if learning_rate is not None:
            args["learning_rate"] = learning_rate
        if epochs is not None:
            args["num_train_epochs"] = epochs
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(**args),
            train_dataset=dataset,
            data_collator=collator,
        )
        trainer.train()
        return self

    def _encode(self, record):
        """Encode one canary as {input_ids, labels} with the prefix masked out of the loss.

        We supervise only the target tokens — the prefix is context to condition on, not predict —
        by setting its labels to -100.
        """
        prefix_ids = self.tokenizer(record["prefix"]).input_ids
        # Restore the space between prefix and secret so the secret tokens carry the leading space
        # the model saw in training.
        input_ids = self.tokenizer(record["prefix"] + " " + record["target"]).input_ids
        # Treats input_ids[len(prefix_ids):] as the secret tokens, valid because the leading space
        # makes the boundary token start fresh (no merge across the prefix/secret boundary).

        # labels copies input_ids with prefix positions set to -100. PEFT injects the prefix as
        # past_key_values, not input tokens, so labels align with input_ids directly.
        labels = list(input_ids)
        labels[: len(prefix_ids)] = [-100] * len(prefix_ids)
        return {"input_ids": input_ids, "labels": labels}

    def generate(self, records, max_new_tokens=24, batch_size=128, key="generation"):
        """Attach a greedy continuation (with the learned adapter active) to every record.

        Reuses the unsupervised batched decoder; PEFT folds the adapter in, so decoding is the plain
        baseline with the adapter silently steering each step.
        """
        for start in tqdm(range(0, len(records), batch_size), desc="extracting"):
            batch = records[start : start + batch_size]
            continuations = generate_continuations(
                self.model, self.tokenizer, [record["prefix"] for record in batch], max_new_tokens
            )
            for record, continuation in zip(batch, continuations):
                record[key] = continuation
        return records


class PrefixTuningExtractor(SupervisedExtractor):
    """Steer extraction with a learned prefix (Li & Liang 2021; Ozdayi et al. 2023).

    Learns key/value vectors injected into every attention layer's `past_key_values`, reaching the
    deeper layers where memorized strings are recalled — unlike input-only `PromptTuningConfig`.
    """

    def __init__(self, model, tokenizer, num_virtual_tokens=20):
        config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=num_virtual_tokens)
        super().__init__(model, tokenizer, config)


class LoraExtractor(SupervisedExtractor):
    """Steer extraction with low-rank weight updates (Hu et al. 2021).

    Learns a rank-`r` update `B @ A` on the attention query/value projections (PEFT auto-selects them
    for Llama), editing the model's computation rather than just prepending context. `lora_alpha`
    scales the update (effective `alpha / r`). More capacity can overfit the train canaries' secrets,
    hurting held-out generalization — the held vs. train gap measures that tension.
    """

    def __init__(self, model, tokenizer, r=8, lora_alpha=16, lora_dropout=0.0):
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        super().__init__(model, tokenizer, config)


class AbstainLoraExtractor(LoraExtractor):
    """LoRA trained with an *abstention* target on non-memorized canaries (duplicates == 0).

    Plain LoRA overfits: it stores each `name -> UUID` mapping in its weights and confabulates on
    held-out names. Adding dup=0 canaries as negatives fixes this — their UUIDs are noise, so the
    only signal is "no recoverable secret", taught as an EOS at the one position right after the
    prefix (where a positive would start its UUID). Since dup=0 and memorized prefixes look identical
    there, satisfying both targets forces keying off the base model's internal recall signal (sharp
    vs. flat distribution), which is what we hope transfers. Abstaining at only that first position
    avoids teaching the model to truncate a UUID it has chosen to start.
    """

    def _encode(self, record):
        # Positives: supervise the UUID tokens as the parent does.
        if record["duplicates"] != 0:
            return super()._encode(record)

        # Negative (dup=0): supervise a single EOS right after the prefix, competing head-to-head
        # with a positive's first UUID token at the identical step.
        prefix_ids = self.tokenizer(record["prefix"]).input_ids
        input_ids = prefix_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * len(prefix_ids) + [self.tokenizer.eos_token_id]
        return {"input_ids": input_ids, "labels": labels}
