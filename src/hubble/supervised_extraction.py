"""Supervised training-data extraction on a Hubble model: learn an attack on labeled canaries.

The plain extractor in `extraction.py` is *unsupervised* — it prompts the frozen model and reads
off greedy decoding, learning nothing. A supervised extractor instead gets a labeled subset of
canaries (prefixes whose secret we already know) and *fits* something on them before attacking the
held-out rest. Every extractor here shares one interface:

    fit(train_records)              # learn from canaries whose `target` is known
    generate(records) -> records    # attach a continuation per record under `key`

so the verbatim metric in `extraction.py` (`extraction_rate`) scores them all the same way, and a
new strategy is just a new class with these two methods.

Two options live here, and they differ *only* in which PEFT adapter they attach to the frozen model:
- **prefix tuning** (`PrefixTuningExtractor`): learn a short sequence of continuous "virtual token"
  key/values, shared across all canaries, injected into every attention layer — the Ozdayi et al.
  (2023) "controlling extraction via prompt-tuning" attack.
- **LoRA** (`LoraExtractor`): learn low-rank update matrices on the attention projections, again
  shared across canaries — the same supervised idea with a higher-capacity adapter.

Everything *around* the adapter — fitting with `Trainer`, masking the prefix out of the loss, batched
greedy decoding — is identical, so it lives once in `SupervisedExtractor` and each attack is a thin
subclass that supplies its PEFT config.

NOTE: [thought process] Room for more siblings: a learned reranker over sampled candidates, or a
soft prompt fit per duplication level. Each is another subclass that builds a different config.
"""

from peft import LoraConfig, PeftModel, PrefixTuningConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from hubble.extraction import generate_continuations


class SupervisedExtractor:
    """Shared protocol for adapter-based extraction: freeze the model, attach a PEFT adapter, fit it
    on labeled canaries, then decode with it in front.

    NOTE: [pedagogical] PEFT does the heavy lifting for every adapter type: `get_peft_model` freezes
    the base model's billions of weights, attaches the chosen adapter (prefix key/values, LoRA
    matrices, ...), threads it through each forward pass, and makes it work with `model.generate` —
    so decoding is the same batched greedy pass as the unsupervised baseline, only steered. Subclasses
    differ in *one line*: which `peft_config` they hand this constructor.
    """

    def __init__(self, model, tokenizer, peft_config):
        self.tokenizer = tokenizer
        # get_peft_model returns a wrapper that owns the adapter and freezes the base model; it
        # forwards `.generate`, `.device`, etc. so the rest of the code treats it as the model.
        self.model = get_peft_model(model, peft_config)
        # NOTE: [thought process] Keep everything in eval mode: the only trainable parameters live in
        # the adapter, so eval changes nothing for them, but it does switch off dropout in the frozen
        # base model — we want its recall to be the clean, deterministic one the attack is trying to
        # exploit, not a noised version.
        self.model.eval()

        # NOTE: [pedagogical] The base model is loaded in bf16, and PEFT creates each adapter in the
        # base's dtype — so LoRA's matrices land in bf16. bf16 has ~3 significant digits, so an AdamW
        # step far smaller than the weight underflows to zero and the adapter never moves (the symptom
        # is a perfectly flat training loss). Optimizer state must live in fp32. We cast the trainable
        # adapter params up to fp32; the frozen billions stay bf16. (Prefix tuning sidestepped this by
        # luck — its prefix-encoder embedding is created fresh in fp32 — so this cast is a no-op there.)
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()

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

    def fit(self, train_records, output_dir, learning_rate=None, epochs=None):
        """Learn one shared prefix with HF `Trainer` (AdamW, batched, linear LR schedule).

        NOTE: [thought process] We hand training to `Trainer` rather than a hand-rolled loop so the
        attack inherits the library's batching, optimizer, and linear LR schedule untouched. Batching
        is the point: averaging the gradient over a batch of canaries cancels most of the per-step
        noise that one-canary-at-a-time updates suffered from (each canary's UUID is a different
        random string, so its individual loss swings wildly). PEFT has already frozen the base model,
        so `Trainer` only ever updates the prefix parameters.

        NOTE: [thought process] `learning_rate=None` keeps the `TrainingArguments` default (5e-5),
        but that default is tuned for *full fine-tuning* of pretrained weights; prefix tuning learns
        a small set of parameters *from scratch* and needs a much larger rate (PEFT examples use
        ~1e-2). So this is the one knob worth overriding — pass it explicitly when the default fails
        to converge.
        """
        # Right-padding for training: the loss is masked per-token by `labels`, so padding sits
        # harmlessly at the end of each sequence under its attention mask. (`generate` flips this to
        # left-padding, which decoding needs — see `generate_continuations`.)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dataset = [self._encode(record) for record in train_records]
        # DataCollatorForSeq2Seq pads each batch to its longest example: input_ids with the pad
        # token (adding an attention mask), and labels with -100 so the pad positions never enter
        # the loss.
        collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        # NOTE: [edge case callout] transformers 5.x defaults `report_to` to nothing, so the wandb
        # logging the project relies on must be requested explicitly; `logging_steps=10` makes the
        # loss a curve rather than a single end-of-run point. These are reporting knobs only — they
        # don't touch the optimization. (Set WANDB_MODE=disabled to silence wandb entirely.)
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
        """Turn one canary into an {input_ids, labels} example with the prefix masked out of the loss.

        NOTE: [thought process] We supervise only the target tokens, not the prefix: the attack's
        job is to produce the secret given the biography, so the prefix is context to condition on,
        not something to predict. Setting the prefix labels to -100 restricts the loss to the secret.
        """
        prefix_ids = self.tokenizer(record["prefix"]).input_ids
        # The original biography put a space between prefix and secret; we restore it so the secret
        # tokens carry their leading space, exactly as the model saw them in training.
        input_ids = self.tokenizer(record["prefix"] + " " + record["target"]).input_ids
        # NOTE: [edge case callout] We treat `input_ids[len(prefix_ids):]` as the secret tokens,
        # assuming the prefix tokenizes the same alone as it does inside the full string. The leading
        # space on the secret makes the boundary token start fresh, so this holds for these
        # biographies; a tokenizer that merged across the boundary would need char->token alignment.

        # NOTE: [thought process] labels is a copy of input_ids with the prefix positions set to -100
        # (the "ignore" label). PEFT injects the prefix as past_key_values, not as extra input tokens,
        # so labels line up with input_ids directly — no shift for the prefix; HF's CausalLM then does
        # the usual internal shift, scoring each token from the one before.
        labels = list(input_ids)
        labels[: len(prefix_ids)] = [-100] * len(prefix_ids)
        return {"input_ids": input_ids, "labels": labels}

    def generate(self, records, max_new_tokens=24, batch_size=128, key="generation"):
        """Attach a greedy continuation (with the learned adapter active) to every record.

        Reuses the unsupervised batched decoder: PEFT folds the adapter into the model, so greedy
        decoding from each biography's prompt is identical to the plain baseline except the learned
        adapter is silently steering every step.
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

    NOTE: [pedagogical] `PrefixTuningConfig`'s learnable parameters are key/value vectors injected
    into every attention layer's `past_key_values`. The lighter `PromptTuningConfig` learns vectors
    at the input-embedding layer only — fewer parameters, but it can't reach the deeper layers where
    memorized strings are recalled. Prefix tuning's per-layer key/values give the steering more places
    to act, which is what we want for pulling a verbatim secret back out.
    """

    def __init__(self, model, tokenizer, num_virtual_tokens=20):
        config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=num_virtual_tokens)
        super().__init__(model, tokenizer, config)


class LoraExtractor(SupervisedExtractor):
    """Steer extraction with low-rank weight updates (Hu et al. 2021).

    NOTE: [pedagogical] LoRA learns a rank-`r` update `B @ A` added to chosen weight matrices (here
    the attention query/value projections, which PEFT auto-selects for the Llama architecture). Unlike
    prefix tuning, which only prepends context, LoRA edits the model's *computation* itself — a
    higher-capacity adapter. `lora_alpha` scales the update (effective scale `alpha / r`); the few MB
    of A/B matrices are the only trainable parameters, the base weights stay frozen.

    NOTE: [thought process] More capacity cuts both ways for this attack: it can fit the train
    canaries' secrets more aggressively, which on held-out canaries usually means *worse*
    generalization (more overfitting), not better extraction. That tension is exactly what the
    held vs. train gap measures.
    """

    def __init__(self, model, tokenizer, r=8, lora_alpha=16, lora_dropout=0.0):
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        super().__init__(model, tokenizer, config)


class AbstainLoraExtractor(LoraExtractor):
    """LoRA trained with an *abstention* target on non-memorized canaries (duplicates == 0).

    NOTE: [thought process] Plain `LoraExtractor` overfits: with only the memorized canaries to
    learn from, the adapter stores each `name -> UUID` mapping in its own weights instead of learning
    to *read out* a memory the base model already holds — so on held-out names it has never seen, it
    confabulates and collapses. The fix here is to add the dup=0 canaries as *negatives*. Their UUIDs
    were never inserted, so their content is irreducible noise (training to predict it just floods the
    gradient); the only usable signal is "this prefix has no recoverable secret". We encode exactly
    that and nothing more: at the one position right after the prefix — the same decision point where a
    positive must begin emitting its UUID — the negative is taught to emit EOS ("stop / abstain").

    NOTE: [pedagogical] The dup=0 and dup>=16 prefixes are surface-indistinguishable (just names), so
    the only way to satisfy both targets at that shared position is to key off the base model's
    *internal* recall signal — a sharp stored distribution vs. a flat one — rather than memorizing
    labels. That gate is what we hope transfers to held-out memorized canaries. Abstaining at the first
    position only (not across the whole UUID slot) keeps the negative from ever teaching the model to
    truncate a UUID it has chosen to start, which would clip the partial recoveries we want to keep.
    """

    def _encode(self, record):
        # Positives are encoded exactly as the parent does: supervise the UUID tokens.
        if record["duplicates"] != 0:
            return super()._encode(record)

        # Negative (dup=0): supervise a single EOS at the position right after the prefix, so the
        # abstention competes head-to-head with a positive's first UUID token at the identical step.
        prefix_ids = self.tokenizer(record["prefix"]).input_ids
        input_ids = prefix_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * len(prefix_ids) + [self.tokenizer.eos_token_id]
        return {"input_ids": input_ids, "labels": labels}
