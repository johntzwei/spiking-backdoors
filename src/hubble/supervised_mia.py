"""Supervised membership inference on a Hubble model: train a classifier to predict membership.

The baselines in `mia.py` (Loss / Min-K% / Reference) read a *fixed* statistic off cached log-probs
— they never learn. A supervised attack instead *fits* on labeled passages: we freeze the model,
add a sequence-classification head, and steer it with a PEFT adapter. Two adapters share one recipe
here (mirroring `supervised_extraction.py`): a prefix-tuning adapter (Li & Liang 2021; Ozdayi et al.
2023) that only prepends learned context, and a higher-capacity LoRA adapter (Hu et al. 2021) that
edits the attention projections themselves. Both keep the MIA interface:

    fit(train_items)            # train the adapter + head on labeled passages
    score(items) -> scores      # P(member) per passage, higher = more likely member

Unlike the score-based baselines, these attacks need the live model (a GPU forward/backward pass),
not the cached log-probs — so they read `item["text"]` directly and train a fresh adapter per call.
"""

import torch
from peft import LoraConfig, PrefixTuningConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import DataCollatorWithPadding, Trainer


class SupervisedMIA:
    """Membership classifier: a frozen model + classification head steered by a learned PEFT adapter.

    Holds the shared fit/encode/score loop; a subclass supplies only `_peft_config()` (which adapter
    to inject). Training is configured entirely by the caller's `TrainingArguments`. `fit` is
    re-runnable from scratch — it asks `model_loader` for a fresh frozen base each time and attaches a
    new adapter.
    """

    label = "supervised"

    def __init__(self, model_loader, training_args, max_length=512):
        # model_loader() -> (model, tokenizer), called lazily in fit() so building the attack (and
        # listing it beside the cheap CPU baselines) costs no GPU.
        self.model_loader = model_loader
        self.training_args = training_args
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

    def _peft_config(self):
        """Return the PEFT config (which adapter to inject); subclasses override."""
        raise NotImplementedError

    def fit(self, train_items):
        """Attach a fresh adapter + classification head to a frozen base and train it on the labels."""
        args = self.training_args
        print(f"[{self.label}] fitting on {len(train_items)} items ({args.num_train_epochs} epochs, "
              f"lr={args.learning_rate}, weight_decay={args.weight_decay})", flush=True)
        model, tokenizer = self.model_loader()
        # Llama has no pad token and no pad id; classification pools the last NON-pad token, so both
        # must be set or every sequence would pool the wrong position.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        # get_peft_model freezes the base, injects the adapter, and (for SEQ_CLS) keeps the new
        # classification head trainable. The base is loaded in fp32 by the loader, so the adapter and
        # head share its dtype — no bf16/​fp32 mismatch at the head and no small-step underflow.
        self.model = get_peft_model(model, self._peft_config())
        self.tokenizer = tokenizer

        # Right-padding for training; the attention mask keeps pad out of the forward, and last-token
        # pooling uses pad_token_id to find the real final token regardless of padding side.
        tokenizer.padding_side = "right"
        dataset = [self._encode(item) for item in train_items]
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset,
            data_collator=DataCollatorWithPadding(tokenizer),
        )
        trainer.train()
        return self

    def _encode(self, item):
        """Encode one passage as {input_ids, attention_mask, labels} for sequence classification."""
        encoding = self.tokenizer(item["text"], truncation=True, max_length=self.max_length)
        encoding["labels"] = item["label"]
        return encoding

    def score(self, items):
        """Return P(member) for each passage (higher = more likely a member), matching mia.py."""
        self.model.eval()
        scores = []
        for start in tqdm(range(0, len(items), self.training_args.per_device_eval_batch_size), desc="scoring"):
            batch = items[start : start + self.training_args.per_device_eval_batch_size]
            encoding = self.tokenizer(
                [item["text"] for item in batch],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.no_grad():
                logits = self.model(**encoding).logits
            # Column 1 is the "member" class; softmax turns the two logits into P(member).
            scores.extend(torch.softmax(logits.float(), dim=-1)[:, 1].tolist())
        return scores


class PrefixTuningMIA(SupervisedMIA):
    """Membership classifier steered by a learned prefix (Li & Liang 2021; Ozdayi et al. 2023).

    Learns key/value vectors injected into every attention layer's `past_key_values` — it only
    prepends learned context, leaving the model's weights untouched.
    """

    label = "prefix"

    def __init__(self, model_loader, training_args, num_virtual_tokens=20, **kwargs):
        self.num_virtual_tokens = num_virtual_tokens
        super().__init__(model_loader, training_args, **kwargs)

    def _peft_config(self):
        return PrefixTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=self.num_virtual_tokens)


class LoraMIA(SupervisedMIA):
    """Membership classifier steered by a low-rank weight update (Hu et al. 2021).

    Learns a rank-`r` update `B @ A` on the attention query/value projections (PEFT auto-selects them
    for Llama), editing the model's computation rather than just prepending context. `lora_alpha`
    scales the update (effective `alpha / r`); `r` and `lora_dropout` are the capacity/regularization
    knobs. LoRA edits weights directly, so it wants the standard LoRA learning rate (~1e-4..3e-4), NOT
    prefix tuning's 1e-2 — set it on the `TrainingArguments` you pass in.
    """

    label = "lora"

    def __init__(self, model_loader, training_args, r=8, lora_alpha=16, lora_dropout=0.0, **kwargs):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        super().__init__(model_loader, training_args, **kwargs)

    def _peft_config(self):
        return LoraConfig(
            task_type=TaskType.SEQ_CLS, r=self.r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout
        )
