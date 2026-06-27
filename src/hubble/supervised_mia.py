"""Supervised membership inference on a Hubble model: train a classifier to predict membership.

The baselines in `mia.py` (Loss / Min-K% / Reference) read a *fixed* statistic off cached log-probs
— they never learn. A supervised attack instead *fits* on labeled passages: we freeze the model,
add a sequence-classification head, and steer it with a prefix-tuning adapter (Li & Liang 2021;
Ozdayi et al. 2023). It is the same PEFT scaffolding as `supervised_extraction.py`, but the
CAUSAL_LM head becomes a binary member/non-member head, so the attack keeps the MIA interface:

    fit(train_items)            # train the prefix + head on labeled passages
    score(items) -> scores      # P(member) per passage, higher = more likely member

Unlike the score-based baselines, this attack needs the live model (a GPU forward/backward pass),
not the cached log-probs — so it reads `item["text"]` directly and trains a fresh adapter per call.
"""

import torch
from peft import PrefixTuningConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments


class PrefixTuningMIA:
    """Membership classifier: a frozen model + classification head steered by a learned prefix.

    `fit` is re-runnable from scratch — it asks `model_loader` for a fresh frozen base each time and
    attaches a new adapter, so the same attack object can be fit independently on each dup level's
    task (as run.py reuses it across levels) without one level's prefix leaking into the next.
    """

    def __init__(
        self,
        model_loader,
        num_virtual_tokens=20,
        max_length=512,
        learning_rate=1e-2,
        epochs=10,
        batch_size=16,
        output_dir=None,
        report_to="none",
        run_name=None,
    ):
        # model_loader() -> (LlamaForSequenceClassification, tokenizer), built lazily in fit() so
        # constructing the attack (and listing it alongside the cheap CPU baselines) costs no GPU.
        self.model_loader = model_loader
        self.num_virtual_tokens = num_virtual_tokens
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        # Trainer always wants somewhere to write checkpoints; default to a scratch dir it can reuse.
        self.output_dir = output_dir or "/tmp/prefix_tuning_mia"
        # report_to / run_name flow straight to TrainingArguments — set report_to="wandb" to track,
        # and run_name to label each fit (a caller sweeping dup levels can retag before each fit()).
        self.report_to = report_to
        self.run_name = run_name
        self.model = None
        self.tokenizer = None

    def fit(self, train_items):
        """Attach a fresh prefix + classification head to a frozen base and train it on the labels."""
        print(f"[prefix] loading classifier and fitting on {len(train_items)} items "
              f"({self.epochs} epochs, lr={self.learning_rate})", flush=True)
        model, tokenizer = self.model_loader()
        # Llama has no pad token and no pad id; classification pools the last NON-pad token, so both
        # must be set or every sequence would pool the wrong position.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        # get_peft_model freezes the base, injects the prefix, and (for SEQ_CLS) keeps the new
        # classification head trainable. The base is loaded in fp32 by the loader, so the prefix and
        # head share its dtype — no bf16/​fp32 mismatch at the head and no small-step underflow.
        config = PrefixTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=self.num_virtual_tokens)
        self.model = get_peft_model(model, config)
        self.tokenizer = tokenizer

        # Right-padding for training; the attention mask keeps pad out of the forward, and last-token
        # pooling uses pad_token_id to find the real final token regardless of padding side.
        tokenizer.padding_side = "right"
        dataset = [self._encode(item) for item in train_items]
        collator = DataCollatorWithPadding(tokenizer)
        args = TrainingArguments(
            output_dir=self.output_dir,
            report_to=self.report_to,
            run_name=self.run_name,
            logging_steps=10,
            learning_rate=self.learning_rate,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
            data_collator=collator,
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
        for start in tqdm(range(0, len(items), self.batch_size), desc="scoring"):
            batch = items[start : start + self.batch_size]
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
