"""Membership-inference attacks on a perturbed Hubble model.

For each duplication level we ask: can an attack tell passages the model was trained on (members,
inserted `dup` times) from passages it never saw (non-members, dup=0)? We run the model once to
cache per-token log-probs for every passage, then for each dup level score each attack on a
held-out item split and report AUC. Expectation (per the Hubble paper): near-chance at 1x, rising
as duplication grows.
"""

import json
import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import hubble
from hubble import LossThreshold, MinK, evaluate

EXPERIMENT_DIR = os.path.dirname(__file__)
LOG_PROBS_PATH = os.path.join(EXPERIMENT_DIR, "results", "log_probs.jsonl")
RESULTS_PATH = os.path.join(EXPERIMENT_DIR, "results", "mia_results.json")


@dataclass
class Config:
    dataset: str = "wikipedia"  # any short name in hubble.data.DATASETS
    size: str = "1b"
    toks: str = "100b"
    condition: str = "perturbed"
    dup_levels: tuple = (1, 4, 16)
    k: float = 0.2  # Min-K% fraction
    test_size: float = 0.5
    seed: int = 42


config = Config()


def load_model():
    """Build the model on a cache miss. Passed to `attach_log_probs` so it never runs on a rerun."""
    repo_id = f"allegrolab/hubble-{config.size}-{config.toks}_toks-{config.condition}-hf"
    # NOTE: [edge case callout] 500B-token models need revision="step238500" here; the 100B
    # models we use have a single final checkpoint, so the default revision is correct.
    model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    return model, tokenizer


records = hubble.load_passages(config.dataset)
hubble.attach_log_probs(records, LOG_PROBS_PATH, load_model)

# Same interface for both baselines: nothing to learn in `fit`, just a score per passage.
attacks = {"loss": LossThreshold(), "mink": MinK(config.k)}

results = []
for dup in config.dup_levels:
    train_items, test_items = hubble.split_items(records, dup, config.test_size, config.seed)
    result = {
        "dup": dup,
        "n_pos": sum(item["label"] for item in train_items + test_items),
        "n_neg": sum(1 - item["label"] for item in train_items + test_items),
    }
    for name, attack in attacks.items():
        result[f"auc_{name}"] = evaluate(attack, train_items, test_items)
    results.append(result)

with open(RESULTS_PATH, "w") as out:
    json.dump(results, out, indent=2)

# Print the AUC table: one row per duplication level, one column per attack.
print(f"{'dup':>5} {'n_pos':>6} {'n_neg':>6} {'auc_loss':>9} {'auc_mink':>9}")
for result in results:
    print(f"{result['dup']:>5} {result['n_pos']:>6} {result['n_neg']:>6} "
          f"{result['auc_loss']:>9.3f} {result['auc_mink']:>9.3f}")
