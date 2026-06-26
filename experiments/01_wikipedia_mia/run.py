"""Membership-inference attacks on a perturbed Hubble model.

For each duplication level we ask: can an attack tell passages the model was trained on (members,
inserted `dup` times) from passages it never saw (non-members, dup=0)? We run the model once to
cache per-token log-probs for every passage, then for each dup level score each attack on a
held-out item split and report AUC. Expectation (per the Hubble paper): near-chance at 1x, rising
as duplication grows.
"""

import argparse
import json
import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import hubble
from hubble import LossThreshold, MinK, ReferenceAttack, evaluate

EXPERIMENT_DIR = os.path.dirname(__file__)


@dataclass
class Config:
    # One script runs every passage set; pick which with --dataset (default from the env).
    dataset: str = os.environ.get("HUBBLE_DATASET", "wikipedia")
    size: str = "1b"
    toks: str = "100b"
    condition: str = "perturbed"  # the target model under attack (saw the insertions)
    ref_condition: str = "standard"  # the reference model (same corpus, no insertions)
    dup_levels: tuple = ()  # empty -> use every duplication level present in the data
    k: float = 0.2  # Min-K% fraction
    test_size: float = 0.5
    seed: int = 42


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=Config.dataset, choices=("wikipedia", "gutenberg_popular", "gutenberg_unpopular"))
config = Config(dataset=parser.parse_args().dataset)

# Cache and results are per-dataset: the log-prob cache is matched to records by line position,
# so a wikipedia cache must never be read back for a Gutenberg run.
LOG_PROBS_PATH = os.path.join(EXPERIMENT_DIR, "results", f"log_probs_{config.dataset}.jsonl")
REF_LOG_PROBS_PATH = os.path.join(EXPERIMENT_DIR, "results", f"log_probs_{config.dataset}_ref.jsonl")
RESULTS_PATH = os.path.join(EXPERIMENT_DIR, "results", f"mia_results_{config.dataset}.json")


def make_model_loader(condition):
    """Return a zero-arg loader for one Hubble condition (target "perturbed" or reference "standard").

    NOTE: [thought process] The loader is a closure so `attach_log_probs` only builds the model on
    a cache miss; we make one per condition so the target and reference passes share the same code.
    """
    def load_model():
        repo_id = f"allegrolab/hubble-{config.size}-{config.toks}_toks-{condition}-hf"
        # NOTE: [edge case callout] 500B-token models need revision="step238500" here; the 100B
        # models we use have a single final checkpoint, so the default revision is correct.
        model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        return model, tokenizer

    return load_model


records = hubble.load_passages(config.dataset)
# Two GPU passes, each cached: the target model's log-probs feed Loss/Min-K%, and the reference
# (standard) model's log-probs — attached under "ref_log_probs" — calibrate the ReferenceAttack.
hubble.attach_log_probs(records, LOG_PROBS_PATH, make_model_loader(config.condition))
hubble.attach_log_probs(records, REF_LOG_PROBS_PATH, make_model_loader(config.ref_condition), key="ref_log_probs")

# Different datasets were inserted at different duplication levels (e.g. Gutenberg-popular
# only has 1/16/256), so default to whatever member levels this dataset actually contains.
dup_levels = config.dup_levels or tuple(
    sorted({record["duplicates"] for record in records if record["duplicates"] > 0})
)

# Same interface for every attack: nothing to learn in `fit`, just a score per passage. Loss and
# Min-K% read only the target model; ReferenceAttack also reads the standard model's log-probs.
attacks = {"loss": LossThreshold(), "mink": MinK(config.k), "reference": ReferenceAttack()}

results = []
for dup in dup_levels:
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

# Print the AUC table: one row per duplication level, one column per attack (built from `attacks`
# so adding an attack above extends the table without touching this block).
header = f"{'dup':>5} {'n_pos':>6} {'n_neg':>6}" + "".join(f"{'auc_' + name:>14}" for name in attacks)
print(header)
for result in results:
    row = f"{result['dup']:>5} {result['n_pos']:>6} {result['n_neg']:>6}"
    row += "".join(f"{result['auc_' + name]:>14.3f}" for name in attacks)
    print(row)
