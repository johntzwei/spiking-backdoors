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
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from hubble.data import load_passages, split_items
from hubble.mia import LossThreshold, MinK, ReferenceAttack, token_log_probs

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
RESULTS_MD_PATH = os.path.join(EXPERIMENT_DIR, "results", f"mia_results_{config.dataset}.md")


def load_model(condition):
    """Load one Hubble condition's model and tokenizer (target "perturbed" or reference "standard")."""
    repo_id = f"allegrolab/hubble-{config.size}-{config.toks}_toks-{condition}-hf"
    # 500B-token models need revision="step238500"; the 100B models we use have a single final
    # checkpoint, so the default revision is correct.
    model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    return model, tokenizer


def attach_log_probs(records, cache_path, condition, key="log_probs"):
    """Attach a per-token log-prob tensor to every record under `key`, caching the GPU pass. Cache
    lines match records by position (so cache_path is keyed by dataset); a rerun reloads them and
    runs the attacks on CPU. `key` lets the reference model's log-probs land under "ref_log_probs".
    """
    if os.path.exists(cache_path):
        with open(cache_path) as cache:
            for record, line in zip(records, cache):
                record[key] = torch.tensor(json.loads(line))
        return

    model, tokenizer = load_model(condition)
    for record in records:
        record[key] = token_log_probs(model, tokenizer, record["text"])
    with open(cache_path, "w") as out:
        for record in records:
            out.write(json.dumps(record[key].tolist()) + "\n")


records = load_passages(config.dataset)
# Two GPU passes, each cached: the target model's log-probs feed Loss/Min-K%, and the reference
# (standard) model's log-probs — attached under "ref_log_probs" — calibrate the ReferenceAttack.
attach_log_probs(records, LOG_PROBS_PATH, config.condition)
attach_log_probs(records, REF_LOG_PROBS_PATH, config.ref_condition, key="ref_log_probs")

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
    train_items, test_items = split_items(records, dup, config.test_size, config.seed)
    result = {
        "dup": dup,
        "n_pos": sum(item["label"] for item in train_items + test_items),
        "n_neg": sum(1 - item["label"] for item in train_items + test_items),
    }
    # Fit each attack on the train items (a no-op for these unsupervised baselines), then score both
    # the train and held-out items and report AUC against their true membership labels. Comparing
    # train vs test AUC surfaces any overfitting once a *trained* attack slots into this harness.
    train_labels = [item["label"] for item in train_items]
    test_labels = [item["label"] for item in test_items]
    for name, attack in attacks.items():
        attack.fit(train_items)
        result[f"auc_train_{name}"] = roc_auc_score(train_labels, attack.score(train_items))
        result[f"auc_test_{name}"] = roc_auc_score(test_labels, attack.score(test_items))
    results.append(result)

with open(RESULTS_PATH, "w") as out:
    json.dump(results, out, indent=2)

# Build one Markdown table per attack: one row per duplication level, comparing train vs held-out
# AUC (iterates `attacks`, so adding an attack above extends the output without touching this block).
lines = [f"# MIA results — {config.dataset}", ""]
for name in attacks:
    lines.append(f"## {name}")
    lines.append("")
    lines.append("| dup | n_pos | n_neg | auc_train | auc_test |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        lines.append(
            f"| {result['dup']} | {result['n_pos']} | {result['n_neg']} |"
            f" {result[f'auc_train_{name}']:.3f} | {result[f'auc_test_{name}']:.3f} |"
        )
    lines.append("")

markdown = "\n".join(lines)
with open(RESULTS_MD_PATH, "w") as out:
    out.write(markdown)
print(markdown)
