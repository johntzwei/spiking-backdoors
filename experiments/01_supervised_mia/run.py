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
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from hubble.data import load_passages, split_items
from hubble.mia import LossThreshold, MinK, ReferenceAttack, token_log_probs
from hubble.supervised_mia import PrefixTuningMIA

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
# --dup / --attacks pick which cells to (re)compute this run; the full grid is always written, with
# untouched cells read back from the previous results file. Empty (default) = recompute everything.
parser.add_argument("--dup", type=int, nargs="*", default=[], help="dup levels to (re)compute")
parser.add_argument("--attacks", nargs="*", default=[], help="attack names to (re)compute")
args = parser.parse_args()
config = Config(dataset=args.dataset)

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


def load_classifier():
    """Load the target (perturbed) model with a fresh sequence-classification head, for the
    supervised prefix-tuning attack. fp32 (a 1B model is ~4GB) so the prefix and the randomly
    initialised head share the base dtype — no bf16 head/​hidden-state mismatch, no step underflow.
    """
    repo_id = f"allegrolab/hubble-{config.size}-{config.toks}_toks-{config.condition}-hf"
    model = AutoModelForSequenceClassification.from_pretrained(
        repo_id, num_labels=2, torch_dtype=torch.float32, device_map="cuda"
    )
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

# Same interface for every attack: `fit(train_items)` then `score(items)`. The score-based baselines
# read cached log-probs (a no-op `fit`); `prefix` instead trains a prefix-tuning classifier on the
# live model, so it ignores the cache and reads `item["text"]`.
# Group every prefix-tuning fit under one wandb project; each dup level becomes its own run, named
# per level in the loop below. WANDB_MODE=disabled (per CLAUDE.md) silences it without code changes.
os.environ.setdefault("WANDB_PROJECT", "01_supervised_mia")
attacks = {
    "loss": LossThreshold(),
    "mink": MinK(config.k),
    "reference": ReferenceAttack(),
    "prefix": PrefixTuningMIA(load_classifier, num_virtual_tokens=2, report_to="wandb"),
}

# Different datasets were inserted at different duplication levels (e.g. Gutenberg-popular
# only has 1/16/256), so the grid is whatever member levels this dataset actually contains.
dup_levels = config.dup_levels or tuple(
    sorted({record["duplicates"] for record in records if record["duplicates"] > 0})
)

# We always write the WHOLE grid (every attack x every dup). --attacks / --dup only pick which cells
# to (re)compute this run; every other cell is read back from the previous results file, which thus
# doubles as a cache — so an expensive prefix fit survives a later run that touches one other cell.
selected_attacks = set(args.attacks) or set(attacks)
selected_dups = set(args.dup) or set(dup_levels)
cache = (
    {row["dup"]: row for row in json.load(open(RESULTS_PATH))}
    if os.path.exists(RESULTS_PATH)
    else {}
)

# Only the score-based baselines read log-probs; attach those GPU passes (themselves disk-cached)
# only when we're actually recomputing one of them this run.
if selected_attacks & {"loss", "mink", "reference"}:
    attach_log_probs(records, LOG_PROBS_PATH, config.condition)
    attach_log_probs(records, REF_LOG_PROBS_PATH, config.ref_condition, key="ref_log_probs")

results = []
for dup in dup_levels:
    train_items, test_items = split_items(records, dup, config.test_size, config.seed)
    prior = cache.get(dup, {})
    result = {
        "dup": dup,
        "n_pos": sum(item["label"] for item in train_items + test_items),
        "n_neg": sum(1 - item["label"] for item in train_items + test_items),
    }
    # For each attack: recompute the cell if it was selected, else carry the cached AUCs forward (or
    # leave it blank if we've never computed it). Comparing train vs test AUC surfaces overfitting.
    train_labels = [item["label"] for item in train_items]
    test_labels = [item["label"] for item in test_items]
    for name, attack in attacks.items():
        if not (name in selected_attacks and dup in selected_dups):
            for col in (f"auc_train_{name}", f"auc_test_{name}"):
                if col in prior:
                    result[col] = prior[col]
            continue
        print(f"[dup={dup}] {name}: fit on {len(train_items)} / score on {len(test_items)}", flush=True)
        # Tag this fit's wandb run by dup level (attacks that don't train ignore the attribute).
        if hasattr(attack, "run_name"):
            attack.run_name = f"{name}_{config.dataset}_dup{dup}"
        attack.fit(train_items)
        result[f"auc_train_{name}"] = roc_auc_score(train_labels, attack.score(train_items))
        result[f"auc_test_{name}"] = roc_auc_score(test_labels, attack.score(test_items))
        print(f"[dup={dup}] {name}: auc_train={result[f'auc_train_{name}']:.3f} "
              f"auc_test={result[f'auc_test_{name}']:.3f}", flush=True)
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
        # A cell is blank ("—") only if this attack has never been computed at this dup level.
        cells = [
            f"{result[col]:.3f}" if col in result else "—"
            for col in (f"auc_train_{name}", f"auc_test_{name}")
        ]
        lines.append(
            f"| {result['dup']} | {result['n_pos']} | {result['n_neg']} | {cells[0]} | {cells[1]} |"
        )
    lines.append("")

markdown = "\n".join(lines)
with open(RESULTS_MD_PATH, "w") as out:
    out.write(markdown)
print(markdown)
