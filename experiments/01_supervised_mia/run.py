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

from hubble.data import attack_split, load_passages, zero_vs_dup
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
# --attacks picks which attacks to (re)compute this run; every other attack's column is read back
# from the previous results file (so an expensive prefix fit survives a rerun of the baselines).
# Empty (default) = recompute every attack. Duplication levels are always all reported (they are a
# reporting axis over the one held-out set, not a compute axis).
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

# --attacks picks which attacks to (re)compute; every other attack's column is read back from the
# previous results file, which doubles as a cache — so an expensive prefix fit survives a rerun that
# only touches the baselines. Duplication levels are always all reported (a slice of the one split).
selected_attacks = set(args.attacks) or set(attacks)
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

# One global split across all dup levels: the supervised attack fits on `train`, and EVERY attack is
# scored only on the held-out `test`, so nothing sees its own eval rows. We fit each attack once and
# score the whole test set once (keyed by id); per-dup AUC is then just a slice of those scores.
train_items, test_items = attack_split(records, config.test_size, config.seed)

scores_by_attack = {}
for name in selected_attacks:
    attack = attacks[name]
    print(f"{name}: fit on {len(train_items)} train / score {len(test_items)} held-out test", flush=True)
    if hasattr(attack, "run_name"):  # tag this fit's wandb run (attacks that don't train ignore it)
        attack.run_name = f"{name}_{config.dataset}"
    attack.fit(train_items)
    scores_by_attack[name] = dict(zip((item["id"] for item in test_items), attack.score(test_items)))

# Per duplication level, membership AUC on the held-out test set: `zero_vs_dup` slices out that
# level's members vs the shared dup=0 non-members. A recomputed attack is scored from
# `scores_by_attack`; any other attack's column is carried forward from the cache (or left blank).
results = []
for dup in dup_levels:
    eval_items, labels = zero_vs_dup(test_items, dup)
    prior = cache.get(dup, {})
    result = {"dup": dup, "n_pos": sum(labels), "n_neg": len(labels) - sum(labels)}
    for name in attacks:
        col = f"auc_{name}"
        if name in scores_by_attack:
            preds = [scores_by_attack[name][item["id"]] for item in eval_items]
            result[col] = roc_auc_score(labels, preds)
            print(f"[dup={dup}] {name}: auc={result[col]:.3f}", flush=True)
        elif col in prior:
            result[col] = prior[col]
    results.append(result)

with open(RESULTS_PATH, "w") as out:
    json.dump(results, out, indent=2)

# One Markdown table: a row per duplication level, a column per attack (held-out AUC). Iterates
# `attacks`, so adding an attack above extends the table without touching this block.
header = ["dup", "n_pos", "n_neg", *attacks]
lines = [
    f"# MIA results — {config.dataset} (held-out test split)",
    "",
    "| " + " | ".join(header) + " |",
    "| " + " | ".join(["---:"] * len(header)) + " |",
]
for result in results:
    cells = [str(result["dup"]), str(result["n_pos"]), str(result["n_neg"])]
    # A cell is blank ("—") only if this attack has never been computed.
    cells += [f"{result[f'auc_{name}']:.3f}" if f"auc_{name}" in result else "—" for name in attacks]
    lines.append("| " + " | ".join(cells) + " |")

markdown = "\n".join(lines)
with open(RESULTS_MD_PATH, "w") as out:
    out.write(markdown)
print(markdown)
