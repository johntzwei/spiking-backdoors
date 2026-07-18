"""Membership-inference attacks on a perturbed Hubble model.

For each duplication level we ask: can an attack tell passages the model was trained on (members,
inserted `dup` times) from passages it never saw (non-members, dup=0)? We run the model once to
cache per-token log-probs for every passage, then for each dup level score each attack on a
held-out item split and report AUC. Expectation (per the Hubble paper): near-chance at 1x, rising
as duplication grows.
"""

import argparse
import gc
import json
import os
from dataclasses import dataclass

import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from hubble.data import attack_split, load_passages, zero_vs_dup
from hubble.mia import LossThreshold, MinK, ReferenceAttack, token_log_probs
from hubble.supervised_mia import LoraMIA, PrefixTuningMIA

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
# --sweep switches to the low-expressivity hyperparameter sweep (capacity x weight_decay) for the
# supervised attacks, writing its own sweep_<dataset>.{json,md} and leaving the main table alone.
# `--sweep` alone sweeps both methods; `--sweep prefix` / `--sweep lora` restricts to one.
parser.add_argument("--sweep", nargs="*", default=None, choices=("prefix", "lora"), help="run the PEFT sweep")
args = parser.parse_args()
config = Config(dataset=args.dataset)
sweep_mode = args.sweep is not None

# Cache and results are per-dataset: the log-prob cache is matched to records by line position,
# so a wikipedia cache must never be read back for a Gutenberg run.
LOG_PROBS_PATH = os.path.join(EXPERIMENT_DIR, "results", f"log_probs_{config.dataset}.jsonl")
REF_LOG_PROBS_PATH = os.path.join(EXPERIMENT_DIR, "results", f"log_probs_{config.dataset}_ref.jsonl")
RESULTS_PATH = os.path.join(EXPERIMENT_DIR, "results", f"mia_results_{config.dataset}.json")
RESULTS_MD_PATH = os.path.join(EXPERIMENT_DIR, "results", f"mia_results_{config.dataset}.md")
SWEEP_PATH = os.path.join(EXPERIMENT_DIR, "results", f"sweep_{config.dataset}.json")
SWEEP_MD_PATH = os.path.join(EXPERIMENT_DIR, "results", f"sweep_{config.dataset}.md")


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


def training_args(output_dir):
    """Shared TrainingArguments for the supervised attacks; learning_rate is left at the HF default
    (5e-5). run.py retags each fit's wandb run via `attack.training_args.run_name` in the loop below.
    """
    return TrainingArguments(
        output_dir=output_dir,
        report_to="wandb",
        logging_steps=10,
        weight_decay=0.1,
        num_train_epochs=2,
        per_device_train_batch_size=32,
    )


attacks = {
    "loss": LossThreshold(),
    "mink": MinK(config.k),
    "reference": ReferenceAttack(),
    "prefix": PrefixTuningMIA(load_classifier, training_args("/tmp/prefix_tuning_mia"), num_virtual_tokens=2),
    "lora": LoraMIA(load_classifier, training_args("/tmp/lora_mia"), r=8, lora_alpha=16),
}


# --- PEFT sweep (--sweep) -------------------------------------------------------------------------
# The supervised attacks overfit (train AUC ~1.0, test ~0.5), so we sweep the LOW-expressivity end of
# each adapter — prefix `num_virtual_tokens` and LoRA rank `r` — crossed with `weight_decay`, to see
# whether shrinking capacity / adding regularization narrows the train-test gap. Learning rate is set
# explicitly per method to a value that actually converges: the HF default (5e-5) never moves LoRA.
SWEEP_EPOCHS = 3
SWEEP_CAPACITY = (1, 2, 4, 8)  # prefix virtual tokens / LoRA rank
SWEEP_WEIGHT_DECAY = (0.0, 0.1)
SWEEP_LR = {"prefix": 1e-3, "lora": 2e-4}


def sweep_configs(methods):
    """Yield (config_id, method, attack) over the capacity x weight_decay grid for each method."""
    for method in methods:
        for weight_decay in SWEEP_WEIGHT_DECAY:
            for capacity in SWEEP_CAPACITY:
                tag = "vt" if method == "prefix" else "r"
                config_id = f"{method}_{tag}{capacity}_wd{weight_decay}"
                ta = TrainingArguments(
                    output_dir=f"/tmp/sweep_{config_id}",
                    report_to="wandb",
                    run_name=f"sweep_{config_id}_{config.dataset}",
                    logging_steps=10,
                    learning_rate=SWEEP_LR[method],
                    weight_decay=weight_decay,
                    num_train_epochs=SWEEP_EPOCHS,
                    per_device_train_batch_size=32,
                )
                if method == "prefix":
                    attack = PrefixTuningMIA(load_classifier, ta, num_virtual_tokens=capacity)
                else:  # alpha = 2*r keeps the effective LoRA scale (alpha/r) fixed at 2.0 across ranks
                    attack = LoraMIA(load_classifier, ta, r=capacity, lora_alpha=2 * capacity)
                yield config_id, method, attack


def run_sweep(methods, train_items, test_items, dup_levels):
    """Fit every sweep config on `train`, score train+test, and record per-dup AUC on both splits.

    Each config loads a fresh ~4GB fp32 classifier, so we drop the fitted model and empty the CUDA
    cache between configs — otherwise 16 loads would exhaust the GPU. Rows are written incrementally
    so a crash mid-sweep keeps the configs already finished.
    """
    rows = []
    for config_id, method, attack in sweep_configs(methods):
        print(f"\n=== sweep {config_id} ===", flush=True)
        attack.fit(train_items)
        scored = train_items + test_items
        scores = dict(zip((item["id"] for item in scored), attack.score(scored)))
        row = {"config": config_id, "method": method}
        for dup in dup_levels:
            for split, split_items in (("train", train_items), ("test", test_items)):
                eval_items, labels = zero_vs_dup(split_items, dup)
                row[f"{split}_{dup}"] = roc_auc_score(labels, [scores[item["id"]] for item in eval_items])
            print(f"[dup={dup}] {config_id}: train={row[f'train_{dup}']:.3f} test={row[f'test_{dup}']:.3f}", flush=True)
        rows.append(row)
        del attack, scores
        gc.collect()
        torch.cuda.empty_cache()
        with open(SWEEP_PATH, "w") as out:
            json.dump(rows, out, indent=2)
    return rows


def sweep_table(rows, split, dup_levels):
    """One Markdown table: a row per config, a column per dup level, cells = AUC on `split`."""
    header = ["config", *[f"dup{dup}" for dup in dup_levels]]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---:"] * len(header)) + " |"]
    for row in rows:
        cells = [row["config"], *[f"{row[f'{split}_{dup}']:.3f}" for dup in dup_levels]]
        lines.append("| " + " | ".join(cells) + " |")
    return lines

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
# only when we're actually recomputing one of them this run (never in sweep mode — it fits only the
# supervised adapters, which read item["text"] directly).
if not sweep_mode and selected_attacks & {"loss", "mink", "reference"}:
    attach_log_probs(records, LOG_PROBS_PATH, config.condition)
    attach_log_probs(records, REF_LOG_PROBS_PATH, config.ref_condition, key="ref_log_probs")

# One global split across all dup levels: the supervised attack fits on `train`, and EVERY attack is
# scored only on the held-out `test`, so nothing sees its own eval rows. We fit each attack once and
# score the whole test set once (keyed by id); per-dup AUC is then just a slice of those scores.
train_items, test_items = attack_split(records, config.test_size, config.seed)

# Sweep mode fits the low-expressivity grid over this same split and writes its own tables, then
# stops — it deliberately does not touch the baseline mia_results files.
if sweep_mode:
    methods = args.sweep or ["prefix", "lora"]
    rows = run_sweep(methods, train_items, test_items, dup_levels)
    lines = [
        f"# PEFT sweep — {config.dataset}",
        "",
        f"lr(prefix)={SWEEP_LR['prefix']}, lr(lora)={SWEEP_LR['lora']}, {SWEEP_EPOCHS} epochs, batch 32. "
        "Capacity = prefix virtual tokens / LoRA rank.",
        "",
        "## Held-out (test attack split)",
        "",
        *sweep_table(rows, "test", dup_levels),
        "",
        "## Train attack split (overfitting check)",
        "",
        *sweep_table(rows, "train", dup_levels),
    ]
    markdown = "\n".join(lines)
    with open(SWEEP_MD_PATH, "w") as out:
        out.write(markdown)
    print(markdown)
    raise SystemExit(0)

# Fit on the train half, then score BOTH halves (keyed by id). The supervised prefix attack only
# ever fits on `train`, so comparing its train-split AUC (the rows it saw) against its held-out
# test-split AUC exposes overfitting; the score-based baselines don't fit, so their two columns
# should match up to sampling noise.
scores_by_attack = {}
for name in selected_attacks:
    attack = attacks[name]
    print(f"{name}: fit on {len(train_items)} train / score {len(train_items) + len(test_items)} (train+test)", flush=True)
    if hasattr(attack, "training_args"):  # tag this fit's wandb run (baselines don't train, no args)
        attack.training_args.run_name = f"{name}_{config.dataset}"
    attack.fit(train_items)
    scored = train_items + test_items
    scores_by_attack[name] = dict(zip((item["id"] for item in scored), attack.score(scored)))

# Per duplication level, membership AUC on each split: `zero_vs_dup` slices out that level's members
# vs the shared dup=0 non-members, once for the held-out test half and once for the train half. A
# recomputed attack is scored from `scores_by_attack`; any other attack's columns are carried
# forward from the cache (or left blank).
results = []
for dup in dup_levels:
    test_eval, test_labels = zero_vs_dup(test_items, dup)
    train_eval, train_labels = zero_vs_dup(train_items, dup)
    prior = cache.get(dup, {})
    result = {"dup": dup, "n_pos": sum(test_labels), "n_neg": len(test_labels) - sum(test_labels)}
    for name in attacks:
        for split, eval_items, labels in (("train", train_eval, train_labels), ("test", test_eval, test_labels)):
            col = f"auc_{split}_{name}"
            if name in scores_by_attack:
                preds = [scores_by_attack[name][item["id"]] for item in eval_items]
                result[col] = roc_auc_score(labels, preds)
            elif col in prior:
                result[col] = prior[col]
        if name in scores_by_attack:
            print(f"[dup={dup}] {name}: train={result[f'auc_train_{name}']:.3f} test={result[f'auc_test_{name}']:.3f}", flush=True)
    results.append(result)

with open(RESULTS_PATH, "w") as out:
    json.dump(results, out, indent=2)

# Two Markdown tables — held-out test AUC and train-split AUC — a row per duplication level and a
# column per attack. The gap between them is the overfitting check (see the prefix attack). Iterates
# `attacks`, so adding an attack above extends both tables without touching this block.
def auc_table(split):
    header = ["dup", "n_pos", "n_neg", *attacks]
    rows = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---:"] * len(header)) + " |"]
    for result in results:
        cells = [str(result["dup"]), str(result["n_pos"]), str(result["n_neg"])]
        # A cell is blank ("—") only if this attack has never been computed.
        cells += [f"{result[f'auc_{split}_{name}']:.3f}" if f"auc_{split}_{name}" in result else "—" for name in attacks]
        rows.append("| " + " | ".join(cells) + " |")
    return rows

lines = [
    f"# MIA results — {config.dataset}",
    "",
    "## Held-out (test attack split)",
    "",
    *auc_table("test"),
    "",
    "## Train attack split (overfitting check)",
    "",
    *auc_table("train"),
]

markdown = "\n".join(lines)
with open(RESULTS_MD_PATH, "w") as out:
    out.write(markdown)
print(markdown)
