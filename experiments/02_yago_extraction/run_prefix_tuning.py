"""Supervised training-data extraction on a perturbed Hubble model via PREFIX TUNING.

The sibling `run.py` is the *unsupervised* baseline: prompt the frozen model, greedily decode, check
whether the UUID comes back. Here we give the attack a labeled budget. We split the inserted
biographies (members) into two halves: on one half — canaries whose UUID we already know — we learn
a shared prefix that steers the frozen model toward regurgitating the secret; on the held-out half
we measure how well that prefix generalizes. The dup=0 non-members ride along as a control: the
model never saw their UUIDs, so extraction there should stay ~0 no matter how good the prefix is.

Two caches make reruns cheap: the trained prefix adapter (the expensive GPU step) and the held-out
generations (so recomputing the metric is pure CPU, as in the baseline).
"""

import argparse
import json
import os
from dataclasses import dataclass

import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

import hubble

EXPERIMENT_DIR = os.path.dirname(__file__)

# Trainer logs here; set before it runs so the wandb run lands in the project's tracker rather than
# wandb's default "huggingface" project.
os.environ.setdefault("WANDB_PROJECT", "hubble-extraction")


@dataclass
class Config:
    dataset: str = "yago"  # the allegrolab/biographies_<dataset> set
    secret: str = "uuid"  # which meta field to extract (uuid is the cleanest: uniform-random)
    size: str = "1b"
    toks: str = "100b"
    condition: str = "perturbed"  # the target model (saw the inserted biographies)
    max_new_tokens: int = 24  # comfortably covers a ~19-token UUID; every extra token costs a pass
    num_virtual_tokens: int = 5  # length of the learned prefix; smaller -> less room to memorize secrets
    learning_rate: float = 1e-2  # well above the Trainer default (5e-5): prefix tuning trains from scratch
    epochs: int = 30  # None -> the Trainer default (3)
    # Only train on canaries the model actually memorized: at low duplication the model never encoded
    # the UUID, so its secret is as unguessable as a non-member's — training on it just feeds the
    # prefix irreducible-noise targets. We keep duplicates >= this threshold for the fit set.
    min_train_dup: int = 16
    test_size: float = 0.5  # half the canaries train the prefix, half are held out for evaluation
    seed: int = 42


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=Config.dataset)
parser.add_argument("--secret", default=Config.secret)
args = parser.parse_args()
config = Config(dataset=args.dataset, secret=args.secret)

# Caches are per (dataset, secret): the generation cache is matched to held-out records by line
# position, so it must never be read back for a different secret, dataset, or split.
ADAPTER_PATH = os.path.join(EXPERIMENT_DIR, "results", f"prefix_{config.dataset}_{config.secret}")
GEN_PATH = os.path.join(EXPERIMENT_DIR, "results", f"prefix_generations_{config.dataset}_{config.secret}.jsonl")
RESULTS_PATH = os.path.join(EXPERIMENT_DIR, "results", f"prefix_results_{config.dataset}_{config.secret}.json")
# Trainer writes its checkpoints and logs here; the fitted prefix is saved separately to ADAPTER_PATH.
CHECKPOINT_DIR = os.path.join(EXPERIMENT_DIR, "results", f"prefix_{config.dataset}_{config.secret}_trainer")


def load_base_model():
    """Load the target Hubble model — the one whose memorized UUIDs we are trying to extract."""
    repo_id = f"allegrolab/hubble-{config.size}-{config.toks}_toks-{config.condition}-hf"
    # NOTE: [edge case callout] 500B-token models need revision="step238500"; the 100B models we use
    # have a single final checkpoint, so the default revision is correct.
    model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    return model, tokenizer


records = hubble.load_biographies(config.dataset, config.secret)

# Split over canaries, stratified by duplication level so each level is represented in both halves.
# NOTE: [thought process] This split is the *attack's* train/test, NOT the model's training. Every
# member (train or test half) was inserted into the model and is memorized; holding out a half just
# means the prefix is judged on canaries it was never fit on — i.e. whether the steering generalizes.
train_records, test_records = train_test_split(
    records,
    test_size=config.test_size,
    random_state=config.seed,
    stratify=[record["duplicates"] for record in records],
)
# Sort the held-out set by id so the position-matched generation cache is stable across runs.
test_records = sorted(test_records, key=lambda record: record["id"])
# Only canaries the model actually memorized can teach extraction. A non-member's UUID was never in
# the model — but neither was a low-duplication member's, in practice — so training on either just
# pushes the prefix toward an unguessable random string and floods the gradient with noise. We keep
# only members duplicated >= min_train_dup, where the secret is genuinely encoded and recoverable.
fit_records = [record for record in train_records if record["duplicates"] >= config.min_train_dup]


# Reuse a cached run if possible: generations first (a pure-CPU rerun), then the trained prefix.
if os.path.exists(GEN_PATH):
    with open(GEN_PATH) as cache:
        for record, line in zip(test_records, cache):
            record["generation"] = json.loads(line)
else:
    model, tokenizer = load_base_model()
    if os.path.exists(ADAPTER_PATH):
        extractor = hubble.PrefixTuningExtractor.from_pretrained(ADAPTER_PATH, model, tokenizer)
    else:
        extractor = hubble.PrefixTuningExtractor(model, tokenizer, config.num_virtual_tokens)
        extractor.fit(
            fit_records, output_dir=CHECKPOINT_DIR, learning_rate=config.learning_rate, epochs=config.epochs
        )
        extractor.save(ADAPTER_PATH)  # cache the trained prefix so reruns skip the GPU training
    extractor.generate(test_records, config.max_new_tokens)
    with open(GEN_PATH, "w") as cache:
        for record in test_records:
            cache.write(json.dumps(record["generation"]) + "\n")


# Report extraction per duplication level on the HELD-OUT canaries, dup=0 as the control. We score
# two ways: verbatim (the whole UUID exactly) and token match (fraction of the UUID's tokens), the
# latter to surface partial recall that the all-or-nothing verbatim rate hides. Token match needs a
# tokenizer; we load just the tokenizer (no GPU) so a cache-only rerun can still score.
score_tokenizer = AutoTokenizer.from_pretrained(
    f"allegrolab/hubble-{config.size}-{config.toks}_toks-{config.condition}-hf"
)
dup_levels = sorted({record["duplicates"] for record in test_records})

results = []
for dup in dup_levels:
    subset = [record for record in test_records if record["duplicates"] == dup]
    results.append(
        {
            "dup": dup,
            "n": len(subset),
            "extraction_rate": hubble.extraction_rate(subset),
            "token_match": hubble.token_match_rate(subset, score_tokenizer),
        }
    )

with open(RESULTS_PATH, "w") as out:
    json.dump(results, out, indent=2)

print(f"{'dup':>5} {'n':>6} {'extract_rate':>14} {'token_match':>14}")
for result in results:
    print(f"{result['dup']:>5} {result['n']:>6} {result['extraction_rate']:>14.3f} {result['token_match']:>14.3f}")
