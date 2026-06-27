"""Logit-lens probe of YAGO UUID memorization on a perturbed Hubble model.

Exp 02 found a sharp greedy-extraction threshold: nothing comes out below dup=16, then ~80% at
dup=64. The dup=16 canaries are the interesting middle — inserted 16 times, yet greedy decoding
reproduces 0 UUIDs verbatim. Is the UUID *absent* from the model, or *present but not promoted* to
the final-layer argmax? We teacher-force each canary on its true UUID and apply the logit lens: for
every secret token, at every layer, we record the rank of the correct token. If the correct token is
top-1 at some intermediate layer even when it loses the final decode, the memory is there — just
buried. dup=0 (never inserted) is the control: it must stay at chance at every layer, or the lens is
manufacturing signal rather than reading it.
"""

import argparse
import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import hubble
from logit_lens import attach_lens_ranks

EXPERIMENT_DIR = os.path.dirname(__file__)


@dataclass
class Config:
    dataset: str = "yago"  # the allegrolab/biographies_<dataset> set
    secret: str = "uuid"  # which meta field to probe (uuid is the cleanest: uniform-random)
    size: str = "1b"
    toks: str = "100b"
    condition: str = "perturbed"  # the target model (saw the inserted biographies)
    max_per_dup: int = 300  # cap canaries per duplication level so one GPU pass stays quick


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=Config.dataset)
parser.add_argument("--secret", default=Config.secret)
args = parser.parse_args()
config = Config(dataset=args.dataset, secret=args.secret)

RANKS_PATH = os.path.join(EXPERIMENT_DIR, "results", f"lens_ranks_{config.dataset}_{config.secret}.jsonl")
RESULTS_PATH = os.path.join(EXPERIMENT_DIR, "results", f"lens_results_{config.dataset}_{config.secret}.json")
FIGURE_PATH = os.path.join(EXPERIMENT_DIR, "figures", f"lens_top1_by_layer_{config.dataset}_{config.secret}.png")
os.makedirs(os.path.dirname(RANKS_PATH), exist_ok=True)
os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)


def load_model():
    """Load the target Hubble model — the one whose memorized UUIDs we are lensing."""
    repo_id = f"allegrolab/hubble-{config.size}-{config.toks}_toks-{config.condition}-hf"
    # NOTE: [edge case callout] 500B-token models need revision="step238500"; the 100B models we use
    # have a single final checkpoint, so the default revision is correct.
    model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    return model, tokenizer


# Build the analyzed set deterministically: every duplication level, each sorted by id and capped, so
# the position-matched rank cache stays stable across runs. dup=0 rides along as the control.
records = hubble.load_biographies(config.dataset, config.secret)
dup_levels = sorted({record["duplicates"] for record in records})
analyzed = []
for dup in dup_levels:
    subset = sorted((r for r in records if r["duplicates"] == dup), key=lambda r: r["id"])
    analyzed.extend(subset[: config.max_per_dup])

# One cached GPU pass: teacher-force every canary and store its per-token rank-by-layer matrix.
attach_lens_ranks(analyzed, RANKS_PATH, load_model)


# --- scoring (pure CPU from the cached ranks) ---
# A token is "recovered" at top-k if its rank < k. We summarize each duplication level two ways:
#   final  -> using only the FINAL layer's rank (what greedy decoding actually sees)
#   anywhere -> using the best (minimum) rank across ALL layers (what the lens can find)
# The gap between them is the headline: memory the final layer hides but an earlier layer holds.
def rate(token_ranks, k):
    return sum(rank < k for rank in token_ranks) / len(token_ranks)


results = []
per_layer_curves = {}
for dup in dup_levels:
    subset = [r for r in analyzed if r["duplicates"] == dup]
    # Flatten to one entry per secret token; each entry is that token's rank profile across layers.
    token_profiles = [profile for record in subset for profile in record["ranks"]]
    final_ranks = [profile[-1] for profile in token_profiles]
    best_ranks = [min(profile) for profile in token_profiles]

    n_layers = len(token_profiles[0])
    # Per-layer top-1 hit rate: at each depth, the fraction of secret tokens the lens ranks first.
    per_layer_curves[dup] = [
        sum(profile[layer] == 0 for profile in token_profiles) / len(token_profiles)
        for layer in range(n_layers)
    ]

    results.append(
        {
            "dup": dup,
            "n_canaries": len(subset),
            "n_tokens": len(token_profiles),
            "final_top1": rate(final_ranks, 1),
            "final_top10": rate(final_ranks, 10),
            "anywhere_top1": rate(best_ranks, 1),
            "anywhere_top10": rate(best_ranks, 10),
        }
    )

with open(RESULTS_PATH, "w") as out:
    json.dump(results, out, indent=2)


# --- per-layer top-1 figure: one line per duplication level, layer 0 (embeddings) .. last (final) ---
plt.figure(figsize=(7, 5))
for dup in dup_levels:
    plt.plot(range(len(per_layer_curves[dup])), per_layer_curves[dup], marker="o", label=f"dup={dup}")
plt.xlabel("layer (0 = embeddings, last = final)")
plt.ylabel("secret-token top-1 hit rate")
plt.title(f"Logit-lens recovery of {config.secret} tokens by layer")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_PATH, dpi=150)


# Print the summary table: final-layer recovery vs best-across-layers recovery, per duplication level.
header = f"{'dup':>5} {'canaries':>9} {'tokens':>7} {'final@1':>8} {'final@10':>9} {'any@1':>7} {'any@10':>7}"
print(header)
for result in results:
    print(
        f"{result['dup']:>5} {result['n_canaries']:>9} {result['n_tokens']:>7} "
        f"{result['final_top1']:>8.3f} {result['final_top10']:>9.3f} "
        f"{result['anywhere_top1']:>7.3f} {result['anywhere_top10']:>7.3f}"
    )
