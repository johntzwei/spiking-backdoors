"""Plain training-data extraction on a perturbed Hubble model: YAGO biography UUIDs.

Each synthetic biography ends with "<Name> has the unique identifier <uuid>." We prompt the model
with the biography up to the UUID and ask: does greedy decoding reproduce the 32-hex-character
UUID verbatim? The UUID is uniform-random, so a correct continuation can only be memorization.
We generate once per biography (cached), then report the verbatim extraction rate per duplication
level. Expectation (per the Hubble paper): ~0 at dup=0 (never inserted), rising with duplication.
"""

import argparse
import json
import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import hubble

EXPERIMENT_DIR = os.path.dirname(__file__)


@dataclass
class Config:
    dataset: str = "yago"  # the allegrolab/biographies_<dataset> set
    secret: str = "uuid"  # which meta field to extract (uuid is the cleanest: uniform-random)
    size: str = "1b"
    toks: str = "100b"
    condition: str = "perturbed"  # the target model (saw the inserted biographies)
    max_new_tokens: int = 24  # just past a UUID's ~19 tokens; every extra token is another pass
    batch_size: int = 128  # prefixes generated together per GPU call (trades memory for speed)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=Config.dataset)
parser.add_argument("--secret", default=Config.secret)
args = parser.parse_args()
config = Config(dataset=args.dataset, secret=args.secret)

# Cache and results are per (dataset, secret): the generation cache is matched to records by line
# position, so a uuid cache must never be read back for a different secret or dataset.
GEN_PATH = os.path.join(EXPERIMENT_DIR, "results", f"generations_{config.dataset}_{config.secret}.jsonl")
RESULTS_PATH = os.path.join(EXPERIMENT_DIR, "results", f"extraction_results_{config.dataset}_{config.secret}.json")


def make_model_loader(condition):
    """Return a zero-arg loader for one Hubble condition, built only on a generation-cache miss."""
    def load_model():
        repo_id = f"allegrolab/hubble-{config.size}-{config.toks}_toks-{condition}-hf"
        # NOTE: [edge case callout] 500B-token models need revision="step238500" here; the 100B
        # models we use have a single final checkpoint, so the default revision is correct.
        model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        return model, tokenizer

    return load_model


records = hubble.load_biographies(config.dataset, config.secret)
# One cached GPU pass: greedily continue every biography's prefix and store the continuation.
hubble.attach_generations(
    records, GEN_PATH, make_model_loader(config.condition), config.max_new_tokens, config.batch_size
)

# Every duplication level present, including 0 (the test split, never inserted) as a control. We
# score two ways: verbatim (the whole UUID exactly) and token match (fraction of the UUID's tokens),
# the latter to surface partial recall the all-or-nothing verbatim rate hides. Token match needs a
# tokenizer; we load just the tokenizer (no GPU) so a cache-only rerun can still score.
score_tokenizer = AutoTokenizer.from_pretrained(
    f"allegrolab/hubble-{config.size}-{config.toks}_toks-{config.condition}-hf"
)
dup_levels = sorted({record["duplicates"] for record in records})

results = []
for dup in dup_levels:
    subset = [record for record in records if record["duplicates"] == dup]
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

# Print the extraction table: one row per duplication level.
print(f"{'dup':>5} {'n':>6} {'extract_rate':>14} {'token_match':>14}")
for result in results:
    print(f"{result['dup']:>5} {result['n']:>6} {result['extraction_rate']:>14.3f} {result['token_match']:>14.3f}")
