"""Driver: run one extraction method through the locked harness.

Usage:
    uv run python experiments/02_yago_extraction/autoresearch/run_method.py --method greedy_baseline

Each method lives in methods/<name>.py and exposes a `build(model, tokenizer)` factory returning an
object with `fit(train_records)` and `generate(records)`. This driver just loads the model once,
builds the chosen method, and hands it to `harness.evaluate`, which does the split + scoring and
writes results/<name>.json. The driver is editable; the harness (the reward) is not.
"""

import argparse
import importlib
import os
import sys

# Methods that fit an adapter log training to wandb (report_to="wandb" in the library). Group those
# runs under one project rather than wandb's default "huggingface"; set WANDB_MODE=disabled to silence.
os.environ.setdefault("WANDB_PROJECT", "hubble-extraction")

# Make `harness` and the `methods` package importable when run from the repo root.
sys.path.insert(0, os.path.dirname(__file__))

import harness

parser = argparse.ArgumentParser()
parser.add_argument("--method", required=True, help="module name under methods/ (without .py)")
args = parser.parse_args()

model, tokenizer = harness.load_base_model()
module = importlib.import_module(f"methods.{args.method}")
method = module.build(model, tokenizer)
harness.evaluate(method, args.method, config={"model": harness.MODEL_ID})
