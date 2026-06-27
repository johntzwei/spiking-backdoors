"""Logit lens over a Hubble model's secret tokens — kept local to experiment 03.

The plain-extraction baseline (exp 02) reads only the model's FINAL-layer argmax: at dup=16 that
argmax is wrong, so verbatim extraction is 0. But "the final argmax is wrong" is not the same as
"the memorized token is absent." The logit lens projects every intermediate layer's hidden state
through the model's own final norm + unembedding, giving a vocab distribution per layer. We then ask
a sharper question: teacher-forced on the true secret, does the correct next token surface at the top
of SOME layer's distribution — i.e. is it present internally even though greedy decoding buries it?

We keep this here rather than in `src/hubble/` on purpose: it is exploratory. Only if the signal is
real does it earn a place in the shared library.
"""

import json
import os

import torch
from tqdm import tqdm


def lens_ranks(model, tokenizer, prefix, target):
    """Teacher-force `prefix + target`, return an (n_secret_tokens x n_layers) rank matrix.

    `ranks[j][l]` is the rank of the correct (j-th) secret token under layer `l`'s logit lens — 0
    means the correct token is that layer's top-1 prediction. `n_layers = hidden layers + 1` because
    `output_hidden_states` includes the embedding output as index 0 and each transformer layer after.
    """
    # Build the teacher-forced sequence: prefix tokens (with BOS) then the secret's own tokens. We
    # tokenize the secret with a leading space and no special tokens — exactly as exp 02's token_match
    # does — so the secret splits into the tokens it had in training (it carried a leading space there).
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)
    secret_ids = tokenizer(" " + target, add_special_tokens=False).input_ids
    secret_ids = torch.tensor([secret_ids], device=model.device)
    input_ids = torch.cat([prefix_ids, secret_ids], dim=1)
    # NOTE: [shape] input_ids: 1 x (prefix_len + n_secret). A single canary, batch size 1.

    with torch.no_grad():
        hidden_states = model(input_ids, output_hidden_states=True).hidden_states
    # NOTE: [shape] hidden_states: tuple of (n_layers) tensors, each 1 x seq_len x hidden_dim. Index 0
    # is the embedding output; index l>=1 is the output of transformer block l.

    # The prediction FOR secret token j is made at the position just before it. Secret token j sits at
    # absolute position prefix_len + j, so its predictor is position prefix_len + j - 1. The first
    # secret token is thus predicted from the last prefix position; we gather all predictors at once.
    prefix_len = prefix_ids.shape[1]
    n_secret = secret_ids.shape[1]
    predictor_positions = torch.arange(prefix_len - 1, prefix_len - 1 + n_secret, device=model.device)

    # Stack every layer's hidden state at the predictor positions, then run the logit lens once: the
    # model's own final norm followed by its unembedding. Applying these to the FINAL hidden state
    # reproduces the model's real logits exactly, so the last layer doubles as a consistency check.
    stacked = torch.stack([layer[0, predictor_positions] for layer in hidden_states])
    # NOTE: [shape] stacked: n_layers x n_secret x hidden_dim
    lens_logits = model.lm_head(model.model.norm(stacked))
    # NOTE: [shape] lens_logits: n_layers x n_secret x vocab_size

    # Rank of the correct token = how many vocab entries strictly outscore it (0 -> it is the argmax).
    target_ids = secret_ids[0]  # n_secret
    target_logits = lens_logits.gather(
        -1, target_ids.view(1, -1, 1).expand(lens_logits.shape[0], -1, 1)
    ).squeeze(-1)
    # NOTE: [shape] target_logits: n_layers x n_secret — the lens score of the correct token alone.
    ranks = (lens_logits > target_logits.unsqueeze(-1)).sum(dim=-1)
    # NOTE: [shape] ranks: n_layers x n_secret. We transpose to n_secret x n_layers so each row is one
    # secret token's rank profile across depth — the natural unit for the per-token metrics.
    return ranks.t().tolist()


def attach_lens_ranks(records, cache_path, load_model, key="ranks"):
    """Attach a per-secret-token rank matrix to every record under `key`, caching the GPU pass.

    Mirrors exp 02's `attach_generations`: `load_model` is a callable so the model is only built on a
    cache miss. The cache stores plain integers, so a rerun reloads them and computes every metric on
    CPU — no model or even tokenizer needed.

    NOTE: [edge case callout] Cache lines are matched to `records` by position, so the analyzed set
    must be built deterministically (we sort by id and cap per duplication level upstream); changing
    it means deleting the cache rather than reusing a stale one.
    """
    if os.path.exists(cache_path):
        with open(cache_path) as cache:
            for record, line in zip(records, cache):
                record[key] = json.loads(line)
        return records

    model, tokenizer = load_model()
    for record in tqdm(records, desc="logit lens"):
        record[key] = lens_ranks(model, tokenizer, record["prefix"], record["target"])

    with open(cache_path, "w") as cache:
        for record in records:
            cache.write(json.dumps(record[key]) + "\n")
    return records
