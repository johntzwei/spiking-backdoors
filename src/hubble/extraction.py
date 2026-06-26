"""Plain training-data extraction on a Hubble model: prompt with a canary's prefix, greedily
generate, and check whether the secret comes back verbatim.

This is the generation analogue of the MIA library. In MIA the model *scores* known text; here it
must *produce* unknown text. As in `mia.py`, the one model-dependent step — greedy decoding from
each prefix — is shared by every metric and cached, so once the continuations exist the scoring
runs on CPU and new metrics cost nothing to try.
"""

import json
import os

import torch
from tqdm import tqdm


# --- the one model-dependent step: greedy generation, cached as one continuation per record ---


def generate_continuations(model, tokenizer, prefixes, max_new_tokens):
    """Greedily continue a *batch* of prefixes, returning only the newly generated text for each.

    NOTE: [pedagogical] The whole point of batching here is GPU utilization. Greedy decoding is a
    sequence of `max_new_tokens` forward passes; with one prefix at a time the A6000 is mostly idle
    between tiny kernels. Padding many prefixes into a single batch lets one forward pass advance
    every prefix at once, turning ~(num_prefixes * max_new_tokens) passes into ~max_new_tokens.

    NOTE: [pedagogical] We pad on the LEFT. Decoding always continues from the rightmost position,
    so left-padding lines every prefix's last real token up at the same final column — the model
    generates the true next token for all of them. Right-padding would ask the model to continue
    from pad tokens, corrupting the output. The attention mask tells the model to ignore the pads.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prefixes, return_tensors="pt", padding=True).to(model.device)
    # NOTE: [shape] inputs.input_ids: batch x max_prefix_len (shorter prefixes left-padded to the
    # longest in the batch); attention_mask is the same shape, 0 on pad positions.

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: plain extraction reads off the single most likely continuation
            pad_token_id=tokenizer.eos_token_id,
        )
    # NOTE: [shape] output_ids: batch x (max_prefix_len + max_new_tokens). Because padding is on
    # the left, the generated tokens occupy the same trailing columns for every row, so one slice
    # at max_prefix_len peels the continuations off the whole batch at once.
    new_ids = output_ids[:, inputs.input_ids.shape[1]:]
    return tokenizer.batch_decode(new_ids, skip_special_tokens=True)


def attach_generations(records, cache_path, load_model, max_new_tokens=24, batch_size=128, key="generation"):
    """Attach a greedy continuation to every record under `key`, caching the GPU pass.

    Mirrors `attach_log_probs`: `load_model` is a callable so the model is only built on a cache
    miss, and a rerun reloads the cached continuations and scores them on CPU.

    NOTE: [performance improvement] `batch_size` trades GPU memory for speed — larger batches keep
    the GPU busier until memory runs out. `max_new_tokens` should be just past the secret's token
    length (a UUID is ~19 tokens), since every extra token is another forward pass per batch.

    NOTE: [edge case callout] Cache lines are matched to `records` by position, so changing the
    dataset or its load order means deleting the cache rather than reusing a stale one.
    """
    if os.path.exists(cache_path):
        with open(cache_path) as cache:
            for record, line in zip(records, cache):
                record[key] = json.loads(line)
        return records

    model, tokenizer = load_model()
    # Walk the records in batches; tqdm reports batches done, rate, and ETA to stderr.
    for start in tqdm(range(0, len(records), batch_size), desc="generating"):
        batch = records[start : start + batch_size]
        continuations = generate_continuations(model, tokenizer, [r["prefix"] for r in batch], max_new_tokens)
        for record, continuation in zip(batch, continuations):
            record[key] = continuation

    with open(cache_path, "w") as cache:
        for record in records:
            cache.write(json.dumps(record[key]) + "\n")
    return records


# --- metric: did the model reproduce the secret verbatim? ---


def verbatim_match(generation, target):
    """True if the model reproduced the secret exactly as the start of its continuation.

    NOTE: [thought process] The prefix was right-stripped before prompting (see `load_biographies`),
    so the model regenerates the separating space itself; we strip the generation's leading
    whitespace before comparing. We check `target` against the *start* of the continuation rather
    than requiring exact equality, because the model keeps generating past the secret (the next
    sentence, padding) and we only care that the secret came out first.
    """
    return generation.strip()[: len(target)] == target


def extraction_rate(records, key="generation"):
    """Fraction of records whose secret was extracted verbatim — the plain-extraction success rate."""
    matches = [verbatim_match(record[key], record["target"]) for record in records]
    return sum(matches) / len(matches)
