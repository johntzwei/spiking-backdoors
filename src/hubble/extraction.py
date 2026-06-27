"""Plain training-data extraction on a Hubble model: prompt with a canary's prefix, greedily
generate, and check whether the secret comes back verbatim.

This is the generation analogue of the MIA library. In MIA the model *scores* known text; here it
must *produce* unknown text. The one model-dependent step — greedy decoding from each prefix — is
`generate_continuations`; each experiment caches its output, after which the scoring metrics below
run on CPU and new metrics cost nothing to try.
"""

import torch


# --- the one model-dependent step: batched greedy generation ---


def generate_continuations(model, tokenizer, prefixes, max_new_tokens):
    """Greedily continue a *batch* of prefixes, returning only the newly generated text for each.

    Pad on the LEFT: decoding always continues from the rightmost position, so left-padding lines
    every prefix's last real token up at the same final column, and one forward pass advances every
    prefix at once. The attention mask tells the model to ignore the pads.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # input_ids: batch x max_prefix_len (shorter prefixes left-padded to the longest in the batch).
    inputs = tokenizer(prefixes, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: plain extraction reads off the single most likely continuation
            pad_token_id=tokenizer.eos_token_id,
        )
    # Left-padding put the generated tokens in the same trailing columns for every row, so one
    # slice at max_prefix_len peels the continuations off the whole batch at once.
    new_ids = output_ids[:, inputs.input_ids.shape[1]:]
    return tokenizer.batch_decode(new_ids, skip_special_tokens=True)


# --- metric: did the model reproduce the secret verbatim? ---


def verbatim_match(generation, target):
    """True if the model reproduced the secret exactly as the start of its continuation.

    The prefix was right-stripped before prompting (see `load_biographies`), so the model
    regenerates the separating space itself; we strip the generation's leading whitespace before
    comparing. We check `target` against the *start* of the continuation, not for exact equality,
    because the model keeps generating past the secret and we only care that it came out first.
    """
    return generation.strip()[: len(target)] == target


def token_match(generation, target, tokenizer):
    """Fraction of the secret's tokens the generation reproduces, position by position.

    Softens the all-or-nothing verbatim match: a UUID returned with one wrong character scores 0
    there, but the model often nails the first several tokens then derails, and token match credits
    that partial recovery. Both strings are re-tokenized with a leading space so the first secret
    token starts as it did in training; tokens the generation never produced count as misses.
    """
    target_ids = tokenizer(" " + target, add_special_tokens=False).input_ids
    generation_ids = tokenizer(" " + generation.strip(), add_special_tokens=False).input_ids
    matches = sum(t == g for t, g in zip(target_ids, generation_ids))
    return matches / len(target_ids)
