"""Sliding token windows over whole texts, with token and character offsets (for p(AI) curves).

A window of `size` tokens holds `size - 2` text tokens plus <s> and </s>. Window starts form a regular
grid every `step` tokens, and one last window is aligned to the end of the text, so the end (where the
AI part of a fusion text sits) is always covered. Texts shorter than one window give a single window.
A coarser curve (2 × step) is the even grid positions plus the last window, so one scoring pass at the
finest step serves every stride.

Only the tokenizer is needed here (no torch), so this module also runs on the laptop.
"""

import numpy as np
import pandas as pd

INDEX_COLUMNS = ["doc", "position", "is_last", "tok_start", "tok_end", "char_start", "char_end", "n_tokens"]


def window_starts(n_tokens, length, step):
    """First token of each window: every `step` tokens, plus one window that ends at the last token."""
    if n_tokens <= length:
        return [0]
    return list(range(0, n_tokens - length, step)) + [n_tokens - length]


def encode(tokenizer, texts, batch_size=1_000):
    """Token ids and character offsets of whole texts (no special tokens, no truncation)."""
    ids, offsets = [], []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(list(texts[start:start + batch_size]), add_special_tokens=False,
                        return_offsets_mapping=True, verbose=False)
        ids += [np.asarray(x, dtype=np.int32) for x in enc["input_ids"]]
        offsets += [np.asarray(x, dtype=np.int32).reshape(-1, 2) for x in enc["offset_mapping"]]
    return {"ids": ids, "offsets": offsets}


def make_windows(tokenizer, encoded, size, step, max_windows=None, seed=0):
    """Windows of every encoded text: input ids per window and an index table (INDEX_COLUMNS).

    With max_windows, a text with more windows keeps a random subset (fixed seed) that covers the
    whole text rather than only its start.
    """
    length = size - 2
    rng = np.random.default_rng(seed)
    input_ids, rows = [], []
    for doc, (ids, offsets) in enumerate(zip(encoded["ids"], encoded["offsets"])):
        starts = window_starts(len(ids), length, step)
        positions = range(len(starts))
        if max_windows is not None and len(starts) > max_windows:
            positions = np.sort(rng.choice(len(starts), max_windows, replace=False))
        for position in positions:
            start = starts[position]
            end = min(start + length, len(ids))
            input_ids.append(np.concatenate([[tokenizer.cls_token_id], ids[start:end], [tokenizer.sep_token_id]]))
            rows.append((doc, position, position == len(starts) - 1, start, end,
                         offsets[start, 0], offsets[end - 1, 1], len(ids)))
    return input_ids, pd.DataFrame(rows, columns=INDEX_COLUMNS)
