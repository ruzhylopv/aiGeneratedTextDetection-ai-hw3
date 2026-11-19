"""Qwen3-8B log-prob curves: windows from the 32-token chunk table, a p(AI) calibration, text-level statistics.

The cluster job (encoder.qwen_logprobs) stores, per text, sums of token log-probabilities and entropies over
consecutive 32-token chunks. Here they become:
- windows of 64 / 128 / 256 tokens every 32 tokens, with the mean log-prob and mean entropy of their tokens;
- a per-window p(AI): a logistic regression on (mean log-prob, mean entropy, their ratio), fitted on windows
  of the **detector set** only (human = 0, ai = 1). The result has the same columns as the detector curves,
  so encoder.curve_features applies unchanged;
- whole-text statistics (mean log-prob, mean entropy, ratio, spread of chunk means).
Only numpy, pandas and scikit-learn are needed.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CHUNK = 32
WINDOW_STATS = ["mean_logp", "mean_entropy", "logp_entropy_ratio"]
CALIBRATION_WINDOWS = 400_000  # detector-set windows used to fit the calibration (random, fixed seed)


def with_ratio(frame):
    """Add the ratio of surprise to entropy (Binoculars-style: low for text the model finds predictable)."""
    return frame.assign(logp_entropy_ratio=-frame["mean_logp"] / frame["mean_entropy"].clip(lower=1e-3))


def windows(chunks, size):
    """Windows of `size` tokens every CHUNK tokens, one row per window, from the chunk table of one set.

    Columns: id, position, is_last, tok_start, tok_end, char_start, char_end, n_tokens, mean_logp, mean_entropy.
    A text shorter than one window gives one window over all its chunks.
    """
    k = size // CHUNK
    chunks = chunks.sort_values(["id", "chunk"])
    ids = chunks["id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    cols = {c: chunks[c].to_numpy() for c in ["tok_start", "tok_end", "char_start", "char_end", "n_tokens",
                                                "n_scored", "sum_logp", "sum_entropy"]}
    cum = {c: np.r_[0, np.cumsum(cols[c])] for c in ["n_scored", "sum_logp", "sum_entropy"]}
    out = {name: [] for name in ["id", "position", "is_last", "first", "last"]}
    for start, end in zip(starts, ends):
        n_windows = max(end - start - k + 1, 1)
        first = start + np.arange(n_windows)
        last = np.minimum(first + k, end) - 1
        out["id"].append(np.repeat(ids[start], n_windows))
        out["position"].append(np.arange(n_windows))
        out["is_last"].append(np.arange(n_windows) == n_windows - 1)
        out["first"].append(first)
        out["last"].append(last)
    first, last = np.concatenate(out["first"]), np.concatenate(out["last"])
    n_scored = cum["n_scored"][last + 1] - cum["n_scored"][first]
    table = pd.DataFrame({
        "id": np.concatenate(out["id"]), "position": np.concatenate(out["position"]),
        "is_last": np.concatenate(out["is_last"]),
        "tok_start": cols["tok_start"][first], "tok_end": cols["tok_end"][last],
        "char_start": cols["char_start"][first], "char_end": cols["char_end"][last],
        "n_tokens": cols["n_tokens"][first],
        "mean_logp": (cum["sum_logp"][last + 1] - cum["sum_logp"][first]) / np.maximum(n_scored, 1),
        "mean_entropy": (cum["sum_entropy"][last + 1] - cum["sum_entropy"][first]) / np.maximum(n_scored, 1),
        "n_scored": n_scored,
    })
    return with_ratio(table[table["n_scored"] > 0].drop(columns="n_scored").reset_index(drop=True))


def fit_calibration(detector_windows, labels, seed=0):
    """Logistic regression p(AI | window stats), fitted on detector-set windows (labels: id -> human / ai)."""
    sample = detector_windows.sample(min(CALIBRATION_WINDOWS, len(detector_windows)), random_state=seed)
    y = (sample["id"].map(labels) == "ai").to_numpy()
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1_000))
    return model.fit(sample[WINDOW_STATS], y)


def with_p_ai(window_table, calibration):
    """The window table with a p_ai column from the calibration (the input format of curve_features)."""
    return window_table.assign(p_ai=calibration.predict_proba(window_table[WINDOW_STATS])[:, 1])


def text_stats(chunks):
    """Whole-text statistics per id: mean log-prob, mean entropy, their ratio, spread of chunk mean log-probs."""
    per_text = chunks.groupby("id").agg(n=("n_scored", "sum"), logp=("sum_logp", "sum"), ent=("sum_entropy", "sum"))
    chunk_means = chunks["sum_logp"] / chunks["n_scored"].clip(lower=1)
    stats = pd.DataFrame({
        "mean_logp": per_text["logp"] / per_text["n"].clip(lower=1),
        "mean_entropy": per_text["ent"] / per_text["n"].clip(lower=1),
        "chunk_logp_std": chunk_means.groupby(chunks["id"]).std().fillna(0.0),
    })
    return with_ratio(stats)
