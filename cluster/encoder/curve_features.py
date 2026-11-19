"""Features of p(AI) curves (one row per text), shared by the gate check and the stacker notebooks.

Input is a window table as written by encoder.curves: one row per window with id, position, is_last,
tok_start, tok_end, n_tokens and p_ai, at one window size. Only numpy and pandas are needed.
"""

import numpy as np
import pandas as pd

HIGH = 0.5          # a window is "high" (AI-like) above this p(AI)
HEAD_TAIL = 0.25    # head / tail = windows whose centre lies in the first / last 25% of the text

BASIC = ["mean", "max", "std", "frac_high", "longest_high_share", "mean_abs_diff", "n_switches"]
CHANGEPOINT = ["step_gain", "step_pos", "head_mean", "tail_mean", "tail_minus_head", "last_p"]
SEGMENT = ["seg_share", "seg_start", "seg_end", "seg_mean", "seg_contrast"]
CONTEXT = ["n_windows", "log_tokens"]
FEATURES = BASIC + CHANGEPOINT + SEGMENT + CONTEXT
GRID = np.linspace(0, 1, 21)  # common positions for comparing curves of different window sizes


def subsample(curves, factor):
    """A coarser curve: every `factor`-th grid window plus the end-aligned last window of each text."""
    return curves[(curves["position"] % factor == 0) | curves["is_last"]]


def longest_run(mask):
    """Start and end index (exclusive) of the longest run of True in a boolean array, or (0, 0)."""
    best, start = (0, 0), None
    for i, value in enumerate(np.append(mask, False)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            best = max(best, (start, i), key=lambda run: run[1] - run[0])
            start = None
    return best


def best_step(p):
    """Largest mean(after) - mean(before) over all split points of the curve, and the split index."""
    if len(p) < 2:
        return 0.0, 0
    prefix = np.cumsum(p)
    k = np.arange(1, len(p))
    gain = (prefix[-1] - prefix[k - 1]) / (len(p) - k) - prefix[k - 1] / k
    best = int(np.argmax(gain))
    return float(gain[best]), best + 1


def best_segment(p):
    """Contiguous windows (start, end exclusive) maximising the sum of p - HIGH, or (0, 0) if none is positive.

    This is the most AI-like stretch of the text: a fusion text's rewritten part is one such stretch,
    often in the middle (the gate curves drop back to low after it).
    """
    best_sum, best, current_sum, start = 0.0, (0, 0), 0.0, 0
    for i, value in enumerate(p - HIGH):
        if current_sum <= 0:
            current_sum, start = 0.0, i
        current_sum += value
        if current_sum > best_sum:
            best_sum, best = current_sum, (start, i + 1)
    return best


def segment_features(p, start, end, n_tokens):
    """Position, length and contrast of the most AI-like stretch (all 0 if no window is above HIGH)."""
    seg_start, seg_end = best_segment(p)
    if seg_end == seg_start:
        return {name: 0.0 for name in SEGMENT}
    inside = p[seg_start:seg_end]
    outside = np.concatenate([p[:seg_start], p[seg_end:]])
    return {"seg_share": (end[seg_end - 1] - start[seg_start]) / n_tokens,
            "seg_start": start[seg_start] / n_tokens, "seg_end": end[seg_end - 1] / n_tokens,
            "seg_mean": inside.mean(), "seg_contrast": inside.mean() - (outside.mean() if len(outside) else 0.0)}


def text_features(window):
    """Curve features of one text (its windows sorted by position)."""
    p = window["p_ai"].to_numpy()
    start, end = window["tok_start"].to_numpy(), window["tok_end"].to_numpy()
    n_tokens = window["n_tokens"].iloc[0]
    centre = (start + end) / 2 / n_tokens
    high = p > HIGH
    run_start, run_end = longest_run(high)
    run_share = (end[run_end - 1] - start[run_start]) / n_tokens if run_end > run_start else 0.0
    gain, split = best_step(p)
    head = p[centre <= HEAD_TAIL] if (centre <= HEAD_TAIL).any() else p[:1]
    tail = p[centre >= 1 - HEAD_TAIL] if (centre >= 1 - HEAD_TAIL).any() else p[-1:]
    return {
        "mean": p.mean(), "max": p.max(), "std": p.std(), "frac_high": high.mean(),
        "longest_high_share": run_share,
        "mean_abs_diff": np.abs(np.diff(p)).mean() if len(p) > 1 else 0.0,
        "n_switches": int((np.diff(high.astype(int)) != 0).sum()),
        "step_gain": gain, "step_pos": start[split] / n_tokens if split else 0.0,
        "head_mean": head.mean(), "tail_mean": tail.mean(), "tail_minus_head": tail.mean() - head.mean(),
        "last_p": p[window["is_last"].to_numpy()][0],
        **segment_features(p, start, end, n_tokens),
        "n_windows": len(p), "log_tokens": np.log(n_tokens),
    }


def curve_features(curves):
    """FEATURES for every text of a window table, indexed by id."""
    rows = {text_id: text_features(window) for text_id, window in curves.sort_values(["id", "position"]).groupby("id")}
    return pd.DataFrame.from_dict(rows, orient="index")[FEATURES]


def on_grid(curves):
    """Each text's curve interpolated at GRID positions (window centre / text length), one row per id."""
    rows = {}
    for text_id, window in curves.sort_values(["id", "position"]).groupby("id"):
        centre = ((window["tok_start"] + window["tok_end"]) / 2 / window["n_tokens"]).to_numpy()
        rows[text_id] = np.interp(GRID, centre, window["p_ai"].to_numpy())
    return pd.DataFrame.from_dict(rows, orient="index")


def cross_scale_features(curves_a, curves_b, name):
    """Agreement of two window sizes' curves of the same texts: correlation, max gap and high-mask IoU."""
    a, b = on_grid(curves_a), on_grid(curves_b)
    b = b.loc[a.index]
    centred_a, centred_b = a.sub(a.mean(axis=1), axis=0), b.sub(b.mean(axis=1), axis=0)
    norm = np.sqrt((centred_a ** 2).sum(axis=1) * (centred_b ** 2).sum(axis=1))
    high_a, high_b = a > HIGH, b > HIGH
    union = (high_a | high_b).sum(axis=1)
    return pd.DataFrame({
        f"{name}_corr": ((centred_a * centred_b).sum(axis=1) / norm.replace(0, np.nan)).fillna(0.0),
        f"{name}_max_gap": (a.max(axis=1) - b.max(axis=1)).abs(),
        f"{name}_high_iou": ((high_a & high_b).sum(axis=1) / union.replace(0, np.nan)).fillna(1.0),
    })
