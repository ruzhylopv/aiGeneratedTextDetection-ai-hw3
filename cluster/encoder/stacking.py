"""Stacker helpers shared by the stacker notebooks: reading cluster outputs, CV, class weights, scoring.

Same logic as notebooks/curve_stacker.ipynb section 3, but generic over the feature columns: a configuration
is (feature columns, model name, parameters); model choice uses CV on the stacker set; class weights are
tuned on out-of-fold probabilities; validation is predicted once. Only numpy, pandas, scikit-learn.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from encoder.data import LABELS, SEED

N_PARTS = 4
CV_FOLDS = 5
MODELS = {"LogReg": LogisticRegression, "HGB": HistGradientBoostingClassifier}
WEIGHT_GRID = np.round(np.arange(0.5, 2.01, 0.1), 2)  # multipliers for human and fusion (ai fixed at 1)


def read_parts(results_dir, pattern):
    """All per-GPU part files of one cluster output, concatenated."""
    parts = sorted(Path(results_dir).glob(pattern))
    assert len(parts) == N_PARTS, f"{pattern}: {len(parts)} part files in {results_dir}"
    return pd.concat([pd.read_csv(path) for path in parts], ignore_index=True)


def log_probs(probs, prefix):
    """Log of class probabilities (clipped at 1e-6), columns <prefix><class>."""
    return np.log(probs[LABELS].clip(lower=1e-6)).add_prefix(prefix)


def to_labels(probs):
    """Class name of the most probable class per row (columns in LABELS order)."""
    return np.array(LABELS)[np.asarray(probs).argmax(axis=1)]


def score(y_true, y_pred):
    """Macro-F1 and per-class F1."""
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None)
    return {"macro_f1": f1_score(y_true, y_pred, average="macro"),
            **{f"f1_{label}": f1 for label, f1 in zip(LABELS, per_class)}}


def make_model(model_name, params):
    """A fresh model; logistic regression gets a scaler fitted inside the same pipeline."""
    model = MODELS[model_name](**params)
    return make_pipeline(StandardScaler(), model) if model_name == "LogReg" else model


def in_label_order(model, probs):
    """Probability columns reordered from model.classes_ to LABELS."""
    classes = list(model.classes_)
    return probs[:, [classes.index(label) for label in LABELS]]


def oof_probs(X, y, model_name, params):
    """Out-of-fold class probabilities (LABELS order) of one configuration on the stacker set."""
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    probs = cross_val_predict(make_model(model_name, params), X, y, cv=cv, method="predict_proba", n_jobs=CV_FOLDS)
    return pd.DataFrame(probs, columns=sorted(LABELS))[LABELS].to_numpy()  # sklearn orders classes alphabetically


def apply_weights(probs, weights):
    """Labels after multiplying the human and fusion probabilities by weights = (w_human, w_fusion)."""
    w_human, w_fusion = weights
    return to_labels(np.asarray(probs) * np.array([w_human, 1.0, w_fusion]))


def tune_weights(probs, y_true):
    """Class weights from WEIGHT_GRID that maximise macro-F1 of the weighted argmax."""
    candidates = [(float(w_human), float(w_fusion)) for w_human in WEIGHT_GRID for w_fusion in WEIGHT_GRID]
    return max(candidates, key=lambda w: f1_score(y_true, apply_weights(probs, w), average="macro"))


def fast_macro_f1(y_true, y_pred):
    """Macro-F1 from a 3 x 3 confusion matrix of class indices (for the bootstrap)."""
    confusion = np.bincount(3 * y_true + y_pred, minlength=9).reshape(3, 3)
    tp = np.diag(confusion)
    return np.mean(2 * tp / np.maximum(confusion.sum(axis=0) + confusion.sum(axis=1), 1))


def bootstrap_diff(y_true, pred, pred_ref, n=1_000):
    """95% paired-bootstrap interval of macro-F1(pred) - macro-F1(pred_ref) over resampled texts."""
    index = {label: i for i, label in enumerate(LABELS)}
    t, p, r = (np.array([index[v] for v in a]) for a in (y_true, pred, pred_ref))
    rng = np.random.default_rng(SEED)
    diffs = [fast_macro_f1(t[s], p[s]) - fast_macro_f1(t[s], r[s])
             for s in (rng.integers(0, len(t), len(t)) for _ in range(n))]
    return np.percentile(diffs, [2.5, 97.5])


def make_submission(ids, labels, sample):
    """Predictions in sample-submission order, checked against the format rules in COMPETITION.md."""
    sub = sample[["id"]].merge(pd.DataFrame({"id": ids, "label": labels}), on="id", how="left", validate="1:1")
    assert len(sub) == 25_000 and sub["id"].equals(sample["id"]), "rows or order differ from sample_submission.csv"
    assert sub["label"].isin(LABELS).all(), "missing or unknown labels"
    return sub
