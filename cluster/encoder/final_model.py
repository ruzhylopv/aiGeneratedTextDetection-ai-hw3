"""The final model (P6 in notebooks/final_stacker.ipynb) and an exact split of its decisions into its inputs.

P6 is a logistic regression on four inputs: Branch A log-probabilities, encoder v2 log-probabilities, the
detector's curve features and the Qwen features. With standardized features, each class score is

    z_k = intercept_k + sum over features j of coef_kj * x_j

so the score splits exactly into one contribution per input (the sum of its features' terms). For a linear
model this is the same as SHAP values with the mean as the baseline (the standardized features have mean 0
on the stacker set). The class weights (human, fusion) multiply the probabilities, i.e. they add log(w_k) to
z_k. Only numpy, pandas and scikit-learn are needed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from encoder.data import LABELS
from encoder.stacking import log_probs, read_parts

CURVES_JOB, ENCODER_JOB, QWEN_JOB = "3016350", "3017270", "3017273"
# Chosen in final_stacker.ipynb by CV on the stacker set (row "P6: all four"); validation macro-F1 0.9624.
FINAL_PARAMS = {"C": 0.1, "class_weight": None, "max_iter": 5_000}
CLASS_WEIGHTS = {"human": 1.4, "ai": 1.0, "fusion": 1.5}
INPUTS = ["encoder v2", "Branch A", "curves", "Qwen"]


def input_table(set_name, sets, repo_dir):
    """All four inputs of one set as one feature table, in the set's row order (as in final_stacker.ipynb)."""
    repo_dir = Path(repo_dir)
    results = repo_dir / "cluster" / "results"

    def probs(job, pattern):
        return read_parts(results / job, pattern).set_index("id").loc[sets[set_name]["id"], LABELS]

    curves = pd.read_csv(repo_dir / "outputs" / "curves" / f"features_{CURVES_JOB}_{set_name}.csv", index_col="id")
    curves = curves[[c for c in curves.columns if "_fine__" in c or c.startswith("cross_")]]
    qwen = pd.read_csv(repo_dir / "outputs" / "final" / f"qwen_features_{QWEN_JOB}_{set_name}.csv", index_col="id")
    return (log_probs(probs(CURVES_JOB, f"branch_a_probs_{set_name}.part*.csv"), "logp_")
            .join(log_probs(probs(ENCODER_JOB, f"encoder_probs_{set_name}.part*.csv"), "enc2_logp_"))
            .join(curves).join(qwen).loc[sets[set_name]["id"]])


def input_of(column):
    """Name of the input a feature column belongs to."""
    if column.startswith("enc2_logp_"):
        return "encoder v2"
    if column.startswith("logp_"):
        return "Branch A"
    if column.startswith("qwen"):
        return "Qwen"
    return "curves"


def input_groups(columns):
    """Feature columns of each input."""
    groups = {name: [] for name in INPUTS}
    for column in columns:
        groups[input_of(column)].append(column)
    return groups


def fit_final(X, y):
    """The final model (scaler + logistic regression), fitted on the stacker set."""
    return make_pipeline(StandardScaler(), LogisticRegression(**FINAL_PARAMS)).fit(X, y)


def predict(model, X):
    """Final labels: the model's probabilities times the class weights, argmax."""
    weights = np.array([CLASS_WEIGHTS[c] for c in model.classes_])
    return model.classes_[(model.predict_proba(X) * weights).argmax(axis=1)]


def contributions(model, X):
    """Per text and class: the score contribution of each input, the intercept and the class weight.

    Returns {class: DataFrame(index = X.index, columns = inputs + ["intercept", "class weight"])}. The row sums
    are the class scores used for the final decision (log-odds up to a shared constant).
    """
    scaler, logreg = model[0], model[-1]
    standardized = scaler.transform(X)
    groups = input_groups(X.columns)
    out = {}
    for k, label in enumerate(logreg.classes_):
        terms = standardized * logreg.coef_[k]
        frame = pd.DataFrame({name: terms[:, [X.columns.get_loc(c) for c in cols]].sum(axis=1)
                              for name, cols in groups.items()}, index=X.index)
        frame["intercept"] = logreg.intercept_[k]
        frame["class weight"] = np.log(CLASS_WEIGHTS[label])
        out[label] = frame
    return out


def margin(contribs, winner, other):
    """Per text: contributions to score(winner) - score(other), one column per input (+ intercept, class weight)."""
    return contribs[winner] - contribs[other]
