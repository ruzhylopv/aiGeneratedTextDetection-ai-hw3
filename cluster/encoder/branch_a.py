"""Branch A (the tuned 3-class XLM-R) probabilities per text, for the stacker.

    torchrun --standalone --nproc_per_node=4 -m encoder.branch_a --sets stacker val test
    torchrun --standalone --nproc_per_node=4 -m encoder.branch_a --gate

Uses the same windows and aggregation as notebooks/hw3-encoder-tuning.ipynb: 512-token windows with 128
tokens of overlap, mean of the window probabilities. Val and test are scored again here, so all three
sets come from the same hardware and precision. The val run is checked against the Kaggle probabilities.
Each process writes branch_a_probs_<set>.part<rank>.csv (id, human, ai, fusion) to the run folder.
"""

import argparse

import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from encoder.data import LABELS, branch_a_dir, load_gate, load_sets, run_dir
from encoder.infer import device, log, predict_proba, rank_info, shard

MAX_LEN = 512
STRIDE = 128  # tokens shared by consecutive windows, as in the notebook
CHUNK_TEXTS = 2_000
BATCH_SIZE = 128


def text_probs(model, tokenizer, texts):
    """Mean window probabilities per text, for this process's shard of texts."""
    texts = shard(texts)
    columns = [model.config.id2label[i] for i in range(model.config.num_labels)]
    parts = []
    for start in range(0, len(texts), CHUNK_TEXTS):
        chunk = texts.iloc[start:start + CHUNK_TEXTS]
        enc = tokenizer(chunk["text"].tolist(), truncation=True, max_length=MAX_LEN, stride=STRIDE,
                        return_overflowing_tokens=True)
        probs = predict_proba(model, enc["input_ids"], tokenizer.pad_token_id, BATCH_SIZE)
        doc = np.asarray(enc["overflow_to_sample_mapping"])
        mean = pd.DataFrame(probs, columns=columns).groupby(doc).mean().reindex(range(len(chunk)))
        parts.append(mean[LABELS].assign(id=chunk["id"].to_numpy()))
        log(f"  {min(start + CHUNK_TEXTS, len(texts)):,} / {len(texts):,} texts (rank 0)")
    out = pd.concat(parts, ignore_index=True)[["id", *LABELS]]
    assert not out.isna().any().any(), "some texts have no window"
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", default=["stacker", "val", "test"])
    parser.add_argument("--gate", action="store_true", help="score the gate texts instead of --sets")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(branch_a_dir())
    model = AutoModelForSequenceClassification.from_pretrained(branch_a_dir()).to(device())
    texts_by_set = {"gate": load_gate()} if args.gate else {k: v for k, v in load_sets().items() if k in args.sets}

    rank = rank_info()[0]
    for name, texts in texts_by_set.items():
        log(f"{name}: {len(texts):,} texts")
        text_probs(model, tokenizer, texts).to_csv(run_dir() / f"branch_a_probs_{name}.part{rank}.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()
