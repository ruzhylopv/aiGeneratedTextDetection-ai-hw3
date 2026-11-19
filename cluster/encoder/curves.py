"""Score p(AI) curves: every window of every text, at each window size, with a trained detector.

    torchrun --standalone --nproc_per_node=4 -m encoder.curves --model detector-full --sets stacker val test
    torchrun --standalone --nproc_per_node=4 -m encoder.curves --model detector-gate --gate --sizes 128

Windows start every size / 4 tokens. Coarser strides are subsamples of these windows (see windows.py),
so one pass serves every stride. Each process writes curves_<set>_<size>.part<rank>.csv.gz to the run
folder, with one row per window: id, position, is_last, token and character offsets, n_tokens, p_ai.
"""

import argparse

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from encoder.data import load_gate, load_sets, models_dir, run_dir
from encoder.infer import device, log, predict_proba, rank_info, shard
from encoder.windows import encode, make_windows

SIZES = [64, 128, 256]
STEP_DIV = 4
CHUNK_TEXTS = 2_000
BATCH_SIZE = {64: 1024, 128: 512, 256: 256}


def score_texts(model, tokenizer, texts, sizes):
    """Window table with p_ai for every size, for this process's shard of texts."""
    ai = model.config.label2id["ai"]
    texts = shard(texts)
    frames = {size: [] for size in sizes}
    for start in range(0, len(texts), CHUNK_TEXTS):
        chunk = texts.iloc[start:start + CHUNK_TEXTS]
        encoded = encode(tokenizer, chunk["text"].tolist())
        for size in sizes:
            ids, index = make_windows(tokenizer, encoded, size, size // STEP_DIV)
            index["p_ai"] = predict_proba(model, ids, tokenizer.pad_token_id, BATCH_SIZE[size])[:, ai]
            index.insert(0, "id", chunk["id"].to_numpy()[index["doc"]])
            frames[size].append(index.drop(columns="doc"))
        log(f"  {min(start + CHUNK_TEXTS, len(texts)):,} / {len(texts):,} texts (rank 0)")
    return {size: pd.concat(parts, ignore_index=True) for size, parts in frames.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="folder name under $HW3_MODELS, e.g. detector-full")
    parser.add_argument("--sets", nargs="+", default=["stacker", "val", "test"])
    parser.add_argument("--gate", action="store_true", help="score the gate texts instead of --sets")
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    args = parser.parse_args()

    model_path = models_dir() / args.model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device())
    texts_by_set = {"gate": load_gate()} if args.gate else {k: v for k, v in load_sets().items() if k in args.sets}

    rank = rank_info()[0]
    for name, texts in texts_by_set.items():
        log(f"{name}: {len(texts):,} texts, sizes {args.sizes}")
        for size, curves in score_texts(model, tokenizer, texts, args.sizes).items():
            curves.to_csv(run_dir() / f"curves_{name}_{size}.part{rank}.csv.gz", index=False)
    log("done")


if __name__ == "__main__":
    main()
