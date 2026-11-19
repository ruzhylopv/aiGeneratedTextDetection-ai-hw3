"""Encoder v2: BGE-M3 (XLM-R large with 8k positions) fine-tuned for human / ai / fusion on whole texts.

    torchrun --standalone --nproc_per_node=4 -m encoder.long_encoder train
    torchrun --standalone --nproc_per_node=4 -m encoder.long_encoder predict --sets stacker val test

Unlike Branch A (512-token windows, mean of the window probabilities), each text is read in one pass of up
to MAX_LEN tokens. A longer text keeps its first and last tokens (head + tail): a fusion text's AI part comes
after its copied human start, so the end must not be cut off.

Training uses the encoder set (splits/encoder_train.csv: 100k texts, never a stacker text or its twin), so
the stacker later sees probabilities of texts the encoder has not seen. The model is saved to
$HW3_MODELS/encoder-v2; predictions go to the run folder as encoder_probs_<set>.part<rank>.csv.
"""

import argparse
import json
import math

import numpy as np
import pandas as pd
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments, set_seed)

from encoder.data import LABEL2ID, LABELS, SEED, load_sets, models_dir, run_dir
from encoder.detector import WindowDataset
from encoder.infer import device, log, predict_proba, rank_info, shard

MODEL = "BAAI/bge-m3"
NAME = "encoder-v2"
MAX_LEN = 4096          # tokens per text, including <s> and </s>
CONFIG = {"epochs": 2, "lr": 1e-5, "batch_size": 8, "grad_accum": 2, "warmup_frac": 0.06, "weight_decay": 0.01}
PREDICT_BATCH = 16
CHUNK_TEXTS = 2_000


def head_tail(ids, length):
    """The first and last tokens of ids, `length` in total (ids unchanged if short enough)."""
    if len(ids) <= length:
        return list(ids)
    head = length // 2
    return list(ids[:head]) + list(ids[len(ids) - (length - head):])


def encode_texts(tokenizer, texts, batch_size=1_000):
    """Token ids of whole texts, cut to MAX_LEN with head + tail, with <s> and </s>."""
    out = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(list(texts[start:start + batch_size]), add_special_tokens=False, verbose=False)
        out += [np.asarray([tokenizer.cls_token_id, *head_tail(ids, MAX_LEN - 2), tokenizer.sep_token_id])
                for ids in enc["input_ids"]]
    return out


def train():
    """Fine-tune BGE-M3 with a 3-class head on the encoder set and save it to $HW3_MODELS/encoder-v2."""
    set_seed(SEED)
    texts = load_sets()["encoder"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    train_ds = WindowDataset(encode_texts(tokenizer, texts["text"].tolist()),
                             texts["label"].map(LABEL2ID).to_numpy())
    lengths = np.array([len(ids) for ids in train_ds.input_ids])
    log(f"{NAME}: {len(texts):,} texts, tokens per text: median {np.median(lengths):.0f}, "
        f"p99 {np.percentile(lengths, 99):.0f}, cut to {MAX_LEN}: {(lengths >= MAX_LEN).mean():.1%}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID)
    world = rank_info()[1]
    total_steps = CONFIG["epochs"] * math.ceil(len(train_ds) / (CONFIG["batch_size"] * CONFIG["grad_accum"] * world))
    args = TrainingArguments(
        output_dir=str(run_dir() / "trainer"),
        num_train_epochs=CONFIG["epochs"],
        learning_rate=CONFIG["lr"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["grad_accum"],
        warmup_steps=int(CONFIG["warmup_frac"] * total_steps),
        weight_decay=CONFIG["weight_decay"],
        lr_scheduler_type="linear",
        bf16=True,
        gradient_checkpointing=True,              # long texts: trade compute for memory
        gradient_checkpointing_kwargs={"use_reentrant": False},  # the variant that works with DDP
        train_sampling_strategy="group_by_length",  # batches of similar length, little padding
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=4,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))
    output = trainer.train()
    trainer.save_model(str(models_dir() / NAME))
    if trainer.is_world_process_zero():
        summary = {"name": NAME, "model": MODEL, "max_len": MAX_LEN, **CONFIG, "texts": len(texts),
                   "train_minutes": round(output.metrics["train_runtime"] / 60, 1),
                   "model_dir": str(models_dir() / NAME)}
        (run_dir() / f"{NAME}.json").write_text(json.dumps(summary, indent=2))
        pd.DataFrame([row for row in trainer.state.log_history if "loss" in row]).to_csv(
            run_dir() / f"{NAME}_loss.csv", index=False)
        print(json.dumps(summary, indent=2), flush=True)


def predict(set_names):
    """Class probabilities of every text of the given sets, for this process's shard."""
    model_path = models_dir() / NAME
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device())
    columns = [model.config.id2label[i] for i in range(model.config.num_labels)]
    rank = rank_info()[0]
    for name, texts in {k: v for k, v in load_sets().items() if k in set_names}.items():
        texts = shard(texts)
        log(f"{name}: {len(texts):,} texts on this GPU")
        parts = []
        for start in range(0, len(texts), CHUNK_TEXTS):
            chunk = texts.iloc[start:start + CHUNK_TEXTS]
            probs = predict_proba(model, encode_texts(tokenizer, chunk["text"].tolist()), tokenizer.pad_token_id,
                                  PREDICT_BATCH)
            parts.append(pd.DataFrame(probs, columns=columns)[LABELS].assign(id=chunk["id"].to_numpy()))
            log(f"  {min(start + CHUNK_TEXTS, len(texts)):,} / {len(texts):,} texts (rank 0)")
        pd.concat(parts, ignore_index=True)[["id", *LABELS]].to_csv(
            run_dir() / f"encoder_probs_{name}.part{rank}.csv", index=False)
    log("done")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["train", "predict"])
    parser.add_argument("--sets", nargs="+", default=["stacker", "val", "test"])
    args = parser.parse_args()
    if args.step == "train":
        train()
    else:
        predict(args.sets)


if __name__ == "__main__":
    main()
