"""Binary human-vs-ai window detector (Branch B). Runs on all 4 GPUs of a node:

    torchrun --standalone --nproc_per_node=4 -m encoder.detector --mode gate
    torchrun --standalone --nproc_per_node=4 -m encoder.detector --mode full

Trains on the detector set only: the human and ai texts of the fixed 50k sample (fusion is excluded).
It starts from the tuned Branch A encoder with a new 2-class head (`--init xlmr`: plain
xlm-roberta-base instead, for experiment E6). The model is saved to $HW3_MODELS/<name> for the
scoring jobs, and the training summary and loss log go to the run folder.
"""

import argparse
import json
import math

import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments, set_seed)

from encoder.data import SEED, branch_a_dir, load_sets, models_dir, run_dir, stratified_sample
from encoder.infer import log, rank_info
from encoder.windows import encode, make_windows

DETECTOR_LABELS = ["human", "ai"]
XLMR = "FacebookAI/xlm-roberta-base"
STEP_DIV = 4  # windows start every size / 4 tokens
MODES = {
    "gate": {"n_texts": 10_000, "sizes": [128], "max_windows": 8, "epochs": 1},
    "full": {"n_texts": None, "sizes": [128, 256], "max_windows": 6, "epochs": 2},
}
TRAIN = {"lr": 2e-5, "batch_size": 32, "warmup_frac": 0.06, "weight_decay": 0.01}


class WindowDataset(torch.utils.data.Dataset):
    """Token windows with their 0/1 labels, as the dicts the Trainer's collator expects."""

    def __init__(self, input_ids, labels):
        self.input_ids, self.labels = input_ids, labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i].tolist(), "labels": int(self.labels[i])}


def training_windows(tokenizer, texts, config):
    """Windows of every configured size over the detector texts, labelled 0 = human, 1 = ai."""
    encoded = encode(tokenizer, texts["text"].tolist())
    is_ai = (texts["label"] == "ai").astype(int).to_numpy()
    input_ids, labels, counts = [], [], {}
    for size in config["sizes"]:
        ids, index = make_windows(tokenizer, encoded, size, size // STEP_DIV, config["max_windows"], seed=SEED + size)
        input_ids += ids
        labels += is_ai[index["doc"]].tolist()
        counts[size] = len(ids)
    return WindowDataset(input_ids, labels), counts


def build_trainer(model, tokenizer, train_ds, config):
    """Trainer with the same optimiser settings as Branch A, in bf16, without checkpoints."""
    world = rank_info()[1]
    total_steps = config["epochs"] * math.ceil(len(train_ds) / (config["batch_size"] * world))
    args = TrainingArguments(
        output_dir=str(run_dir() / "trainer"),
        num_train_epochs=config["epochs"],
        learning_rate=config["lr"],
        per_device_train_batch_size=config["batch_size"],
        warmup_steps=int(config["warmup_frac"] * total_steps),
        weight_decay=config["weight_decay"],
        lr_scheduler_type="linear",
        bf16=True,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=4,
    )
    return Trainer(model=model, args=args, train_dataset=train_ds, processing_class=tokenizer,
                   data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=list(MODES), required=True)
    parser.add_argument("--init", choices=["branch_a", "xlmr"], default="branch_a")
    args = parser.parse_args()
    name = f"detector-{args.mode}" + ("-xlmr" if args.init == "xlmr" else "")
    config = {"name": name, "mode": args.mode, "init": args.init, **MODES[args.mode], **TRAIN}
    set_seed(SEED)

    texts = load_sets()["detector"]
    if config["n_texts"]:
        texts = stratified_sample(texts, config["n_texts"])
    init_path = str(branch_a_dir()) if args.init == "branch_a" else XLMR
    tokenizer = AutoTokenizer.from_pretrained(init_path)
    train_ds, window_counts = training_windows(tokenizer, texts, config)
    log(f"{name}: {len(texts):,} texts ({texts['label'].value_counts().to_dict()}), windows per size {window_counts}")

    # The Branch A head has 3 outputs; ignore_mismatched_sizes gives a fresh 2-class output layer.
    model = AutoModelForSequenceClassification.from_pretrained(
        init_path, num_labels=len(DETECTOR_LABELS), id2label=dict(enumerate(DETECTOR_LABELS)),
        label2id={label: i for i, label in enumerate(DETECTOR_LABELS)}, ignore_mismatched_sizes=True)
    trainer = build_trainer(model, tokenizer, train_ds, config)
    train_output = trainer.train()
    trainer.save_model(str(models_dir() / name))  # written by the main process only

    if trainer.is_world_process_zero():
        summary = {**config, "texts": len(texts), "windows": window_counts,
                   "train_minutes": round(train_output.metrics["train_runtime"] / 60, 1),
                   "model_dir": str(models_dir() / name)}
        (run_dir() / f"{name}.json").write_text(json.dumps(summary, indent=2))
        pd.DataFrame([row for row in trainer.state.log_history if "loss" in row]).to_csv(
            run_dir() / f"{name}_loss.csv", index=False)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
