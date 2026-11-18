"""Check that a GPU job can do everything the encoder runs need. Submit with `make -C cluster smoke`.

Checks, in order: GPUs and CPUs visible, data and split load, offline model + tokenizer load,
sliding-window tokenization, a few bf16 training steps and an inference pass (with throughput, to
estimate the cost of the real runs), and the run folder is writable. Each step prints its result;
a failed required check stops the job with FAIL.
"""

import json
import os
import socket
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from encoder.data import LABEL2ID, LABELS, SEED, load_data, run_dir

MODEL = "FacebookAI/xlm-roberta-base"
EXTRA_MODELS = ["microsoft/mdeberta-v3-base"]  # for step 3; a failure here is only a warning
MAX_LEN = 512
STRIDE = 128  # overlap between windows, so windows start every 384 tokens
BATCH_SIZE = 16
TRAIN_STEPS = 20
N_TEXTS = BATCH_SIZE * TRAIN_STEPS


def step(name):
    """Print a section header."""
    print(f"\n== {name}", flush=True)


def check_hardware():
    """GPUs and CPU cores the job can use."""
    if not torch.cuda.is_available():
        sys.exit("FAIL: no GPU visible. Is this running inside a Slurm job with --gres=gpu?")
    gpus = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    return {
        "host": socket.gethostname(),
        "torch": torch.__version__,
        "gpus": [f"{p.name}, {p.total_memory / 1024**3:.0f} GB" for p in gpus],
        "cpus": len(os.sched_getaffinity(0)),
        "bf16": torch.cuda.is_bf16_supported(),
    }


def check_data():
    """Row counts of the split; must match the notebooks (fold 0 = 27,236 texts)."""
    data = load_data()
    sizes = {name: len(df) for name, df in data.items()}
    if sizes["val"] != 27_236 or sizes["test"] != 25_000:
        sys.exit(f"FAIL: unexpected split sizes {sizes}")
    return data, sizes


def load_encoder(name):
    """Tokenizer and 3-class classification model from the offline HF cache."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID)
    return tokenizer, model


def windows_per_text(tokenizer, texts):
    """Number of 512-token sliding windows each text is split into."""
    enc = tokenizer(list(texts), max_length=MAX_LEN, stride=STRIDE, truncation=True,
                    return_overflowing_tokens=True)
    return torch.bincount(torch.tensor(enc["overflow_to_sample_mapping"])).tolist()


def batches(tokenizer, texts, labels):
    """Head-512 batches of tokenized texts with their label ids, on the GPU."""
    for i in range(0, len(texts), BATCH_SIZE):
        enc = tokenizer(list(texts[i:i + BATCH_SIZE]), max_length=MAX_LEN, truncation=True,
                        padding="max_length", return_tensors="pt").to("cuda")
        enc["labels"] = torch.tensor([LABEL2ID[y] for y in labels[i:i + BATCH_SIZE]], device="cuda")
        yield enc


def time_training(model, tokenizer, sample):
    """Run TRAIN_STEPS bf16 AdamW steps; return losses and sequences per second."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    losses, start = [], time.time()
    for batch in batches(tokenizer, sample["text"].tolist(), sample["label"].tolist()):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    torch.cuda.synchronize()
    if not torch.isfinite(torch.tensor(losses)).all():
        sys.exit(f"FAIL: non-finite training loss {losses}")
    return losses, len(sample) / (time.time() - start)


@torch.no_grad()
def time_inference(model, tokenizer, sample):
    """bf16 forward passes over the sample; return sequences per second."""
    model.eval()
    start = time.time()
    for batch in batches(tokenizer, sample["text"].tolist(), sample["label"].tolist()):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(**batch).logits.float().cpu()
    torch.cuda.synchronize()
    return len(sample) / (time.time() - start)


def main():
    report = {}

    step("Hardware")
    report["hardware"] = check_hardware()
    print(json.dumps(report["hardware"], indent=2))

    step("Data and split")
    data, report["sizes"] = check_data()
    print(report["sizes"])
    sample = data["train"].sample(N_TEXTS, random_state=SEED)  # training folds only

    step(f"Load {MODEL} (offline)")
    tokenizer, model = load_encoder(MODEL)
    model.to("cuda")
    print(f"{type(model).__name__}, {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

    step("Sliding windows (512 tokens, stride 384)")
    counts = windows_per_text(tokenizer, sample["text"])
    report["mean_windows_per_text"] = sum(counts) / len(counts)
    print(f"mean windows per text: {report['mean_windows_per_text']:.2f} (EDA: ~2.1)")

    step(f"Training: {TRAIN_STEPS} steps, batch {BATCH_SIZE}, bf16, 1 GPU")
    losses, report["train_seq_per_s"] = time_training(model, tokenizer, sample)
    print(f"loss first/last: {losses[0]:.3f} / {losses[-1]:.3f} | {report['train_seq_per_s']:.0f} seq/s")

    step("Inference")
    report["infer_seq_per_s"] = time_inference(model, tokenizer, sample)
    print(f"{report['infer_seq_per_s']:.0f} seq/s | peak GPU memory "
          f"{torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
    del model
    torch.cuda.empty_cache()

    for name in EXTRA_MODELS:
        step(f"Load {name} (offline, optional)")
        try:
            load_encoder(name)
            report[f"load_{name}"] = "ok"
        except Exception as err:  # optional model: report, don't stop the smoke test
            report[f"load_{name}"] = f"WARNING: {type(err).__name__}: {err}"
        print(report[f"load_{name}"])

    step("Write to the run folder")
    out = run_dir() / "smoke_test.json"
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out)
    print("\nAll required checks passed.")


if __name__ == "__main__":
    main()
