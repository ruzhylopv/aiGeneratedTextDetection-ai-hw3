"""Per-token log-probabilities and entropies of an open LLM (Qwen3-8B), summed over 32-token chunks.

    torchrun --standalone --nproc_per_node=4 -m encoder.qwen_logprobs --model <path> --sets detector stacker val test

AI-written text tends to be unusually predictable to a language model: the actual tokens get a high
log-probability and the predicted distribution has a low entropy. Each text is scored in one causal pass,
without truncation. For every token after the first: log p(token | previous tokens) and the entropy of the
model's next-token distribution. Nothing is trained here.

Values are summed over consecutive chunks of CHUNK tokens, so the laptop can rebuild windows of any multiple
of CHUNK (64, 128, 256 tokens) at a CHUNK-token step without scoring again. Each process writes
qwen_chunks_<set>.part<rank>.csv.gz: id, chunk, tok_start, tok_end, char_start, char_end, n_tokens (of the
text), n_scored (tokens with a value in the chunk), sum_logp, sum_entropy.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from encoder.data import load_sets, run_dir
from encoder.infer import device, log, rank_info, shard

CHUNK = 32              # tokens per stored chunk
TOKEN_BUDGET = 16_384   # padded tokens per forward pass
HEAD_ROWS = 4_096       # positions per slice through the LM head (vocabulary ~151k: bounds memory)
CHUNK_TEXTS = 2_000


def chunk_rows(text_id, logp, entropy, offsets):
    """Rows of the chunk table for one text. logp / entropy have one value per token (NaN for the first)."""
    n_tokens = len(logp)
    rows = []
    for start in range(0, n_tokens, CHUNK):
        end = min(start + CHUNK, n_tokens)
        scored = ~np.isnan(logp[start:end])
        rows.append((text_id, start // CHUNK, start, end, int(offsets[start][0]), int(offsets[end - 1][1]), n_tokens,
                     int(scored.sum()), float(np.nansum(logp[start:end])), float(np.nansum(entropy[start:end]))))
    return rows


CHUNK_COLUMNS = ["id", "chunk", "tok_start", "tok_end", "char_start", "char_end", "n_tokens", "n_scored",
                 "sum_logp", "sum_entropy"]


def length_batches(lengths, budget):
    """Index lists of texts sorted by length, each batch with (count × longest) <= budget tokens."""
    order = np.argsort(lengths, kind="stable")
    batches, current = [], []
    for i in order:
        if current and (len(current) + 1) * lengths[i] > budget:
            batches.append(current)
            current = []
        current.append(i)
    return batches + ([current] if current else [])


@torch.no_grad()
def token_scores(model, input_ids, pad_id):
    """Log-probability of each actual token and entropy of each next-token distribution, per text.

    Returns lists of numpy arrays aligned with the input tokens (position 0 has no value: NaN).
    """
    width = max(len(ids) for ids in input_ids)
    batch = torch.full((len(input_ids), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(batch)
    for row, ids in enumerate(input_ids):
        batch[row, :len(ids)] = torch.as_tensor(ids)
        mask[row, :len(ids)] = 1
    batch, mask = batch.to(device()), mask.to(device())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(input_ids=batch, attention_mask=mask).last_hidden_state  # (B, L, d)
    logp_all = torch.full(batch.shape, float("nan"), device=device())
    ent_all = torch.full(batch.shape, float("nan"), device=device())
    flat_hidden = hidden[:, :-1].reshape(-1, hidden.shape[-1])  # position t predicts token t + 1
    flat_target = batch[:, 1:].reshape(-1)
    flat_logp = torch.empty(flat_target.shape, device=device())
    flat_ent = torch.empty(flat_target.shape, device=device())
    for start in range(0, len(flat_target), HEAD_ROWS):
        rows = slice(start, start + HEAD_ROWS)
        log_probs = torch.log_softmax(model.lm_head(flat_hidden[rows]).float(), dim=-1)
        flat_logp[rows] = log_probs.gather(1, flat_target[rows, None]).squeeze(1)
        flat_ent[rows] = -(log_probs.exp() * log_probs).sum(dim=-1)
    logp_all[:, 1:] = flat_logp.view(len(input_ids), -1)
    ent_all[:, 1:] = flat_ent.view(len(input_ids), -1)
    logp_all, ent_all = logp_all.cpu().numpy(), ent_all.cpu().numpy()
    return ([logp_all[row, :len(ids)] for row, ids in enumerate(input_ids)],
            [ent_all[row, :len(ids)] for row, ids in enumerate(input_ids)])


def score_set(model, tokenizer, texts):
    """Chunk table of this process's shard of texts."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    texts = shard(texts)
    rows = []
    started = time.time()
    for start in range(0, len(texts), CHUNK_TEXTS):
        chunk = texts.iloc[start:start + CHUNK_TEXTS]
        enc = tokenizer(chunk["text"].tolist(), add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        lengths = np.array([len(ids) for ids in enc["input_ids"]])
        for batch in length_batches(lengths, TOKEN_BUDGET):
            logps, ents = token_scores(model, [enc["input_ids"][i] for i in batch], pad_id)
            for i, logp, ent in zip(batch, logps, ents):
                rows += chunk_rows(chunk["id"].iloc[i], logp, ent, enc["offset_mapping"][i])
        done = min(start + CHUNK_TEXTS, len(texts))
        log(f"  {done:,} / {len(texts):,} texts (rank 0), {(time.time() - started) / 60:.1f} min")
    return pd.DataFrame(rows, columns=CHUNK_COLUMNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="local folder of the LLM (e.g. Qwen3-8B snapshot)")
    parser.add_argument("--sets", nargs="+", default=["detector", "stacker", "val", "test"])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to(device()).eval()
    log(f"{args.model}: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")
    rank = rank_info()[0]
    if rank == 0:
        (run_dir() / "qwen.json").write_text(json.dumps({"model": args.model, "chunk": CHUNK, "sets": args.sets}, indent=2))
    for name, texts in {k: v for k, v in load_sets().items() if k in args.sets}.items():
        log(f"{name}: {len(texts):,} texts")
        score_set(model, tokenizer, texts).to_csv(run_dir() / f"qwen_chunks_{name}.part{rank}.csv.gz", index=False)
    log("done")


if __name__ == "__main__":
    main()
