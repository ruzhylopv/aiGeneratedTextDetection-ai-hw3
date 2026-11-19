"""Batched bf16 inference shared by the scoring scripts.

Scoring jobs run one process per GPU (`torchrun --standalone --nproc_per_node=4 -m encoder.<module>`).
Each process scores its own shard of the texts and writes its own part file. Readers glob the parts,
so no process has to wait for the others.
"""

import os

import numpy as np
import torch


def rank_info():
    """This process's rank and the number of processes (1 process when started without torchrun)."""
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def device():
    """The GPU of this process."""
    return torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))


def shard(df):
    """This process's share of the rows of df (every WORLD_SIZE-th row)."""
    rank, world = rank_info()
    return df.iloc[rank::world].reset_index(drop=True)


def log(*args):
    """Print once per job (rank 0 only)."""
    if rank_info()[0] == 0:
        print(*args, flush=True)


@torch.no_grad()
def predict_proba(model, input_ids, pad_id, batch_size):
    """Softmax class probabilities of each window (windows are batched by length, results in input order)."""
    model.eval()
    order = np.argsort([len(x) for x in input_ids], kind="stable")
    probs = np.empty((len(input_ids), model.config.num_labels), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        width = max(len(input_ids[i]) for i in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        for row, i in enumerate(batch):
            ids[row, :len(input_ids[i])] = torch.from_numpy(np.asarray(input_ids[i], dtype=np.int64))
        ids = ids.to(device(), non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=(ids != pad_id).long()).logits
        probs[batch] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return probs
