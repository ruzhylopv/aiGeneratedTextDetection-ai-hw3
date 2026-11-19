# Three inputs on JUPITER: step by step

Two new GPU jobs feed the final stacker. They don't depend on each other, so submit both at once.

| Job | What it does | Command | Time limit (expected) |
|---|---|---|---|
| **encoder v2** | fine-tunes BGE-M3 (XLM-R large, 8k tokens) on 100k whole texts, then predicts the stacker, val and test sets | `make -C cluster encoder2` | 3 h (~1–1.5 h) |
| **qwen** | Qwen3-8B per-token log-probabilities of the detector, stacker, val and test sets (no training) | `make -C cluster qwen` | 3 h (~0.5–1 h) |

The curve features from job 3016350 stay as they are. Nothing to rerun there.

---

## 1. Before you start (laptop)

The SSH certificate lasts about 10 hours. If `ssh jupiter hostname` fails, renew it first. See step 0 in `CURVES_JUPITER.md` (https://sshca.my-eurohpc.eu/ → token → `ssh -p 2226 ...`).

## 2. Get the code (login node)

```bash
cd /e/scratch/e-dev-2026d09-075/$USER/ai-hw3/repo
git fetch origin
git checkout three-inputs
git pull
ls splits/                  # must list encoder_train.csv
```

## 3. Download BGE-M3 (login node, once, ~5 min)

```bash
make -C cluster data
```
This re-runs the download script. Files already there are skipped, and the new model is `BAAI/bge-m3` (~2.3 GB). The last lines must include:
```
model: BAAI/bge-m3 -> /e/scratch/.../cache/huggingface/hub/models--BAAI--bge-m3/snapshots/...
```

## 4. Check that Qwen3-8B is found (login node)

```bash
make -C cluster qwen-path
```
**Expected:** one folder path ending in a commit hash, e.g. `/e/project1/e-dev-2026d09-075/fairforget/models/Qwen3-8B/<hash>`.

If it prints `Qwen3-8B not found`, run:
```bash
ls /e/project1/e-dev-2026d09-075/fairforget/models/
```
Then send me the output. If you know the folder, you can also pass it yourself in step 5: `make -C cluster qwen QWEN=/path/to/folder-with-config.json`.

## 5. Submit both jobs (login node)

```bash
make -C cluster encoder2
make -C cluster qwen
```
Each prints its job id and a `watch` command. Follow a job with:
```bash
make -C cluster watch JOB=<job_id>      # Ctrl-C stops watching only
```

**What the logs show:**

- **encoder2**
  1. `encoder-v2: 100,001 texts, tokens per text: median ..., p99 ..., cut to 4096: ...%`
  2. a `LOAD REPORT` (the new 3-class head is `MISSING`, which is expected)
  3. the training progress bar, with a `{'loss': ...}` line every 50 steps
  4. a JSON summary with `train_minutes`
  5. the predictions for `stacker`, `val` and `test`, each with progress lines
  6. `done`
- **qwen**
  1. `model: <path>`
  2. `<path>: 8.2B params`
  3. for each of `detector`, `stacker`, `val` and `test`: progress lines with minutes, e.g. `2,000 / 8,660 texts (rank 0), 3.1 min`
  4. `done`

The `errno: 97` and `OMP_NUM_THREADS` lines are harmless, as before.

## 6. Check the outputs and pull (laptop)

When both jobs show `COMPLETED`:
```bash
make -C cluster pull
ls cluster/results/<encoder2_job_id>/   # encoder-v2.json, encoder-v2_loss.csv, encoder_probs_{stacker,val,test}.part0..3.csv
ls cluster/results/<qwen_job_id>/       # qwen.json, qwen_chunks_{detector,stacker,val,test}.part0..3.csv.gz
```
Send me both job ids. I'll then build the final stacker on the three inputs (encoder v2, curves, Qwen).

## If something fails

Send me the last ~40 lines of the log (`tail -n 40 <log>`). Common cases:

| Message in the log | Meaning / fix |
|---|---|
| `FileNotFoundError: ... encoder_train.csv` / `KeyError: 'encoder'` | Old code on the cluster. Run `git checkout three-inputs && git pull`. |
| `LocalEntryNotFoundError` / `HF_HUB_OFFLINE` for `BAAI/bge-m3` | Step 3 wasn't run or didn't finish. Rerun `make -C cluster data`. |
| `Qwen3-8B not found` | See step 4. |
| `CUDA out of memory` (encoder2) | Tell me. I'll lower the batch size or the 4,096-token limit. |
| `CUDA out of memory` (qwen) | Tell me. I'll lower the token budget per batch. |
| `DUE TO TIME LIMIT` | Tell me. Then `make -C cluster encoder2-predict` if the model was already saved, or more time. |
| job `PENDING` for a long time | Normal queueing. `make -C cluster watch` keeps waiting. |
