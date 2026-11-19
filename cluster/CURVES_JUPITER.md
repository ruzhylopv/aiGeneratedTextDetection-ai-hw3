# Curve pipeline on JUPITER: step by step

What runs on the cluster for the own approach (p(AI) curves). Three things, in this order:

| Step | Command | Where | Time |
|---|---|---|---|
| A | `make -C cluster branch-a` | login node | ~5 min, once |
| B | `make -C cluster gate` | GPU job | ~20–30 min |
| C | `make -C cluster full` | GPU job | ~1.5–3 h, only after the gate says "go" |

After B and after C, pull the results to the laptop (`make -C cluster pull`) and send me the job id.

---

## 0. Before you start

1. **Renew the SSH certificate.** It lasts only ~10 hours; the one on the laptop expired on 2026-11-11.
   1. Log in at https://sshca.my-eurohpc.eu/ (EuroHPC SSH CA, via MyAccessID) and copy the one-time token.
   2. On the laptop:
      ```bash
      ssh -i ~/.ssh/id_EFP -o IdentitiesOnly=yes -p 2226 sshca.my-eurohpc.eu token <TOKEN> \
          > ~/.ssh/id_login.jupiter.fz-juelich.de-cert.pub
      ssh-keygen -L -f ~/.ssh/id_login.jupiter.fz-juelich.de-cert.pub | grep Valid   # new validity window
      ssh jupiter hostname
      ```
2. **Check the env exists.** If you have never run `make -C cluster env`, `make -C cluster data` and `make -C cluster smoke`, do the "First-time setup" in `cluster/README.md` first. The smoke test must end with `All required checks passed.`

## 1. Get the new code (login node)

```bash
cd /e/scratch/e-dev-2026d09-075/$USER/ai-hw3/repo
git fetch origin
git checkout curve-fusion
git pull
git log --oneline -1        # must show the latest commit of curve-fusion
ls splits/                  # must list curve_roles.csv and gate_texts.csv
```

## 2. Step A: download Branch A (login node, once)

```bash
make -C cluster branch-a
```

This downloads the Kaggle output of `hw3-encoder-tuning` to `$HW3_ROOT/models/branch_a/`. It needs `~/.kaggle/access_token` on JUPITER (it's already there if `make -C cluster data` worked).

**Expected end of the output:**
```
Branch A model in .../models/branch_a/encoder/model: ['config.json', 'model.safetensors', 'tokenizer.json', ...]
OK: 1.1 GB of weights      (about; anything above 1.0 GB is fine)
```
If it stops with `AssertionError: weights missing or too small`, send me the printed file list.

## 3. Step B: gate job (GPU)

```bash
make -C cluster gate
```
It prints the job id and the log path. Follow the job with:
```bash
make -C cluster watch JOB=<job_id>   # waits while queued, then follows the log until the job ends
```
Ctrl-C only stops watching; the job keeps running.

The job runs three parts, one after the other, on all 4 GPUs:
1. `encoder.detector --mode gate`: trains a small human-vs-ai detector on 10k texts. The log shows `detector-gate: 10,000 texts ...`, the training progress, and finally a JSON summary with `train_minutes`.
2. `encoder.curves --gate`: p(AI) curves of the 2,540 gate texts. The log shows `gate: 2,540 texts, sizes [128]`, then `done`.
3. `encoder.branch_a --gate`: Branch A probabilities of the same texts. Again `gate: 2,540 texts` and `done`.

**It worked if** the run folder has these files:
```bash
ls /e/scratch/e-dev-2026d09-075/$USER/ai-hw3/runs/<job_id>/
# detector-gate.json  detector-gate_loss.csv  curves_gate_128.part0.csv.gz ... part3
# branch_a_probs_gate.part0.csv ... part3  job.txt
```

**Then on the laptop:**
```bash
make -C cluster pull     # copies runs/ to cluster/results/ (no checkpoints)
```
Send me the job id. I'll do the gate plots and the K1–K3 numbers, and tell you go / no-go.

## 4. Step C: full run (GPU, only after "go")

```bash
make -C cluster full
```
One job (time limit 3 h) that:
1. trains the full detector (`encoder.detector --mode full`: ~34.6k texts, 128- and 256-token windows, 2 epochs) and saves it to `$HW3_ROOT/models/detector-full/`
2. scores the stacker, val and test sets (`jobs/score.sh`): curves at 64, 128 and 256 tokens, and Branch A probabilities.

**It worked if** the run folder has `detector-full.json`, and for each of `stacker`, `val`, `test`: `curves_<set>_{64,128,256}.part{0..3}.csv.gz` and `branch_a_probs_<set>.part{0..3}.csv`. That is 36 + 12 files plus the JSON and the loss log.

**If training finished but scoring failed** (the checkpoint is in `$HW3_ROOT/models/detector-full/`), redo only the scoring:
```bash
make -C cluster score
```

Then pull on the laptop with `make -C cluster pull` and send me the job id.

## 5. Optional: experiment E6 (only if there is time)

The detector initialised from plain xlm-roberta-base instead of Branch A:
```bash
make -C cluster submit PURPOSE="curves: detector from xlm-roberta-base" JOB_TIME=01:30:00 \
    CMD="python -m torch.distributed.run --standalone --nproc_per_node=4 -m encoder.detector --mode full --init xlmr"
```
Ask me before running it. It saves to `$HW3_ROOT/models/detector-full-xlmr/`; scoring it is
`encoder.curves --model detector-full-xlmr --sets stacker val` (no code change needed).

---

## If something fails

Send me the last ~40 lines of the log (`tail -n 40 <log>`). Common cases:

| Message in the log | Meaning / fix |
|---|---|
| `KeyError: 'HW3_MODELS'` | The repo on JUPITER has an old `cluster/env.sh`. Run `git pull` on `curve-fusion`. |
| `FileNotFoundError: .../splits/curve_roles.csv` | Same: `git pull`. |
| `OSError: ... models/branch_a/encoder/model ...` | Step A wasn't run or failed. |
| `CUDA out of memory` | Tell me. I'll lower the batch sizes. |
| `LocalEntryNotFoundError` / `HF_HUB_OFFLINE` | A model isn't in the HF cache. Only `--init xlmr` needs `FacebookAI/xlm-roberta-base`, which `make -C cluster data` downloads. |
| job `PENDING` for hours | Queue. Check `make -C cluster jobs`. Nothing to fix. |

Usage (node-hours) is logged automatically. Run `make -C cluster usage` to see it.
