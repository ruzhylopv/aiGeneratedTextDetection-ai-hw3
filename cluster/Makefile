# HW3 encoder on JUPITER. Run from the repo root as `make -C cluster <target>`.
.PHONY: env data smoke submit jobs usage pull

ACCOUNT := e-dev-2026d09-075

# ---- JUPITER login node -------------------------------------------------

# One-time: create the Python env (venv on top of the PyTorch module).
env:
	bash setup_env.sh

# One-time: competition data + pretrained weights (needs internet, so login node only).
data:
	bash download.sh

# Short GPU job: GPUs, data, offline model load, a few training steps, throughput.
smoke:
	JOB_TIME=00:20:00 JOB_NAME=smoke bash submit.sh "smoke test" python -m encoder.smoke_test

# Any command as a GPU job (runs from cluster/), e.g.
#   make -C cluster submit PURPOSE="..." CMD="python -m encoder.<module> ..." [JOB_TIME=02:00:00]
submit:
	@test -n "$(PURPOSE)" -a -n "$(CMD)" || \
		{ echo 'usage: make -C cluster submit PURPOSE="..." CMD="python -m ..." [JOB_TIME=hh:mm:ss]'; exit 1; }
	JOB_TIME=$(or $(JOB_TIME),01:00:00) bash submit.sh "$(PURPOSE)" $(CMD)

# Your queued and running jobs.
jobs:
	squeue --me

# Usage log of all HW3 jobs so far.
usage:
	@bash -c 'source env.sh && column -t -s "$$(printf "\t")" "$$HW3_ROOT/usage_log.tsv"'

# ---- Laptop -------------------------------------------------------------

# Copy the small run outputs (metrics, logs, logits) to cluster/results/; checkpoints stay on scratch.
JUPITER ?= jupiter
pull:
	@remote=$$(ssh $(JUPITER) 'echo /e/scratch/$(ACCOUNT)/$$USER/ai-hw3/runs') && \
	rsync -av --exclude 'checkpoint-*/' --exclude '*.safetensors' --exclude '*.bin' --exclude '*.pt' \
		"$(JUPITER):$$remote/" results/
