#!/bin/bash
# Qwen3-8B per-token log-probabilities (no training) for the detector, stacker, val and test sets.
# Submit with `make -C cluster qwen`; runs from cluster/. QWEN_MODEL overrides the model folder;
# by default it is found in the team's shared models folder (find_qwen.sh).
set -eo pipefail
TORCHRUN="python -m torch.distributed.run --standalone --nproc_per_node=4"  # = torchrun

QWEN_MODEL=${QWEN_MODEL:-$(bash find_qwen.sh)}
[[ -n $QWEN_MODEL ]] || { echo "Qwen3-8B not found; set QWEN=<folder with config.json>"; exit 1; }
echo "model: $QWEN_MODEL"
$TORCHRUN -m encoder.qwen_logprobs --model "$QWEN_MODEL" --sets detector stacker val test
