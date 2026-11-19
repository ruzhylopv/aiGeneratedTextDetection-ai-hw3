#!/bin/bash
# Scoring only (plan steps 4–5): curves of the full detector and Branch A probabilities for the
# stacker, val and test sets. Submit with `make -C cluster score`; runs from cluster/.
set -eo pipefail
TORCHRUN="python -m torch.distributed.run --standalone --nproc_per_node=4"  # = torchrun

$TORCHRUN -m encoder.curves --model detector-full --sets stacker val test
$TORCHRUN -m encoder.branch_a --sets stacker val test
