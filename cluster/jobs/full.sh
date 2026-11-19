#!/bin/bash
# Full run (plan steps 3–5), one GPU job: the detector on the whole detector set, then the scoring of
# the stacker, val and test sets (curves at 64/128/256 tokens and Branch A probabilities).
# Submit with `make -C cluster full`; runs from cluster/. To redo only the scoring: `make -C cluster score`.
set -eo pipefail
TORCHRUN="python -m torch.distributed.run --standalone --nproc_per_node=4"  # = torchrun

$TORCHRUN -m encoder.detector --mode full
bash jobs/score.sh
