#!/bin/bash
# Gate check (plan section 1b), one GPU job: a small detector on 10k texts, p(AI) curves of the gate
# texts (128-token windows), and Branch A probabilities of the gate texts.
# Submit with `make -C cluster gate`; runs from cluster/.
set -eo pipefail
TORCHRUN="python -m torch.distributed.run --standalone --nproc_per_node=4"  # = torchrun

$TORCHRUN -m encoder.detector --mode gate
$TORCHRUN -m encoder.curves --model detector-gate --gate --sizes 128
$TORCHRUN -m encoder.branch_a --gate
