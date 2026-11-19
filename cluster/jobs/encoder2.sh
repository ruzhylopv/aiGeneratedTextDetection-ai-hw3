#!/bin/bash
# Encoder v2 (BGE-M3 on whole texts), one GPU job: fine-tune on the encoder set (splits/encoder_train.csv),
# then predict the stacker, val and test sets. Submit with `make -C cluster encoder2`; runs from cluster/.
# To redo only the predictions with the saved model: `make -C cluster encoder2-predict`.
set -eo pipefail
TORCHRUN="python -m torch.distributed.run --standalone --nproc_per_node=4"  # = torchrun

$TORCHRUN -m encoder.long_encoder train
$TORCHRUN -m encoder.long_encoder predict --sets stacker val test
