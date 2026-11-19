#!/bin/bash
# Download the tuned Branch A model (the Kaggle output of hw3-encoder-tuning) to $HW3_MODELS/branch_a/.
# Run on the login node (compute nodes have no internet):
#
#   make -C cluster branch-a
#
# Needs your Kaggle token at ~/.kaggle/access_token. Safe to re-run.
set -eo pipefail

CLUSTER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$CLUSTER_DIR/env.sh"

python - <<'PY'
import os
from pathlib import Path

import kagglehub

out = Path(os.environ["HW3_MODELS"]) / "branch_a"
kagglehub.notebook_output_download("ruzhylopv/hw3-encoder-tuning", output_dir=str(out))
model = out / "encoder" / "model"
files = sorted(p.name for p in model.iterdir())
print(f"Branch A model in {model}: {files}")
size = sum(p.stat().st_size for p in model.glob("*.safetensors"))
assert size > 1e9, f"weights missing or too small ({size / 1e9:.2f} GB of .safetensors)"
print(f"OK: {size / 1e9:.2f} GB of weights")
PY
