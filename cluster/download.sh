#!/bin/bash
# Download the competition data and the pretrained encoder weights. Run on the login node (compute
# nodes have no internet):
#
#   make -C cluster data
#
# Needs your Kaggle token at ~/.kaggle/access_token (copy it once from the laptop, see README.md).
# Data goes to $HW3_DATA, weights to the Hugging Face cache in $HF_HOME. Safe to re-run.
set -eo pipefail

CLUSTER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$CLUSTER_DIR/env.sh"
export HF_HUB_OFFLINE=0

python - <<'PY'
import os
import shutil
from pathlib import Path

import kagglehub
from huggingface_hub import snapshot_download

COMPETITION = "ai-generated-text-detection-hw3-ucu-ai-2026"
FILES = ["train.csv", "test.csv", "sample_submission.csv"]
# Open-weight encoders only. mDeBERTa is downloaded now for step 3, so no second login-node trip.
MODELS = ["FacebookAI/xlm-roberta-base", "microsoft/mdeberta-v3-base"]
# PyTorch weights, configs and tokenizer files; skips the TF / Flax / ONNX copies.
WEIGHT_PATTERNS = ["*.json", "*.safetensors", "pytorch_model.bin", "*.model", "*.txt"]

src = Path(kagglehub.competition_download(COMPETITION))
dst = Path(os.environ["HW3_DATA"])
for name in FILES:
    shutil.copy2(src / name, dst / name)
    print(f"data:  {dst / name} ({(dst / name).stat().st_size / 1e6:.0f} MB)")

for repo_id in MODELS:
    path = snapshot_download(repo_id, allow_patterns=WEIGHT_PATTERNS)
    print(f"model: {repo_id} -> {path}")
PY
