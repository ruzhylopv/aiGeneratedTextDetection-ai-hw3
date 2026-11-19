#!/bin/bash
# Print the local folder of Qwen3-8B (the folder with its config.json) in the team's shared models folder.
#   bash find_qwen.sh            (or: make -C cluster qwen-path)
# Prints nothing and exits 1 if it is not found.
SHARED_MODELS=/e/project1/e-dev-2026d09-075/fairforget/models
config=$(find "$SHARED_MODELS"/Qwen3-8B -maxdepth 3 -name config.json 2>/dev/null | head -n 1)
[[ -n $config ]] || exit 1
dirname "$config"
