#!/bin/bash
# Create (or update) the Python env. Run once, on the login node (it needs internet):
#
#   make -C cluster env
#
# torch comes from the PyTorch module; the venv adds cluster/requirements.txt on top of it
# (--system-site-packages). Same approach as the fairforget project's slurm/setup_env.sh.
set -eo pipefail

CLUSTER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HW3_SETUP=1 source "$CLUSTER_DIR/env.sh"
mkdir -p "$HW3_ROOT" "$HW3_DATA" "$HW3_RUNS" "$HW3_LOGS"

if [[ ! -d $ENV_DIR ]]; then
    echo "Creating env at $ENV_DIR"
    python3 -m venv --system-site-packages "$ENV_DIR"
fi
# Re-source to activate the env and put it first on PYTHONPATH.
source "$CLUSTER_DIR/env.sh"

echo "Installing cluster/requirements.txt"
pip install --quiet -r "$CLUSTER_DIR/requirements.txt"

# The module stack has some of transformers' dependencies in versions pip accepts but transformers'
# import-time check rejects (e.g. regex). Install exactly what that check wants, read without
# importing transformers.
echo "Upgrading packages that are too old for transformers' import check"
mapfile -t TRANSFORMERS_REQS < <(python - <<'PY'
import ast
import importlib.util
import pathlib
import re

pkg_dir = pathlib.Path(importlib.util.find_spec("transformers").submodule_search_locations[0])
table = {}
exec((pkg_dir / "dependency_versions_table.py").read_text(), table)
check = (pkg_dir / "dependency_versions_check.py").read_text()
names = ast.literal_eval(re.search(r"pkgs_to_check_at_runtime = (\[.*?\])", check, re.S).group(1))
for name in names:
    if name != "python" and name in table["deps"]:
        print(table["deps"][name])
PY
)
pip install --quiet "${TRANSFORMERS_REQS[@]}"

python - "$ENV_DIR" <<'PY'
import sys

import accelerate
import pandas
import sklearn
import torch
import transformers

env_dir = sys.argv[1]
print(f"torch:         {torch.__version__} from {torch.__file__}")
print(f"transformers:  {transformers.__version__}")
print(f"accelerate:    {accelerate.__version__}")
print(f"pandas:        {pandas.__version__} | scikit-learn: {sklearn.__version__}")
if torch.__file__.startswith(env_dir):
    sys.exit("ERROR: pip installed its own torch into the env. It must come from the module.")
print("Env OK.")
PY
