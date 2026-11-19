# Modules, paths, caches and the Python env for HW3 on JUPITER. Source it, don't run it:
#
#   source cluster/env.sh
#
# Every cluster script sources this file, so it is the only place that knows about modules and paths.
# Everything lives under one folder on scratch (files older than 90 days are deleted there; the small
# results are pulled back to the laptop with `make pull`).

PROJECT_ACCOUNT=e-dev-2026d09-075
export HW3_ROOT=/e/scratch/$PROJECT_ACCOUNT/$USER/ai-hw3
export HW3_DATA=$HW3_ROOT/data
export HW3_RUNS=$HW3_ROOT/runs
export HW3_LOGS=$HW3_ROOT/logs
export HW3_MODELS=$HW3_ROOT/models  # checkpoints reused by later jobs (not pulled to the laptop)
ENV_DIR=$HW3_ROOT/venv

# Without GCCcore the PyTorch module is hidden and there is no `python`. torch comes from this module
# (built for GH200 / aarch64); the venv only adds the packages in cluster/requirements.txt.
module load Stages/2026 GCCcore/.14.3.0 PyTorch/2.9.1

# Keep caches out of $HOME (20 GB / 80k files quota).
CACHE_DIR=$HW3_ROOT/cache
export XDG_CACHE_HOME=$CACHE_DIR
export PIP_CACHE_DIR=$CACHE_DIR/pip
export HF_HOME=$CACHE_DIR/huggingface
export KAGGLEHUB_CACHE=$CACHE_DIR/kagglehub
export TRITON_CACHE_DIR=$CACHE_DIR/triton

# Compute nodes have no internet: fail fast instead of hanging on a Hub download.
# download.sh overrides this on the login node.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

# Show prints in the job log immediately, so `tail -f` shows progress.
export PYTHONUNBUFFERED=1

# Homework is independent work: new files are private to you, not readable by the project group.
umask 077

if [[ -f $ENV_DIR/bin/activate ]]; then
    source "$ENV_DIR/bin/activate"
    # Modules put their packages on PYTHONPATH, which Python searches before the venv. Put the venv
    # first so packages installed in it win over older module copies.
    ENV_SITE=$(echo "$ENV_DIR"/lib/python3.*/site-packages)
    export PYTHONPATH=$ENV_SITE${PYTHONPATH:+:$PYTHONPATH}
elif [[ -z $HW3_SETUP ]]; then
    echo "cluster/env.sh: no Python env at $ENV_DIR yet. Create it once with: make -C cluster env" >&2
fi
