#!/bin/bash
# Run one command on a GPU node, from the cluster/ folder (so `python -m encoder.<module>` works).
# Don't sbatch this directly; use cluster/submit.sh (or `make -C cluster submit`), which sets the log
# path and the purpose.
#
# A job is billed for the whole node (288 cores, 4 GPUs) whatever it uses, so it asks for all of it.
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:4
#SBATCH --cpus-per-task=288
#SBATCH --time=01:00:00

set -eo pipefail

# sbatch runs a copy of this script; submit.sh submits from cluster/, so that is the submit dir.
CLUSTER_DIR=$SLURM_SUBMIT_DIR
cd "$CLUSTER_DIR"
source env.sh
source usage_log.sh

export RUN_DIR=$HW3_RUNS/$SLURM_JOB_ID
mkdir -p "$RUN_DIR"
usage_log_start "$*"

# What ran, where and when.
{
    echo "job_id:   $SLURM_JOB_ID"
    echo "purpose:  ${RUN_PURPOSE:-}"
    echo "command:  $*"
    echo "nodes:    $SLURM_JOB_NODELIST"
    echo "started:  $(date "+%F %T")"
    echo "commit:   ${RUN_COMMIT:-unknown}"
} | tee "$RUN_DIR/job.txt"

# Since Slurm 22.05, srun does not inherit --cpus-per-task from sbatch; without this the step gets 1 core.
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
srun "$@"
