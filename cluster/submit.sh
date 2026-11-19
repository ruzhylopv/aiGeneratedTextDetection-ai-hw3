#!/bin/bash
# Submit a command as a GPU job (logged to $HW3_ROOT/usage_log.tsv automatically).
#
#   cluster/submit.sh "<purpose>" <command...>
#
# The command runs from cluster/, e.g.
#   JOB_TIME=00:20:00 cluster/submit.sh "smoke test" python -m encoder.smoke_test
#
# Optional settings: JOB_TIME (default 01:00:00), JOB_NAME (default hw3).
set -eo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

CLUSTER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$CLUSTER_DIR/env.sh"
mkdir -p "$HW3_LOGS" "$HW3_RUNS" "$HW3_MODELS"

export RUN_PURPOSE=$1
# Compute nodes have no git, so record the commit here; sbatch passes the environment on to the job.
export RUN_COMMIT="$(git -C "$CLUSTER_DIR" rev-parse --short HEAD) ($(git -C "$CLUSTER_DIR" rev-parse --abbrev-ref HEAD))"
shift

cd "$CLUSTER_DIR"
name=${JOB_NAME:-hw3}
job_id=$(sbatch --parsable --time="${JOB_TIME:-01:00:00}" --job-name="$name" \
    --output="$HW3_LOGS/%x-%j.out" run.sbatch "$@")
echo "Submitted job $job_id"
echo "  log:     $HW3_LOGS/$name-$job_id.out   (follow: tail -f <log>)"
echo "  outputs: $HW3_RUNS/$job_id/"
echo "  status:  squeue -j $job_id     cancel: scancel $job_id"
