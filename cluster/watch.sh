#!/bin/bash
# Follow a job from submission to the end:
#
#   make -C cluster watch JOB=<job_id>
#
# While the job waits in the queue, one status line is updated every 15 s (state and Slurm's reason).
# Once its log exists, the log is followed like `tail -f`. When the job ends, watching stops and the
# final state is printed. Ctrl-C only stops watching; the job keeps running.
set -o pipefail

JOB=$1
POLL_SECONDS=15
ACTIVE_STATES="PENDING REQUEUED RUNNING CONFIGURING COMPLETING SUSPENDED RESIZING"

CLUSTER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$CLUSTER_DIR/env.sh"

job_state() {
    # Job state from the accounting database (still works after the job has left the queue); right
    # after submission the database can lag behind, so fall back to the queue.
    local state
    state=$(sacct -X -n -j "$JOB" -o State%30 2>/dev/null | head -n 1 | awk '{print $1}')
    [[ -z $state ]] && state=$(squeue -h -j "$JOB" -o %T 2>/dev/null)
    echo "$state"
}

is_active() {
    [[ " $ACTIVE_STATES " == *" $1 "* ]]
}

log_file() {
    ls "$HW3_LOGS"/*-"$JOB".out 2>/dev/null | head -n 1
}

state=$(job_state)
if [[ -z $state ]]; then
    echo "job $JOB: not found (check the id with: make -C cluster jobs)"
    exit 1
fi

# 1. Wait in the queue until the job has written its log.
while [[ -z $(log_file) ]]; do
    state=$(job_state)
    if ! is_active "$state"; then
        echo
        echo "job $JOB ended ($state) without writing a log."
        sacct -j "$JOB" --format=JobID,State,ExitCode,Start,End,NodeList%20 -D
        exit 1
    fi
    reason=$(squeue -h -j "$JOB" -o %r 2>/dev/null)
    printf '\r%s  job %s: %s (%s)        ' "$(date +%T)" "$JOB" "$state" "${reason:-?}"
    sleep "$POLL_SECONDS"
done

# 2. Follow the log until the job ends.
log=$(log_file)
echo
echo "== $log"
tail -n +1 -f "$log" &
tail_pid=$!
trap 'kill "$tail_pid" 2>/dev/null' EXIT
while is_active "$(job_state)"; do
    sleep "$POLL_SECONDS"
done
sleep 3  # let tail print the last lines
kill "$tail_pid" 2>/dev/null
echo
echo "== job $JOB finished"
sacct -X -j "$JOB" --format=JobID,JobName,State,ExitCode,Elapsed
