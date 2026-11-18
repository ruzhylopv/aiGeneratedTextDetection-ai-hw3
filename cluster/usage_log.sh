# Append one row per job to the HW3 usage log ($HW3_ROOT/usage_log.tsv). It is kept separate from the
# fairforget team log and has the same columns, so rows can be copied over if the grant report needs them.
# Sourced by run.sbatch; call usage_log_start once and the row is written when the job ends.
#
# Columns: date job_id user project purpose config nodes gpus runtime node_hours result notes
# "runtime" is measured inside the job; `sacct -j <JOB_ID>` has the exact billed time.

_usage_clean() {
    # Tabs and newlines would break the TSV.
    local text=${1//$'\t'/ }
    printf '%s' "${text//$'\n'/ }"
}

usage_log_start() {  # $1 = what ran (command and options)
    USAGE_CONFIG=$1
    USAGE_START=$(date +%s)
    trap _usage_log_write EXIT
    # Slurm sends TERM at the time limit; turn it into a normal exit so the row still gets written.
    trap 'exit 143' TERM
}

_usage_log_write() {
    local status=$?
    local log=$HW3_ROOT/usage_log.tsv
    local elapsed runtime node_hours nodes
    elapsed=$(( $(date +%s) - USAGE_START ))
    runtime=$(printf '%02d:%02d:%02d' $((elapsed / 3600)) $((elapsed % 3600 / 60)) $((elapsed % 60)))
    nodes=${SLURM_JOB_NUM_NODES:-1}
    # Billing is per whole node, whatever the number of GPUs.
    node_hours=$(awk -v s="$elapsed" -v n="$nodes" 'BEGIN { printf "%.3f", s / 3600 * n }')

    if [[ ! -f $log ]]; then
        printf 'date\tjob_id\tuser\tproject\tpurpose\tconfig\tnodes\tgpus\truntime\tnode_hours\tresult\tnotes\n' > "$log"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date +%F)" "$SLURM_JOB_ID" "$USER" "ai-hw3" \
        "$(_usage_clean "${RUN_PURPOSE:-}")" "$(_usage_clean "$USAGE_CONFIG")" \
        "$nodes ($SLURM_JOB_NODELIST)" "${SLURM_GPUS_ON_NODE:-?}" "$runtime" "$node_hours" \
        "exit=$status; ${RUN_DIR:-}" "" >> "$log"
    echo "Usage log: added job $SLURM_JOB_ID to $log"
}
