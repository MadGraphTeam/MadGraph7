#!/bin/bash
# Runs on the cluster LOGIN node, piped through ssh by ceci-gate (vm_gate/ceci-gate) for the
# start_runner job of gpu_runner_ci.yml:
#     ssh <cluster> bash -l -s -- LABEL RUN_ID RUN_TAG IDLE_MINUTES [sbatch options...] -- TIER... < remote_start_runner.sh
# The copy that runs is the one installed on the vmadgraph VM (/usr/local/share/ceci-gate):
# install it there again after a change (README.md, "One-time setup").
# Submits runner_batch.sh (copied beforehand into ~/mg7-gpu-ci/runner-LABEL, next to
# job_started_hook.sh) once per TIER, and waits until one of them has the runner connected to
# GitHub. A TIER is one argument "HH:MM:SS [sbatch options...]" (checked by ceci-gate): e.g.
# 30 minutes on the fast GPUs, 60 on the others. The options before -- are common to all.
#  - the runner job of tier N is called mg7-gh-runner-LABEL (N = 1) or mg7-gh-runner-LABEL-N,
#    with --dependency=singleton: a runner job only starts once the previous one of its name
#    has stopped. Not one name for all: singleton would hold every tier but the first back,
#    even while the first is only waiting in the queue;
#  - the first runner job that starts cancels the other tiers, and waits until the runner jobs
#    of the other names, of older runs, have stopped (runner_batch.sh): one runner at a time;
#  - RUN_TAG is stored as the Slurm comment, to recognise the jobs in squeue.
# Exits non-zero, with the runner logs, if every runner job ends before being ready
# (e.g. registration deleted by GitHub: run install_runner.sh again).
LABEL=${1:?missing LABEL}
RUN_ID=${2:?missing RUN_ID}
RUN_TAG=${3:?missing RUN_TAG}
IDLE_MINUTES=${4:?missing IDLE_MINUTES}
shift 4
COMMON=()
while [ $# -gt 0 ] && [ "$1" != -- ]; do
    COMMON+=("$1")
    shift
done
if [ "${1:-}" != -- ] || [ $# -lt 2 ]; then
    echo "ERROR: no tier after -- (ceci-gate and remote_start_runner.sh are not the same version?)"
    exit 1
fi
shift
RUNNER_DIR=$HOME/mg7-gpu-ci/runner-$LABEL
if [ ! -f "$RUNNER_DIR/.runner" ]; then
    echo "ERROR: no runner configured in $RUNNER_DIR, run .github/gpu_ci/install_runner.sh first"
    exit 1
fi
NAME=mg7-gh-runner-$LABEL

IDS=()
NAMES=()
cancel_all() {
    local i
    for i in "${!IDS[@]}"; do
        scancel --name="${NAMES[$i]}" "${IDS[$i]}" 2> /dev/null
    done
}
N=0
for TIER in "$@"; do
    N=$((N + 1))
    JOBNAME=$NAME
    [ "$N" -eq 1 ] || JOBNAME=$NAME-$N
    # read -a: split on spaces, without glob expansion of the [] of node lists
    read -r TIME REST <<< "$TIER"
    read -r -a OPTS <<< "$REST"
    if ! JOBID=$(sbatch --parsable --job-name="$JOBNAME" --comment="$RUN_TAG" \
                        --dependency=singleton --chdir="$RUNNER_DIR" --output="$RUNNER_DIR/slurm-%j.out" \
                        "${COMMON[@]}" --time="$TIME" "${OPTS[@]}" \
                        "$RUNNER_DIR/runner_batch.sh" "$RUN_ID" "$RUNNER_DIR/stop-$RUN_TAG" "$IDLE_MINUTES" \
                        "$RUN_TAG" "$NAME"); then
        echo "ERROR: sbatch of tier $N ($TIER) failed"
        cancel_all
        exit 1
    fi
    JOBID=${JOBID%%;*}
    IDS+=("$JOBID")
    NAMES+=("$JOBNAME")
    echo "Submitted runner job $JOBID $JOBNAME: --time=$TIME ${OPTS[*]} (${COMMON[*]})"
done

declare -A LAST
while true; do
    ALIVE=0
    for i in "${!IDS[@]}"; do
        JOBID=${IDS[$i]}
        LOG=$RUNNER_DIR/slurm-$JOBID.out
        STATE=$(squeue -h -j "$JOBID" -o %T 2> /dev/null)
        case $STATE in
            PENDING|CONFIGURING|RUNNING) ALIVE=$((ALIVE + 1)) ;;
        esac
        if [ "$STATE" = RUNNING ] && grep -q 'Listening for Jobs' "$LOG" 2> /dev/null; then
            cat "$LOG"
            echo "Runner $LABEL online in Slurm job $JOBID on $(squeue -h -j "$JOBID" -o %N) (${NAMES[$i]}, time limit $(squeue -h -j "$JOBID" -o %l))"
            # the runner job has cancelled the other tiers already; again, in case
            for j in "${!IDS[@]}"; do
                [ "$j" = "$i" ] || scancel --name="${NAMES[$j]}" "${IDS[$j]}" 2> /dev/null
            done
            exit 0
        fi
        if [ "$STATE" != "${LAST[$JOBID]:-}" ]; then
            echo "$(date +%T) job $JOBID ${NAMES[$i]}: ${STATE:-ended} $(squeue -h -j "$JOBID" -o %r 2> /dev/null)"
            LAST[$JOBID]=$STATE
        fi
    done
    if [ "$ALIVE" -eq 0 ]; then
        echo "ERROR: every runner job ended before the runner was ready:"
        for JOBID in "${IDS[@]}"; do
            echo "--- runner job $JOBID"
            cat "$RUNNER_DIR/slurm-$JOBID.out" 2> /dev/null
        done
        exit 1
    fi
    sleep 20
done
