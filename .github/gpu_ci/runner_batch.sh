#!/bin/bash
# Slurm batch script, submitted by remote_start_runner.sh from the runner directory, where
# ceci-gate copies it at every start from the vmadgraph VM (/usr/local/share/ceci-gate):
# runs the self-hosted GitHub runner for ONE workflow run, then ends the allocation,
# so that all the CI jobs of that run share this single allocation.
#     sbatch [options] runner_batch.sh RUN_ID STOP_FILE IDLE_MINUTES [RUN_TAG NAME]
#   RUN_ID        the only GitHub workflow run whose jobs are accepted (job_started_hook.sh)
#   STOP_FILE     created by the last CI job (stop_runner): the runner is then stopped
#   IDLE_MINUTES  stop anyway when no job ran for that long (e.g. the run was cancelled)
#   RUN_TAG NAME  the run attempt (the Slurm comment) and the job name of the first tier,
#                 when remote_start_runner.sh submitted one runner job per tier (NAME, NAME-2...)
# The job output (this file's and run.sh's) goes to slurm-<jobid>.out in this directory.
export CI_RUN_ID=${1:?missing RUN_ID}
export CI_STOP_FILE=${2:?missing STOP_FILE}
IDLE_MINUTES=${3:-10}
RUN_TAG=${4:-}
NAME=${5:-}
export ACTIONS_RUNNER_HOOK_JOB_STARTED=$PWD/job_started_hook.sh
LOG=$PWD/slurm-$SLURM_JOB_ID.out
echo "Runner for workflow run $CI_RUN_ID, Slurm job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname)"

if [ -n "$RUN_TAG" ] && [ -n "$NAME" ]; then
    # the runner jobs of every tier, of any run: "JOBID STATE RUN_TAG JOBNAME"
    runner_jobs() {
        squeue -h -u "$USER" -o '%i %T %k %j' 2> /dev/null |
            awk -v n="$NAME" '$4 == n || (index($4, n "-") == 1 && substr($4, length(n) + 2) ~ /^[0-9]+$/)'
    }
    # 1. The tiers of this run: the first to start runs the CI. A tier started at the same time
    #    with a lower job id wins; otherwise this one cancels the others.
    while read -r ID STATE TAG JOBNAME; do
        [ -n "$ID" ] && [ "$ID" != "$SLURM_JOB_ID" ] && [ "$TAG" = "$RUN_TAG" ] || continue
        case $STATE in
            CONFIGURING|RUNNING|COMPLETING)
                if [ "$ID" -lt "$SLURM_JOB_ID" ]; then
                    echo "Runner job $ID ($JOBNAME) of the same run started too: it runs the CI, not this one"
                    exit 0
                fi ;;
        esac
    done <<< "$(runner_jobs)"
    while read -r ID STATE TAG JOBNAME; do
        [ -n "$ID" ] && [ "$ID" != "$SLURM_JOB_ID" ] && [ "$TAG" = "$RUN_TAG" ] || continue
        echo "Cancelling the runner job $ID ($JOBNAME, $STATE) of another tier"
        scancel --name="$JOBNAME" "$ID"
    done <<< "$(runner_jobs)"
    # 2. Runner jobs of older runs under another name (the same name is held back by
    #    --dependency=singleton): wait until they stop, one runner at a time. ceci-gate has
    #    cancelled those of the runs that are over, so this is at most the end of one.
    for _ in $(seq 40); do
        OTHERS=$(runner_jobs | awk -v tag="$RUN_TAG" '$3 != tag && ($2 == "CONFIGURING" || $2 == "RUNNING" || $2 == "COMPLETING")')
        [ -n "$OTHERS" ] || break
        echo "$(date +%T) waiting for the runner job of another run to stop: $OTHERS"
        sleep 15
    done
fi

rm -f "$CI_STOP_FILE"

# job control: run.sh gets its own process group (PGID = its PID), so that the runner
# and all its children can be stopped with one signal below
set -m
./run.sh &
RUNNER_PID=$!
while kill -0 "$RUNNER_PID" 2> /dev/null; do
    sleep 15
    if [ -e "$CI_STOP_FILE" ]; then
        # give the runner the time to report the end of the stop_runner job: it uploads the
        # logs first, which can take a minute. Stopped before, the job stays in progress on
        # GitHub, and the watchdog takes the end of the allocation for an early one
        for _ in $(seq 300); do
            grep -q -F ' stop_runner completed with result' "$LOG" && break
            sleep 1
        done
        echo "Stop requested by the workflow"
        break
    fi
    IDLE=$(( ($(date +%s) - $(stat -c %Y "$LOG")) / 60 ))
    if [ "$IDLE" -ge "$IDLE_MINUTES" ] && tail -n 1 "$LOG" | grep -q -E 'Listening for Jobs|completed with result'; then
        echo "No job for $IDLE minutes, stopping"
        break
    fi
done
kill -TERM -- "-$RUNNER_PID" 2> /dev/null || kill -TERM "$RUNNER_PID" 2> /dev/null
wait "$RUNNER_PID"
rm -f "$CI_STOP_FILE"
