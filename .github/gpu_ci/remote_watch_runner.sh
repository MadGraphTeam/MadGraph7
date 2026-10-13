#!/bin/bash
# Runs on the cluster LOGIN node, piped through ssh by ceci-gate (vm_gate/ceci-gate) for the
# watchdog job of gpu_runner_ci.yml:
#     ssh <cluster> bash -l -s -- LABEL RUN_TAG < remote_watch_runner.sh
# The copy that runs is the one installed on the vmadgraph VM (/usr/local/share/ceci-gate):
# install it there again after a change (README.md, "One-time setup").
# Waits until the runner job of the workflow run has ended: the Slurm job called
# mg7-gh-runner-LABEL whose comment is RUN_TAG (set by remote_start_runner.sh). Then prints
#     ALLOCATION_ENDED JOBID STATE ELAPSED TIMELIMIT
# (state from sacct: COMPLETED, TIMEOUT, NODE_FAIL, PREEMPTED, CANCELLED...) and exits 0.
# Whether that end was the normal one (after stop_runner) is decided by the watchdog, from GitHub.
# Exits non-zero, without that line, if squeue keeps failing: the watchdog calls again.
LABEL=${1:?missing LABEL}
RUN_TAG=${2:?missing RUN_TAG}
NAME=mg7-gh-runner-$LABEL

JOBID=
LAST=
SHOWN=0
FAILS=0
while true; do
    # by name, not by id: squeue -j fails once the job has left the controller
    if ! JOBS=$(squeue -h -u "$USER" -n "$NAME" -o '%i %T %k %M %l' 2>&1); then
        FAILS=$((FAILS + 1))
        echo "$(date +%T) squeue failed ($FAILS): $JOBS"
        [ "$FAILS" -lt 30 ] || exit 1
        sleep 20
        continue
    fi
    FAILS=0
    read -r ID STATE _ USED LIMIT <<< "$(awk -v tag="$RUN_TAG" '$3 == tag' <<< "$JOBS")"
    JOBID=${ID:-$JOBID}
    case $STATE in
        PENDING|CONFIGURING|RUNNING|SUSPENDED) ;;
        *) break ;;
    esac
    # on every change of state, and every 10 minutes: how much of the time limit is used
    if [ "$STATE" != "$LAST" ] || [ $(($(date +%s) - SHOWN)) -ge 600 ]; then
        echo "$(date +%T) runner job $JOBID: $STATE, $USED of $LIMIT"
        LAST=$STATE
        SHOWN=$(date +%s)
    fi
    sleep 20
done

if [ -z "$JOBID" ]; then
    echo "ALLOCATION_ENDED ? no running job $NAME with comment $RUN_TAG in squeue"
    exit 0
fi
echo "$(date +%T) runner job $JOBID: ${STATE:-no longer in squeue}"
# the accounting database lags a few seconds behind the controller
for _ in $(seq 6); do
    FINAL=$(sacct -n -X -P -j "$JOBID" -o State,Elapsed,Timelimit 2> /dev/null | head -n 1 | tr '|' ' ')
    case $FINAL in
        ''|RUNNING*|COMPLETING*) sleep 5 ;;
        *) break ;;
    esac
done
echo "ALLOCATION_ENDED $JOBID ${FINAL:-${STATE:-?} (no sacct)}"
