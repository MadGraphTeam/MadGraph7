#!/bin/bash

export PATH="$HOME/dmtcp/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/dmtcp/lib:$LD_LIBRARY_PATH"

while [ -d "$RUN_DIR/dmtcp_fail" ]; do
    echo "Waiting for $RUN_DIR/dmtcp_fail to disappear..."
    sleep 1
done

export DMTCP_CHECKPOINT_DIR="$RUN_DIR/dmtcp_$CONDOR_ID"
mkdir -p "$DMTCP_CHECKPOINT_DIR"

dmtcp_coordinator -i 86400 --daemon --exit-on-last -p 0 --port-file "$DMTCP_CHECKPOINT_DIR/dmtcp.port" 1>/dev/null 2>&1
export DMTCP_COORD_HOST=$(hostname)
export DMTCP_COORD_PORT=$(cat "$DMTCP_CHECKPOINT_DIR/dmtcp.port")

timeout() {
    echo "Approaching walltime. Creating checkpoint..."
    dmtcp_command -bcheckpoint
    sleep 2
    echo "Checkpoint created. Requeuing..."
    dmtcp_command --quit
    sleep 10
    exit 85
}

# Trap signals
trap "timeout" USR1

if [[ -e "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" ]]; then
    echo "$(date) - Resuming from checkpoint. Restart: ${SLURM_RESTART_COUNT}"
    /bin/bash "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" -h $DMTCP_COORD_HOST -p $DMTCP_COORD_PORT &
else
    dmtcp_launch --allow-file-overwrite $@ &
fi

wait

# Calculation finished, cleanup
link="$DMTCP_CHECKPOINT_DIR"

while [ -L "$link" ]; do
    next=$(readlink "$link")
    rm "$link"
    link="$next"
done

rm -r "$link"
exit 0
