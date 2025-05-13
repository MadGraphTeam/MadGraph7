#!/bin/bash

export PATH="$HOME/dmtcp/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/dmtcp/lib:$LD_LIBRARY_PATH"

export DMTCP_CHECKPOINT_DIR="$INITIAL_DIR/dmtcp_$CONDOR_ID"
mkdir -p "$DMTCP_CHECKPOINT_DIR"

dmtcp_coordinator -i 86400 --daemon --exit-on-last -p 0 --port-file "$DMTCP_CHECKPOINT_DIR/dmtcp.port" 1>/dev/null 2>&1
export DMTCP_COORD_HOST=$(hostname)
export DMTCP_COORD_PORT=$(cat "$DMTCP_CHECKPOINT_DIR/dmtcp.port")

timeout() {
    echo "$(date) - Approaching walltime. Creating checkpoint..."
    if [[ -e "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" ]]; then
        mv $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script_prev.sh
    fi
    dmtcp_command -bcheckpoint
    count=0
    while [ ! -e "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" ] && [ $count -lt 10 ]; do
        echo "$(date) - Waiting for checkpoint..."
        sleep 20
        ((count++))
    done
    if [ $count -eq 10 ]; then
        mv $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script_prev.sh $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh
        echo "$(date) - Checkpoint creation failed. Requeuing..."
    else
        rm $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script_prev.sh
        echo "$(date) - Checkpoint created. Requeuing..."
    fi
    dmtcp_command --quit
    sleep 10
    exit 85
}

# Trap signals
trap "timeout" SIGTERM

cd "$INITIAL_DIR"
if [[ -e "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" ]]; then
    echo "$(date) - Resuming from checkpoint"
    /bin/bash "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" -h $DMTCP_COORD_HOST -p $DMTCP_COORD_PORT &
else
    dmtcp_launch --allow-file-overwrite $@ 2>&1 &
fi

wait

echo "$(date) - Exit"
exit 0
