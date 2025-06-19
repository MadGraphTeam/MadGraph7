#!/bin/bash

export PATH="$DMTCP_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$DMTCP_PATH/lib:$LD_LIBRARY_PATH"

# Check if shared directory is set
if [ -z "${SHARED_DIR+x}" ]; then
    out="/dev/null"
    echo "$(date) - Shared disk disabled" | tee -a $out
    export DMTCP_CHECKPOINT_DIR="$PWD/dmtcp_$CONDOR_ID"
    echo "$(date) - Shared disk disabled"
    src="$INITIAL_DIR/dmtcp_$CONDOR_ID"
    if [ -d "$src" ]; then
        echo "$(date) - Transferring checkpoint files from $src..." | tee -a $out
        cp -r "$src" .
        sed -i -E "s|/sandbox|$DMTCP_CHECKPOINT_DIR|g" $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh
    fi
else
    if [ -d "$SHARED_DIR" ]; then
        out="$SHARED_DIR/condor_$CONDOR_ID.out"
        echo "$(date) - Shared disk enabled" | tee -a $out
        export DMTCP_CHECKPOINT_DIR="$SHARED_DIR/dmtcp_$CONDOR_ID"
        cd "$SHARED_DIR"
    else
        exit 0
    fi
fi

mkdir -p "$DMTCP_CHECKPOINT_DIR"

dmtcp_coordinator -i 86400 --daemon --exit-on-last -p 0 --port-file "$DMTCP_CHECKPOINT_DIR/dmtcp.port" 1>/dev/null 2>&1
export DMTCP_COORD_HOST=$(hostname)
export DMTCP_COORD_PORT=$(cat "$DMTCP_CHECKPOINT_DIR/dmtcp.port")

timeout() {
    echo "$(date) - Approaching walltime. Creating checkpoint..." | tee -a $out
    if [[ -e "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" ]]; then
        mv $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script_prev.sh
    fi
    dmtcp_command -bcheckpoint
    count=0
    while [ ! -e "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" ] && [ $count -lt 10 ]; do
        echo "$(date) - Waiting for checkpoint..." | tee -a $out
        sleep 20
        ((count++))
    done
    if [ $count -eq 10 ]; then
        mv $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script_prev.sh $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh
        echo "$(date) - Checkpoint creation failed. Requeuing..." | tee -a $out
    else
        rm $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script_prev.sh
        if [ -z "${SHARED_DIR+x}" ]; then
            sed -i -E "s|$DMTCP_CHECKPOINT_DIR|/sandbox|g" $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh
        fi
        echo "$(date) - Checkpoint created. Requeuing..." | tee -a $out
    fi
    dmtcp_command --quit
    sleep 10
    exit 85
}

# Trap signals
trap "timeout" SIGTERM

if [[ -e "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" ]]; then
    echo "$(date) - Resuming from checkpoint" | tee -a $out
    /bin/bash "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" -h $DMTCP_COORD_HOST -p $DMTCP_COORD_PORT | tee -a $out &
else
    dmtcp_launch --allow-file-overwrite $@ | tee -a $out 2>&1 &
fi

wait

echo "$(date) - Exit" | tee -a $out
exit 0
