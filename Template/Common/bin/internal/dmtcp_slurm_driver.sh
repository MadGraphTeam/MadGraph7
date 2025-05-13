#!/bin/bash

module load releases/2023b # Probably shouldn't be hardcoded
module load DMTCP/3.0.0-GCCcore-13.2.0

count=0
while [ -d "$RUN_DIR/dmtcp_fail" ] && [ $count -lt 10 ]; do
    echo "Waiting for $RUN_DIR/dmtcp_fail to disappear..."
    sleep 20
    ((count++))
done

export DMTCP_CHECKPOINT_DIR="$RUN_DIR/dmtcp_$SLURM_JOB_ID"
mkdir -p "$DMTCP_CHECKPOINT_DIR"

dmtcp_coordinator -i 86400 --daemon --exit-on-last -p 0 --port-file "$DMTCP_CHECKPOINT_DIR/dmtcp.port" 1>/dev/null 2>&1
export DMTCP_COORD_HOST=$(hostname)
export DMTCP_COORD_PORT=$(cat "$DMTCP_CHECKPOINT_DIR/dmtcp.port")

timeout() {
    echo "Approaching walltime. Creating checkpoint..."
    if [[ -e "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" ]]; then
        mv $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script_prev.sh
    fi
    dmtcp_command -bcheckpoint
    count=0
    while [ ! -e "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" ] && [ $count -lt 10 ]; do
        echo "Incomplete checkpoint. Waiting..."
        sleep 20
        ((count++))
    done
    if [ $count -eq 10 ]; then
        mv $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script_prev.sh $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh
        echo "Checkpoint creation failed. Requeuing..."
    else
        rm $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script_prev.sh
        echo "Checkpoint created. Requeuing..."
    fi
    dmtcp_command --quit
    sleep 2
    scontrol requeue $SLURM_JOB_ID
    sleep 10
    exit 85
}

# Trap signals
trap "timeout" USR1

if [[ -e "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" ]]; then
    echo "$(date) - Resuming from checkpoint. Restart: ${SLURM_RESTART_COUNT}"
    srun /bin/bash "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" -h $DMTCP_COORD_HOST -p $DMTCP_COORD_PORT &
else
    srun dmtcp_launch --allow-file-overwrite $@ &
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
