#!/bin/bash

module load releases/2023b # Probably shouldn't be hardcoded
module load DMTCP/3.0.0-GCCcore-13.2.0

export DMTCP_CHECKPOINT_DIR=$RUN_DIR/dmtcp_$SLURM_JOB_ID
mkdir $DMTCP_CHECKPOINT_DIR

dmtcp_coordinator -i 86400 --daemon --exit-on-last -p 0 --port-file $DMTCP_CHECKPOINT_DIR/dmtcp.port 1>/dev/null 2>&1
export DMTCP_COORD_HOST=$(hostname)
export DMTCP_COORD_PORT=$(cat $DMTCP_CHECKPOINT_DIR/dmtcp.port)

timeout() {
    echo "Approaching walltime. Creating checkpoint..."
    dmtcp_command -bcheckpoint
    sleep 2
    echo "Checkpoint created. Requeuing..."
    dmtcp_command --quit
    sleep 2
    scontrol requeue $SLURM_JOB_ID
    sleep 10
    exit 85
}

# Trap signals
trap 'timeout' USR1

if [[ -e $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh && "${SLURM_RESTART_COUNT}" != "" ]]; then
    echo "$(date) - Resuming from checkpoint. Restart: ${SLURM_RESTART_COUNT}"
    srun /bin/bash $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh -h $DMTCP_COORD_HOST -p $DMTCP_COORD_PORT &
else
    srun dmtcp_launch --allow-file-overwrite $@ &
fi
wait
exit 0
