#!/bin/bash

module load releases/2023b # Probably shouldn't be hardcoded
module load DMTCP/3.0.0-GCCcore-13.2.0

dmtcp_coordinator -i 300 --daemon --exit-on-last -p 0 --port-file dmtcp.port 1>/dev/null 2>&1
export DMTCP_COORD_HOST=$(hostname)
export DMTCP_COORD_PORT=$(cat dmtcp.port)

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

if [[ -e dmtcp_restart_script.sh && "${SLURM_RESTART_COUNT}" != "" ]]; then
    echo "$(date) - Resuming from checkpoint. Restart: ${SLURM_RESTART_COUNT}"
    srun /bin/bash ./dmtcp_restart_script.sh -h $DMTCP_COORD_HOST -p $DMTCP_COORD_PORT &
else
    srun dmtcp_launch --allow-file-overwrite $@ &
fi
wait
exit 0
