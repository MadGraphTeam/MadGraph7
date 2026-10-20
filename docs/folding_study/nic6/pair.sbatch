#!/bin/bash
#SBATCH --job-name=fz-pair
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=56G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/ucl/cp3/omatt/fold-zjj/logs/pair-%x-%j.out
# usage: sbatch -J p<tag> pair.sbatch <tag> "<folding>" "<old FKS_params lines>" "<new FKS_params lines>"
# The old code (V_<tag>, before #305) and the new code (W_<tag>) run side by side, 16 cores each.
set -u
source /scratch/ucl/cp3/omatt/fold-zjj/env.sh
T=$1; FOLD=$2; XO=${3:-}; XN=${4:-}
echo "pair $T folding '$FOLD' on $(hostname), job $SLURM_JOB_ID"; lscpu | grep "Model name"; uptime; date
run() { # dir tag extra
  local D=$1 TAG=$2 X=$3
  cp $D/Cards/FKS_params.dat.orig $D/Cards/FKS_params.dat
  [ -n "$X" ] && printf "$X\n" >> $D/Cards/FKS_params.dat
  printf "set run_mode 2\nset nb_core 16\nlaunch -n $TAG\nfixed_order=OFF\nshower=OFF\nmadspin=OFF\nreweight=OFF\ndone\nset iseed 21\nset folding $FOLD\ndone\nquit\n" > $D/launch_$TAG.cmd
  cd $D
  local T0=$(date +%s)
  python3 bin/aMCatNLO $D/launch_$TAG.cmd > $FZ/logs/launch_$TAG.log 2>&1
  echo "$TAG wall $(( $(date +%s)-T0 )) s"
  mkdir -p $D/vlogs/$TAG
  for f in $D/SubProcesses/P0_*/G*/log_MINT*.txt; do d=$(echo $f | sed "s|$D/SubProcesses/||; s|/|_|g"); cp $f $D/vlogs/$TAG/$d; done
  grep "Total cross section" $FZ/logs/launch_$TAG.log | tail -1
}
run $FZ/V_$T o21$T "$XO" &
run $FZ/W_$T n21$T "$XN" &
wait
uptime; date; echo PAIR DONE
