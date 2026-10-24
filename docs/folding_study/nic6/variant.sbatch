#!/bin/bash
#SBATCH --job-name=fz-var
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ucl/cp3/omatt/fold-zjj/logs/var-%x-%j.out
# usage: sbatch -J <tag> variant.sbatch <tag> "<folding>" "<FKS_params lines>"
set -u
source /scratch/ucl/cp3/omatt/fold-zjj/env.sh
T=$1; FOLD=$2; EXTRA=${3:-}
D=$FZ/V_$T
echo "variant $T folding '$FOLD' extra '$EXTRA' on $(hostname), job $SLURM_JOB_ID"; lscpu | grep "Model name"; uptime; date
cp $D/Cards/FKS_params.dat.orig $D/Cards/FKS_params.dat
[ -n "$EXTRA" ] && printf "$EXTRA\n" >> $D/Cards/FKS_params.dat
cat > $D/launch.cmd <<EOL
set run_mode 2
set nb_core 32
launch -n z21$T
fixed_order=OFF
shower=OFF
madspin=OFF
reweight=OFF
done
set iseed 21
set folding $FOLD
done
quit
EOL
cd $D
T0=$(date +%s)
python3 bin/aMCatNLO $D/launch.cmd > $FZ/logs/launch_$T.log 2>&1
T1=$(date +%s)
mkdir -p $D/vlogs/z21$T
for f in $D/SubProcesses/P0_*/G*/log_MINT*.txt; do d=$(echo $f | sed "s|$D/SubProcesses/||; s|/|_|g"); cp $f $D/vlogs/z21$T/$d; done
grep "cross section" $FZ/logs/launch_$T.log
echo "wall $((T1-T0)) s"; uptime; date; echo VARIANT DONE
