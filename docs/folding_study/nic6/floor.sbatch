#!/bin/bash
#SBATCH --job-name=fz-floor
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=56G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/ucl/cp3/omatt/fold-zjj/logs/floor-%x-%j.out
# usage: sbatch -J f<P> floor.sbatch <P>   P = tt | eej | ee4b | zjj
# 8,4,1 folding with FoldCubic and FoldSumSets (the defaults of MG7new), seed 21.
set -u
source /scratch/ucl/cp3/omatt/fold-zjj/env.sh
P=$1
D=$FZ/F_$P
echo "floor $P on $(hostname), job $SLURM_JOB_ID"; lscpu | grep "Model name"; uptime; date
case $P in
  tt)   MODEL="import model loop_sm-no_b_mass"; PROC="p p > t t~ [QCD]"; NEV=20000 ;;
  eej)  MODEL="import model loop_sm-no_b_mass"; PROC="p p > e+ e- j [QCD]"; NEV=10000 ;;
  ee4b) MODEL=""; PROC="e+ e- > b b~ b b~ [QCD]"; NEV=10000 ;;
  zjj)  MODEL=""; PROC="" ;;
esac
if [ $P = zjj ]; then
  rm -rf $D; cp -a $FZ/ZJJ2 $D
else
  rm -rf $D
  cd $FZ/MG7new
  printf "set automatic_html_opening False\n$MODEL\ngenerate $PROC\noutput $D\n" > $FZ/gen_$P.mg5
  python3 bin/madgraph $FZ/gen_$P.mg5 > $FZ/logs/gen_$P.log 2>&1
  sed -i -e "s/^ *[0-9]* *= nevents / $NEV = nevents /" $D/Cards/run_card.dat
fi
grep -A1 "#Fold" $D/Cards/FKS_params.dat
grep -E "(nevents|req_acc|jetalgo|ptj) " $D/Cards/run_card.dat | head -4
printf "set run_mode 2\nset nb_core 32\nlaunch -n f21C\nfixed_order=OFF\nshower=OFF\nmadspin=OFF\nreweight=OFF\ndone\nset iseed 21\nset folding 8 4 1\ndone\nquit\n" > $D/launch_f21C.cmd
cd $D
T0=$(date +%s)
python3 bin/aMCatNLO $D/launch_f21C.cmd > $FZ/logs/launch_floor_$P.log 2>&1
echo "wall $(( $(date +%s)-T0 )) s"
mkdir -p $D/vlogs/f21C
for f in $D/SubProcesses/P0_*/G*/log_MINT*.txt; do d=$(echo $f | sed "s|$D/SubProcesses/||; s|/|_|g"); cp $f $D/vlogs/f21C/$d; done
grep "Total cross section" $FZ/logs/launch_floor_$P.log | tail -1
uptime; date; echo FLOOR DONE
