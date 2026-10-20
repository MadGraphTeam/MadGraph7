#!/bin/bash
#SBATCH --job-name=fz-gen
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ucl/cp3/omatt/fold-zjj/logs/gen-%j.out
# Generate p p > z j j [QCD], set the cards of the earlier Zjj benchmark, compile
# MC once, then copy one process directory per variant.
set -e
source /scratch/ucl/cp3/omatt/fold-zjj/env.sh
echo "node $(hostname)"; date
cd $MG7
printf "set automatic_html_opening False\ngenerate p p > z j j [QCD]\noutput $FZ/ZJJ\n" > $FZ/gen.mg5
python3 bin/madgraph $FZ/gen.mg5
C=$FZ/ZJJ/Cards/run_card.dat
sed -i -e 's/^ *[-0-9.]* *= jetalgo/ -1.0  = jetalgo/' -e 's/^ *[0-9.]* *= jetradius/ 0.4  = jetradius/' \
       -e 's/^ *[0-9.]* *= ptj /  30.0  = ptj /' -e 's/^ *[-0-9.]* *= req_acc / 0.0224 = req_acc /' \
       -e 's/^ *[0-9]* *= nevents / 10000 = nevents /' $C
grep -E "= *(jetalgo|jetradius|ptj|req_acc|nevents) " $C
cp $FZ/ZJJ/Cards/FKS_params.dat $FZ/ZJJ/Cards/FKS_params.dat.orig
cd $FZ/ZJJ
printf "set run_mode 2\nset nb_core 32\ncompile MC -f\nquit\n" > $FZ/compile.cmd
python3 bin/aMCatNLO $FZ/compile.cmd > $FZ/logs/compile.log 2>&1
grep -q "Traceback" $FZ/logs/compile.log && { echo "compile failed"; exit 1; }
for T in A J B L I K M D E C; do rm -rf $FZ/V_$T; cp -a $FZ/ZJJ $FZ/V_$T; done
date; echo GEN DONE
