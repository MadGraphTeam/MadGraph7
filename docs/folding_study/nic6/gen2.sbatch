#!/bin/bash
#SBATCH --job-name=fz-gen2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=56G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ucl/cp3/omatt/fold-zjj/logs/gen2-%j.out
# Same as gen.sbatch, with the code after the merge of #305 (MG7new), into ZJJ2 and W_<tag>
set -e
source /scratch/ucl/cp3/omatt/fold-zjj/env.sh
MG7N=$FZ/MG7new
echo "node $(hostname)"; date
cd $MG7N
printf "set automatic_html_opening False\ngenerate p p > z j j [QCD]\noutput $FZ/ZJJ2\n" > $FZ/gen2.mg5
python3 bin/madgraph $FZ/gen2.mg5
C=$FZ/ZJJ2/Cards/run_card.dat
sed -i -e 's/^ *[-0-9.]* *= jetalgo/ -1.0  = jetalgo/' -e 's/^ *[0-9.]* *= jetradius/ 0.4  = jetradius/' \
       -e 's/^ *[0-9.]* *= ptj /  30.0  = ptj /' -e 's/^ *[-0-9.]* *= req_acc / 0.0224 = req_acc /' \
       -e 's/^ *[0-9]* *= nevents / 10000 = nevents /' $C
grep -E "(jetalgo|jetradius|ptj|req_acc|nevents) " $C | head -5
grep -c -i gscale $FZ/ZJJ2/SubProcesses/P0_*/born.f | head -2
cp $FZ/ZJJ2/Cards/FKS_params.dat $FZ/ZJJ2/Cards/FKS_params.dat.orig
cd $FZ/ZJJ2
printf "set run_mode 2\nset nb_core 32\ncompile MC -f\nquit\n" > $FZ/compile2.cmd
python3 bin/aMCatNLO $FZ/compile2.cmd > $FZ/logs/compile2.log 2>&1
grep -q "Traceback" $FZ/logs/compile2.log && { echo "compile failed"; exit 1; }
for T in A J B L I K M D E C; do rm -rf $FZ/W_$T; cp -a $FZ/ZJJ2 $FZ/W_$T; done
date; echo GEN DONE
