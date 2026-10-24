#!/bin/bash
FZ=/scratch/ucl/cp3/omatt/fold-zjj
GEN=$1
o() { local x=""; [ $1 = T ] && x='#FoldTransform\n2'; [ $2 = T ] && x="$x${x:+\n}#FoldSumSets\n.true."; echo "$x"; }
n() { printf '#FoldCubic\\n%s\\n#FoldSumSets\\n%s' $([ $1 = T ] && echo .true. || echo .false.) $([ $2 = T ] && echo .true. || echo .false.); }
sub() { sbatch --parsable --dependency=afterok:$GEN -J p$1 $FZ/pair.sbatch $1 "$2" "$(o $3 $4)" "$(n $3 $4)"; }
sub A "1 1 1" F F
sub J "1 1 1" F T
sub B "2 2 1" F F
sub L "2 2 1" F T
sub I "2 2 1" T F
sub K "2 2 1" T T
sub M "4 4 1" F T
sub D "4 4 1" T F
sub E "4 4 1" T T
sub C "8 4 1" F F
