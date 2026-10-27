#!/bin/bash
# submit the variants after the generation job
FZ=/scratch/ucl/cp3/omatt/fold-zjj
GEN=$1
CUB='#FoldTransform\n2'
SUM='#FoldSumSets\n.true.'
sub() { sbatch --parsable --dependency=afterok:$GEN -J z$1 $FZ/variant.sbatch "$@"; }
sub A "1 1 1" ""
sub J "1 1 1" "$SUM"
sub B "2 2 1" ""
sub L "2 2 1" "$SUM"
sub I "2 2 1" "$CUB"
sub K "2 2 1" "$CUB\n$SUM"
sub M "4 4 1" "$SUM"
sub D "4 4 1" "$CUB"
sub E "4 4 1" "$CUB\n$SUM"
sub C "8 4 1" ""
