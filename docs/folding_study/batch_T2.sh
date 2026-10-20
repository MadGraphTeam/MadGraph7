#!/bin/bash
S=/private/tmp/claude-503/-Users-omattelaer-Documents-git-workspace-MadGraph7--claude-worktrees-goofy-wu-e32d7b/51a9d264-83df-4654-835c-2dd2cb2b5abd/scratchpad/fold
D=$S/$1; NEV=$2; SEED=$3; P=$4
R=$S/results_$1_$P.txt
CUB='#FoldTransform\n2'
SUM='#FoldSumSets\n.true.'
ONE='#FoldRealOnce\n.true.'
$S/run_variant.sh $D ${P}A "1 1 1" $NEV $SEED >> $R
$S/run_variant.sh $D ${P}B "2 2 1" $NEV $SEED >> $R
$S/run_variant.sh $D ${P}C "8 4 1" $NEV $SEED >> $R
$S/run_variant.sh $D ${P}D "4 4 1" $NEV $SEED "$CUB" >> $R
$S/run_variant.sh $D ${P}E "4 4 1" $NEV $SEED "$CUB\n$SUM" >> $R
$S/run_variant.sh $D ${P}F "4 4 1" $NEV $SEED "$CUB\n$SUM\n$ONE" >> $R
$S/run_variant.sh $D ${P}G "2 2 1" $NEV $SEED "$CUB\n$SUM\n$ONE" >> $R
$S/run_variant.sh $D ${P}H "8 4 1" $NEV $SEED "$ONE" >> $R
$S/run_variant.sh $D ${P}I "2 2 1" $NEV $SEED "$CUB" >> $R
$S/run_variant.sh $D ${P}J "1 1 1" $NEV $SEED "$SUM" >> $R
