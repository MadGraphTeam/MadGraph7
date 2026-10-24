#!/bin/bash
# old code (T3_ee4b_v, before the merge of #305) and new code (T3_ee4b_g) side by side, 9 cores each
S=/private/tmp/claude-503/-Users-omattelaer-Documents-git-workspace-MadGraph7--claude-worktrees-goofy-wu-e32d7b/51a9d264-83df-4654-835c-2dd2cb2b5abd/scratchpad/fold
O=$S/T3_ee4b_v; N=$S/T3_ee4b_g; NEV=10000; SEED=21
RO=$S/results_T3_ee4b_v_s21.txt; RN=$S/results_T3_ee4b_g_s21.txt
old() { local c=$1 s=$2 x=""; [ $c = T ] && x='#FoldTransform\n2'; [ $s = T ] && x="$x${x:+\n}#FoldSumSets\n.true."; echo "$x"; }
new() { printf '#FoldCubic\n%s\n#FoldSumSets\n%s' $([ $1 = T ] && echo .true. || echo .false.) $([ $2 = T ] && echo .true. || echo .false.); }
pair() { # tag folding cubic sum
  echo "$1 start $(uptime)" >> $S/batch_pairs.load
  NB=9 $S/run_variant.sh $O p21$1 "$2" $NEV $SEED "$(old $3 $4)" >> $RO &
  NB=9 $S/run_variant.sh $N h21$1 "$2" $NEV $SEED "$(new $3 $4)" >> $RN &
  wait
}
pair A "1 1 1" F F
pair K "2 2 1" T T
pair J "1 1 1" F T
pair B "2 2 1" F F
pair L "2 2 1" F T
pair I "2 2 1" T F
pair D "4 4 1" T F
pair M "4 4 1" F T
pair E "4 4 1" T T
pair C "8 4 1" F F
echo "done $(uptime)" >> $S/batch_pairs.load
