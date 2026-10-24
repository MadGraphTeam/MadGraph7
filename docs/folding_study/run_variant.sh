#!/bin/bash
# run_variant.sh <procdir> <tag> "<folding>" <nevents> <seed> [FKS_params lines, '\n' separated]
D=$1; TAG=$2; FOLD=$3; NEV=$4; SEED=$5; EXTRA=$6
S=/private/tmp/claude-503/-Users-omattelaer-Documents-git-workspace-MadGraph7--claude-worktrees-goofy-wu-e32d7b/51a9d264-83df-4654-835c-2dd2cb2b5abd/scratchpad/fold
cp $D/Cards/FKS_params.dat.orig $D/Cards/FKS_params.dat
[ -n "$EXTRA" ] && printf "$EXTRA\n" >> $D/Cards/FKS_params.dat
cat > $D/launch_$TAG.mg5 <<EOL
set nb_core ${NB:-18}
launch $D -n $TAG
shower=OFF
set nevents $NEV
set iseed $SEED
set folding $FOLD
done
EOL
cd /Users/omattelaer/Documents/git_workspace/MadGraph7/.claude/worktrees/goofy-wu-e32d7b
T0=$(date +%s)
~/.pyenv/versions/mg-3.14/bin/python bin/madgraph $D/launch_$TAG.mg5 > $D/launch_$TAG.log 2>&1
T1=$(date +%s)
mkdir -p $D/vlogs/$TAG
for f in $D/SubProcesses/P0_*/G*/log_MINT*.txt; do d=$(echo $f | sed "s|$D/SubProcesses/||; s|/|_|g"); cp $f $D/vlogs/$TAG/$d; done
python3 $S/summarise_variant.py $D $TAG $((T1-T0))
