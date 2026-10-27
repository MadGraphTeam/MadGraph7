#!/bin/bash
# run_dump.sh <Pdir> <griddir> <tag> <npoints> <seed> [extra FKS_params lines file]
P=$1; W=$2; T=$3; NP=$4; SEED=$5; EXTRA=$6
cd $P; rm -rf $T; mkdir -p $T; cd $T
for f in configs_and_props_info.dat leshouche_info.dat OLE_order.olc param_card.dat initial_states_map.dat symfact.dat; do ln -sf ../$f .; done
cp ../$W/mint_grids.M0 mint_grids; cp ../$W/grid.MC_integer .
cp ../FKS_params.dat .
if [ -n "$EXTRA" ]; then cat $EXTRA >> FKS_params.dat; cp ${FOLD_CFG} fold_dump.dat; else printf '#FoldDump\n1\n' >> FKS_params.dat; cp ${FOLD_CFG} fold_dump.dat; fi
echo $SEED > randinit
sed -e "1s/.*/$NP 1      ! points, iterations/" -e '8s/^[0-9] /1 /' -e "9s/.*/${FOLDING:-1 1 1}      ! folding/" ../$W/inM0.txt > in.txt
ulimit -s 65520
( time ../madevent_mintMC < in.txt > log.txt 2>&1 ) 2> time.txt
echo "$T exit $? $(awk '{print $1}' fold_dump.out 2>/dev/null | sort -u | wc -l) points $(grep real time.txt)"
