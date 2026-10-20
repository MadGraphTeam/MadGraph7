#!/bin/bash
# prep_dir.sh <Pdir> <GFdir> <name> <mint0 points> : copy channel dir, train MINT0 grids
P=$1; G=$2; N=$3; NP=$4
cd $P; rm -rf $N; mkdir $N; cd $N
for f in configs_and_props_info.dat leshouche_info.dat OLE_order.olc param_card.dat initial_states_map.dat symfact.dat randinit; do ln -sf ../$f .; done
cp ../FKS_params.dat FKS_params.dat
cp ../$G/input_app.txt in_base.txt
sed -e "1s/.*/$NP 6      ! points, iterations/" -e '8s/^[0-9] /0 /' -e '9s/.*/1 1 1      ! folding/' -e '10s/.*/all/' in_base.txt > inM0.txt
../madevent_mintMC < inM0.txt > logM0.txt 2>&1
cp mint_grids mint_grids.M0
grep "Final result" logM0.txt
