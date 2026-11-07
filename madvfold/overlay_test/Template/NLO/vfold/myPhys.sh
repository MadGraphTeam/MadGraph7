#!/bin/bash
# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Oct 2026) for project MadVFold (arXiv:2609.23500).

cd $(dirname $0)/..

mkdir -p Events/PHYS
mkdir -p Events/PHYS_DELTA

export ENV_VFOLDTEST_PHYSICS=1
export ENV_VFOLDTEST_REQACC=-1
export ENV_VFOLDTEST_NEVENTS=100

step=step11t

#for folding in 0 1; do
for folding in 0; do

  if [ "${folding}" == "0" ]; then
    export ENV_VFOLDTEST_FOLDING="1,1,1"
    ###iseeds="11111 11112 11113"
    iseeds="11120 11121 11122 11123 11124 11125 11126 11127 11128 11129"
    kavx2=512; kcuda=512; vcuda=512
  else
    export ENV_VFOLDTEST_FOLDING="8,4,1"
    ###iseeds="84111 84112 84113"
    iseeds=
    kavx2=64; kcuda=32; vcuda=1024
  fi
    
  #for delta in 0 1; do
  for delta in 0; do

    xOpt=

    if [ "${delta}" == "0" ]; then  
      unset ENV_VFOLDTEST_MCATNLODELTA
      events=Events/PHYS
    else
      export ENV_VFOLDTEST_MCATNLODELTA=1
      events=Events/PHYS_DELTA
    fi
    
    for iseed in ${iseeds}; do

      export ENV_VFOLDTEST_ISEED=${iseed}

      #ENV_VFOLD_ME_MODE=F0 ./vfold/myLaunch.sh ${xOpt}
      #mv Events/run_01 ${events}/phys_${iseed}_${step}Z
      #xOpt="-x"

      #ENV_VFOLD_ME_MODE=F1 ./vfold/myLaunch.sh ${xOpt}
      #mv Events/run_01 ${events}/phys_${iseed}_${step}S
      #xOpt="-x"

      #ENV_VFOLD_KPOINT_BATCH=${kavx2} \
      #  ENV_VFOLD_SBORN_GSCALE=1 ENV_VFOLD_ME_MODE=C ENV_VFOLD_CUDACPP_BACKEND=avx2 ./vfold/myLaunch.sh ${xOpt}
      #mv Events/run_01 ${events}/phys_${iseed}_${step}Cavx2
      #xOpt="-x"

      ENV_VFOLD_KPOINT_BATCH=${kcuda} ENV_VFOLD_VECSIZE_USED=${vcuda} \
        ENV_VFOLD_SBORN_GSCALE=1 ENV_VFOLD_ME_MODE=C ENV_VFOLD_CUDACPP_BACKEND=cuda ./vfold/myLaunch.sh ${xOpt}
      mv Events/run_01 ${events}/phys_${iseed}_${step}Ccuda
      xOpt="-x"

    done

  done

done
