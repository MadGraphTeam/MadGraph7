#!/bin/bash
# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Aug 2026) for project MadVFold (arXiv:2609.23500).

###set -e
set -o pipefail

cd $(dirname $0)/..
PROC=$(basename $(pwd))

# Command line arguments
usage() {
  echo "Usage:   $0 [--diffonly] [--refbase <refbase>] [--diffmodes '<mode1 mode2...>'] [--modes '<mode1 mode2...>'] <step>"
  echo "Usage:   $0 [--nodiff] [--modes '<mode1 mode2...>'] <step>"
  echo "Example: $0 --modes 'F0 F C' --refbase step2 step3a"
  exit 1
}
refbase=
diffonly=0
diffmodes=
modes=
step=
while [ "$1" != "" ]; do
  if [ "$1" == "--refbase" ] && [ "$2" != "" ] && [ "$refbase" == "" ] && [ "$diffonly" != "-1" ]; then
    refbase=$2; shift; shift
  elif [ "$1" == "--nodiff" ] && [ "$diffonly" == "0" ] && [ "$refbase" == "" ]; then
    diffonly=-1; shift
  elif [ "$1" == "--diffonly" ] && [ "$diffonly" == "0" ]; then
    diffonly=1; shift
  elif [ "$1" == "--diffmodes" ] && [ "$2" != "" ] && [ "$diffmodes" == "" ]; then
    diffmodes=$2; shift; shift
  elif [ "$1" == "--modes" ] && [ "$2" != "" ] && [ "$modes" == "" ]; then
    modes=$2; shift; shift
  elif [ "$step" == "" ]; then
    step=$1; shift
  else
    usage
  fi
done
if [ "${step}" == "" ]; then
  usage
elif [ "${step}" == "${step#step}" ] || [ "${step}" == "step" ]; then
  echo "ERROR! Invalid step=${step} must be of the form step<id>"
  exit 1
fi
if [ "${refbase}" == "" ]; then refbase=step1; fi
if [ "${diffmodes}" == "" ]; then diffmodes=${modes}; fi
echo refbase="$refbase"
echo diffonly="$diffonly"
echo diffmodes="$diffmodes"
echo modes="$modes"
echo step="$step"
###exit 0

# Road B step 9a+9b+9c: EVENT (kpoint) batching
# Use ENV_VFOLD_KPOINT_BATCH=32 as of step 9c to test MINT1 and MINT2 batching
if [ "${ENV_VFOLD_KPOINT_BATCH}" == "" ]; then export ENV_VFOLD_KPOINT_BATCH=32; fi

# Road B step 10b: GSCALE optimization
if [ "${ENV_VFOLD_SBORN_GSCALE}" == "" ]; then export ENV_VFOLD_SBORN_GSCALE=1; fi

# MC@NLO-Delta matching (ENV_MCATNLODELTA=1, see myLaunch.sh): outputs and references are in Events/DELTA
if [ "${ENV_MCATNLODELTA}" == "1" ]; then evdir=Events/DELTA; mkdir -p ${evdir}; else evdir=Events; fi
echo evdir="$evdir"

# Check that no Events/run* exist
if [ "$(ls -d1 Events/run* 2>/dev/null | grep -v run_gen)" != "" ]; then
  echo "ERROR! Events/run* directories exist"
  ls -ld Events/run* | grep -v run_gen
  exit 1
fi

# LOOP OVER MODES (1: LAUNCH)
xOpt=
suffixes=
for mode in $modes; do
  ENV_VFOLD_ME_MODE=$mode
  unset CUDACPP_RUNTIME_BLASCOLORSUM
  # Validate mode and get suffix
  if [ "${ENV_VFOLD_ME_MODE}" == "F0" ]; then
    suffix=Z
  elif [ "${ENV_VFOLD_ME_MODE}" == "F1" ]; then
    suffix=S
  elif [ "${ENV_VFOLD_ME_MODE}" == "F2" ]; then
    suffix=T
  elif [ "${ENV_VFOLD_ME_MODE}" == "F" ]; then
    suffix=F
  elif [ "${ENV_VFOLD_ME_MODE}" == "C" ]; then
    if [ "${ENV_VFOLD_CUDACPP_BACKEND}" == "" ]; then
      ENV_VFOLD_CUDACPP_BACKEND=avx2
    elif [ "${ENV_VFOLD_CUDACPP_BACKEND}" != "none" ] && [ "${ENV_VFOLD_CUDACPP_BACKEND}" != "sse4" ] && [ "${ENV_VFOLD_CUDACPP_BACKEND}" != "avx2" ] && [ "${ENV_VFOLD_CUDACPP_BACKEND}" != "512y" ] && [ "${ENV_VFOLD_CUDACPP_BACKEND}" != "512z" ] && [ "${ENV_VFOLD_CUDACPP_BACKEND}" != "cuda" ]; then
      echo "WARNING! Skip invalid ENV_VFOLD_CUDACPP_BACKEND=${ENV_VFOLD_CUDACPP_BACKEND}"
      continue
    fi
    suffix=C${ENV_VFOLD_CUDACPP_BACKEND}
  elif [ "${ENV_VFOLD_ME_MODE}" == "Ccuda" ]; then
    ENV_VFOLD_ME_MODE=C
    ENV_VFOLD_CUDACPP_BACKEND=cuda
    suffix=C${ENV_VFOLD_CUDACPP_BACKEND}
  elif [ "${ENV_VFOLD_ME_MODE}" == "Ccudb" ]; then
    ENV_VFOLD_ME_MODE=C
    ENV_VFOLD_CUDACPP_BACKEND=cuda
    suffix=Ccudb
    export CUDACPP_RUNTIME_BLASCOLORSUM=1
  else
    echo "WARNING! Skip invalid ENV_VFOLD_ME_MODE=${ENV_VFOLD_ME_MODE}"
    continue
  fi
  events=${evdir}/${step}${suffix}
  for diffmode in $diffmodes; do
    if [ "${diffmode}" == "${mode}" ]; then
      suffixes="${suffixes} ${suffix}"
      break
    fi
  done
  # Launch test
  if [ "$diffonly" != "1" ]; then
    if [ "$(ls -d1 ${events} 2>/dev/null)" != "" ]; then
      echo "ERROR! Directory ${events} already exists"
      exit 1
    fi
    ENV_VFOLD_ME_MODE=${ENV_VFOLD_ME_MODE} ENV_VFOLD_CUDACPP_BACKEND=${ENV_VFOLD_CUDACPP_BACKEND} ./vfold/myLaunch.sh ${xOpt}
    mv Events/run_01 ${events}
  fi
  # Build the code only in the first test then use the existing binaries
  xOpt=-x
done

# LOOP OVER MODES (2: DIFF)
echo
for suffix in $suffixes; do
  if [ "$diffonly" != "-1" ]; then
    # Check that reference directory exists
    eventsref=${evdir}/${refbase}${suffix}
    if [ "$(ls -d1 ${eventsref} 2>/dev/null)" == "" ]; then
      echo "ERROR! Reference directory ${eventsref} does not exist"
      exit 1
    fi
    events=${evdir}/${step}${suffix}
    # Compare to ref
    rm -rf ${events}/log_diff.txt
    ./vfold/myDiffRun.sh -linkDumps ${eventsref} ${events} | tee ${events}/log_diff.txt
  fi
done
