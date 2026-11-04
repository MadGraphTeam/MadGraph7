#!/bin/bash
# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Oct 2026) for project MadVFold (arXiv:2609.23500).

# Create the CUDACPP subdirectory of this process directory, with the cudacpp
# process directories needed by mode C (as in madvfold/DEV/CUDACPP for the
# hand-written <proc>_vfalt):
# - one standalone_cudacpp process directory for each line of vfold/cudacpp_spec.txt
#   (created by vfold/genCudacppSpec.sh if it does not exist yet), generated with
#   the cudacpp plugin installed in <mg5dir>/PLUGIN/CUDACPP_OUTPUT (see
#   madvfold/README_CUDACPP.md): the Born process <ccproc>.vfold.sa with the --vfold
#   output option (which also adds the multichannel support needed for the Born),
#   the real-emission processes <ccproc>.sa without it;
# - only with -tests (for 'output --vfold-with-tests'): GTEST, the googletest
#   installation shared by the cudacpp unit tests of all these process directories
#   (their <ccproc>/test/GTEST is created as a symlink to it by 'make -f cudacpp.mk
#   gtestlibs'): googletest is cloned and built there by the first build, unless a
#   googletest build is found in <mg5dir>/madvfold/DEV/CUDACPP (GTEST/googletest
#   or test/googletest) and copied. Without GTEST, buildCudacppLibs.sh builds
#   the libraries without the cudacpp unit tests.
# The mg5amcnlo directory <mg5dir> is that of 'output --vfold' (passed by
# vfold/generateCudacpp.sh), by default ../../.. (<mg5dir>/madvfold/DEV/<proc>).
# The ccache directory CUDACPP/ccache is created later by vfold/buildCudacppLibs.sh.
# Then build the libraries with vfold/buildCudacppLibs.sh and copy them to
# lib/cudacpp with vfold/copyCudacppLibs.sh (vfold/generateCudacpp.sh does all three).

set -e
set -o pipefail

cd $(dirname $0)/..
procdir=$(pwd -P)
tests=0
if [ "$1" == "-tests" ]; then tests=1; shift; fi
if [ "$2" != "" ] || [ "${1:0:1}" == "-" ]; then
  echo "Usage: $0 [-tests] [<mg5dir>]"
  exit 1
fi
if [ "$1" != "" ]; then
  mg5dir=$(cd $1; pwd -P)
else
  mg5dir=$(cd ../../..; pwd -P) # as in vfold/myLaunch.sh (<mg5dir>/madvfold/DEV/<proc>)
fi
if [ ! -f ${mg5dir}/PLUGIN/CUDACPP_OUTPUT/__init__.py ]; then
  echo "ERROR! The cudacpp plugin is not installed in ${mg5dir}/PLUGIN/CUDACPP_OUTPUT (see madvfold/README_CUDACPP.md)"
  exit 1
fi
if [ -e CUDACPP ]; then
  echo "ERROR! ${procdir}/CUDACPP already exists"
  exit 1
fi

# Read the cudacpp processes for this directory from cudacpp_spec.txt
spec=${procdir}/vfold/cudacpp_spec.txt
if [ ! -f ${spec} ]; then ./vfold/genCudacppSpec.sh; fi
ccBornProc=$(awk '$1=="born"{print $2}' ${spec})
if [ "${ccBornProc}" == "" ]; then
  echo "ERROR! Invalid ${spec}"
  exit 1
fi

# Create the CUDACPP directory, and the GTEST directory (only with -tests)
mkdir CUDACPP
devdir=${mg5dir}/madvfold/DEV/CUDACPP
if [ "${tests}" == "1" ]; then
  mkdir CUDACPP/GTEST
  echo "Shared googletest installation for the cudacpp process directories in CUDACPP" > CUDACPP/GTEST/.keepme
  if [ -d ${devdir}/GTEST/googletest ]; then
    command cp -dpr ${devdir}/GTEST/googletest CUDACPP/GTEST
  elif [ -d ${devdir}/test/googletest ]; then
    command cp -dpr ${devdir}/test/googletest CUDACPP/GTEST
  else
    echo "WARNING! No googletest in ${devdir}: it will be cloned and built by 'make -f cudacpp.mk gtestlibs' (needs git and cmake)"
  fi
fi
if [ -f ${devdir}/.gitignore ]; then command cp -dp ${devdir}/.gitignore CUDACPP; fi

# Generate the cudacpp process directories
grep -v '^#' ${spec} | while read tag ccProc gencmd; do
  gencmd=$(echo ${gencmd} | sed "s/^'//; s/'$//")
  if [ "${tag}" == "born" ]; then vfoldopt=--vfold; ccSa=${ccProc}.vfold.sa; else vfoldopt=; ccSa=${ccProc}.sa; fi
  echo "Generate CUDACPP/${ccSa} (${tag}: '${gencmd}' ${vfoldopt})"
  mgfile=${procdir}/CUDACPP/${ccSa}.mg
  cat > ${mgfile} << EOF
set stdout_level DEBUG
set zerowidth_tchannel F
${gencmd}
output standalone_cudacpp ${procdir}/CUDACPP/${ccSa} ${vfoldopt}
EOF
  (cd ${mg5dir}; ./bin/madgraph ${mgfile} > ${procdir}/CUDACPP/${ccSa}.log 2>&1) || { echo "ERROR! generation failed, see CUDACPP/${ccSa}.log"; exit 1; }
  # As in madgraph4gpu CODEGEN/generateAndCompare.sh: keep the generation commands in mg5.in
  sed "s|${procdir}/CUDACPP/||" ${mgfile} > CUDACPP/${ccSa}/mg5.in
  mv ${mgfile} CUDACPP/${ccSa}.log CUDACPP/${ccSa}/
done
echo "Generation of CUDACPP completed"
ls CUDACPP
