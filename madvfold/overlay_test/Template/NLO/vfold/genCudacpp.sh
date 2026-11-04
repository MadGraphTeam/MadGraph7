#!/bin/bash
# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Oct 2026) for project MadVFold (arXiv:2609.23500).

# Create the CUDACPP subdirectory of this process directory, with the cudacpp
# process directories needed by mode C (as in madvfold/DEV/CUDACPP for the
# hand-written <proc>_vfalt):
# - one madevent_simd process directory for each line of vfold/cudacpp_spec.txt
#   (created by vfold/genCudacppSpec.sh if it does not exist yet), generated with
#   the cudacpp plugin installed in <mg5dir>/PLUGIN/CUDACPP_OUTPUT (see
#   madvfold/README_CUDACPP.md): the Born process with the --vfold output option,
#   the real-emission processes without it;
# - test, the googletest build directory used by 'make -f cudacpp.mk gtestlibs'
#   (found as ../../../test from SubProcesses/P1_<proc>), copied from
#   <mg5dir>/madvfold/DEV/CUDACPP/test (with googletest, if already built there).
# The ccache directory CUDACPP/ccache is created later by vfold/buildCudacppLibs.sh.
# Then build the libraries with vfold/buildCudacppLibs.sh and copy them to
# lib/cudacpp with vfold/copyCudacppLibs.sh.

set -e
set -o pipefail

cd $(dirname $0)/..
procdir=$(pwd -P)
if [ "$1" != "" ]; then
  echo "Usage: $0"
  exit 1
fi
mg5dir=$(cd ../../..; pwd -P) # as in vfold/myLaunch.sh (<mg5dir>/madvfold/DEV/<proc>)
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

# Create the test directory
mkdir CUDACPP
devdir=${mg5dir}/madvfold/DEV/CUDACPP
mkdir CUDACPP/test
command cp -dpr ${devdir}/test/makefile ${devdir}/test/include CUDACPP/test
if [ -d ${devdir}/test/googletest ]; then
  command cp -dpr ${devdir}/test/googletest CUDACPP/test
else
  echo "WARNING! No googletest in ${devdir}/test: it will be cloned by 'make -f cudacpp.mk gtestlibs'"
fi
command cp -dp ${devdir}/.gitignore CUDACPP

# Generate the cudacpp process directories
grep -v '^#' ${spec} | while read tag ccProc gencmd; do
  gencmd=$(echo ${gencmd} | sed "s/^'//; s/'$//")
  if [ "${tag}" == "born" ]; then vfoldopt=--vfold; else vfoldopt=; fi
  ccMad=${ccProc}.mad
  echo "Generate CUDACPP/${ccMad} (${tag}: '${gencmd}' ${vfoldopt})"
  mgfile=${procdir}/CUDACPP/${ccMad}.mg
  cat > ${mgfile} << EOF
set stdout_level DEBUG
set zerowidth_tchannel F
${gencmd}
output madevent_simd ${procdir}/CUDACPP/${ccMad} --hel_recycling=False --vector_size=32 ${vfoldopt}
EOF
  (cd ${mg5dir}; ./bin/mg5_aMC ${mgfile} > ${procdir}/CUDACPP/${ccMad}.log 2>&1) || { echo "ERROR! generation failed, see CUDACPP/${ccMad}.log"; exit 1; }
  # As in madgraph4gpu CODEGEN/generateAndCompare.sh: keep the generation commands
  # in mg5.in and make the directory relocatable (#400)
  sed "s|${procdir}/CUDACPP/||" ${mgfile} > CUDACPP/${ccMad}/mg5.in
  sed -i 's/mg5_path/#mg5_path/' CUDACPP/${ccMad}/Cards/me5_configuration.txt
  mv ${mgfile} CUDACPP/${ccMad}.log CUDACPP/${ccMad}/
done
echo "Generation of CUDACPP completed"
ls CUDACPP
