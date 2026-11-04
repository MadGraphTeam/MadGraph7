#!/bin/bash
# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Nov 2026) for project MadVFold (arXiv:2609.23500).

# Generate and build the cudacpp libraries for mode C of this process directory,
# and copy them to lib/cudacpp. This is run by 'output <procdir> --vfold' (and
# --vfold-with-tests, which passes -tests) with the mg5amcnlo directory <mg5dir>:
# - vfold/genCudacpp.sh creates CUDACPP, with one standalone_cudacpp process
#   directory per cudacpp library (and, with -tests, the googletest directory
#   GTEST for the cudacpp unit tests), using the cudacpp plugin installed in
#   <mg5dir>/PLUGIN/CUDACPP_OUTPUT (see madvfold/README_CUDACPP.md);
# - vfold/buildCudacppLibs.sh builds the libraries in CUDACPP (only its progress
#   lines are shown, the full output is in CUDACPP/buildCudacppLibs.log);
# - vfold/copyCudacppLibs.sh copies them to lib/cudacpp.
# If the cudacpp plugin is not installed, there is no mode C (a warning only).
# With VFOLD_SETUP_NOCUDACPPBUILD=1 (madvfold/DEV/genProc.sh -nocudacppbuild),
# the cudacpp processes are only generated, the libraries are not built.

set -e
set -o pipefail

cd $(dirname $0)/..
testsopt=
if [ "$1" == "-tests" ]; then testsopt=-tests; shift; fi
if [ "$1" == "" ] || [ "$2" != "" ] || [ "${1:0:1}" == "-" ]; then
  echo "Usage: $0 [-tests] <mg5dir>"
  exit 1
fi
mg5dir=$(cd $1; pwd -P)

if [ ! -f ${mg5dir}/PLUGIN/CUDACPP_OUTPUT/__init__.py ]; then
  echo "WARNING! The cudacpp plugin is not installed in ${mg5dir}/PLUGIN/CUDACPP_OUTPUT: no cudacpp libraries for mode C (see madvfold/README_CUDACPP.md)"
  exit 0
fi
echo "INFO: generate CUDACPP (vfold/genCudacpp.sh)"
./vfold/genCudacpp.sh ${testsopt} ${mg5dir} > genCudacpp.log 2>&1 || { echo "ERROR! vfold/genCudacpp.sh failed, see genCudacpp.log"; exit 1; }
mv genCudacpp.log CUDACPP/
if [ "${VFOLD_SETUP_NOCUDACPPBUILD}" == "1" ]; then
  echo "INFO: cudacpp libraries not built (VFOLD_SETUP_NOCUDACPPBUILD=1): for mode C, run ./vfold/buildCudacppLibs.sh and ./vfold/copyCudacppLibs.sh"
  exit 0
fi
# Show the progress lines of the build (INFO:...), the full output is in the log
echo "INFO: build the cudacpp libraries (vfold/buildCudacppLibs.sh, log in CUDACPP/buildCudacppLibs.log)"
./vfold/buildCudacppLibs.sh 2>&1 | tee CUDACPP/buildCudacppLibs.log | { grep --line-buffered '^INFO:' || true; } || { echo "ERROR! vfold/buildCudacppLibs.sh failed, see CUDACPP/buildCudacppLibs.log"; exit 1; }
echo "INFO: copy the cudacpp libraries to lib/cudacpp (vfold/copyCudacppLibs.sh)"
./vfold/copyCudacppLibs.sh > CUDACPP/copyCudacppLibs.log 2>&1 || { echo "ERROR! vfold/copyCudacppLibs.sh failed, see CUDACPP/copyCudacppLibs.log"; exit 1; }
echo "INFO: generateCudacpp.sh completed (to run in mode C, first: source vfold/setupVfoldDefaults.sh)"
