#!/bin/bash
# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Aug 2026) for project MadVfold (arXiv:2609.23500).

set -e
set -o pipefail

cd $(dirname $0)/..
PROC=$(basename $(pwd))

force=0
if [ "$1" == "-ff" ]; then
  force=2; shift
elif [ "$1" == "-f" ]; then
  force=1
elif [ "$1" != "" ]; then
  echo "Usage: $0 [-f|-ff]"
  exit 1
fi

# Read the cudacpp processes for this directory from cudacpp_spec.txt (see genCudacppSpec.sh)
# NB! The Born library needs multichannel support: .vfold.sa (standalone with --vfold) or .mad, not plain .sa
spec=$(pwd)/vfold/cudacpp_spec.txt
if [ ! -f ${spec} ]; then
  echo "ERROR! ${spec} not found (create it using ./vfold/genCudacppSpec.sh)"
  exit 1
fi
ccBornProc=$(awk '$1=="born"{print $2}' ${spec})
ccRealTags=$(awk '$1~/^real[0-9]+$/{print $1}' ${spec})
if [ "${ccBornProc}" == "" ] || [ "${ccRealTags}" == "" ]; then
  echo "ERROR! Invalid ${spec}"
  exit 1
fi
# Print the cudacpp process for a given tag (e.g. real1 -> epem_bbxg)
function ccProcForTag() {
  awk -v tag=$1 '$1==tag{print $2}' ${spec}
}
# The cudacpp process directory for a cudacpp process (relative to the directory with
# the cudacpp process directories): <ccproc>.vfold.sa for the Born (standalone_cudacpp
# with --vfold) or <ccproc>.sa for the Real processes (standalone_cudacpp), as created
# by vfold/genCudacpp.sh, or else <ccproc>.mad (madevent_simd, the layout of
# madvfold/DEV/CUDACPP for the hand-written madvfold/DEV/<proc>_vfalt)
function ccProcDir() {
  if [ -d $1.$2 ]; then echo $1.$2; else echo $1.mad; fi
}

# Set up build configuration
# The cudacpp process directories are in CUDACPP (created by vfold/genCudacpp.sh)
# or else in ../CUDACPP (the layout of the hand-written madvfold/DEV/<proc>_vfalt)
if [ -d CUDACPP ]; then cd CUDACPP; else cd ../CUDACPP; fi
echo "Build the cudacpp libraries in $(pwd)"
# Use ccache only if it is installed
if which ccache > /dev/null 2>&1; then export USECCACHE=1; else export USECCACHE=0; fi
echo USECCACHE=${USECCACHE}
# Use the ccache directory ccache next to the cudacpp process directories:
# if it does not exist, create it as a symlink to ../../ccache if that exists
# (i.e. <mg5dir>/madvfold/DEV/ccache, shared by all process directories in
# madvfold/DEV, so that the cache survives the regeneration of a process
# directory), or else as a local directory
if [ ! -e ccache ] && [ ! -L ccache ]; then
  if [ -d ../../ccache ]; then
    ln -sn ../../ccache ccache
  else
    mkdir ccache
  fi
fi
export CCACHE_DIR=$(cd ccache; pwd -P)
echo CCACHE_DIR=${CCACHE_DIR}
export FC=gfortran
export FPTYPE=d
export HASCURAND=hasNoCurand
# Build the cudacpp unit tests only if there is a googletest directory (GTEST,
# created by 'vfold/genCudacpp.sh -tests', or test for the old .mad layout)
if [ -d GTEST ] || [ -d test ]; then usegtest=1; else usegtest=0; export USEGTEST=0; fi
echo USEGTEST=${usegtest}

# Set up f90cache (disabled to avoid any module-related issues)
unset F90CACHE_DIR
unset F90CACHE_LOGFILE
#pushd ../../../ > /dev/null
#if [ -x madvfold/DEV/F90CACHE/bin/f90cache ]; then
#  export PATH=$(cd madvfold/DEV/F90CACHE/bin; pwd):${PATH}
#  # Use a username-specific cache to avoid interferences
#  # Use a process-specific cache to avoid interferences (f90cache uses relative paths?)
#  if [ "$(whoami)" == "" ]; then echo "ERROR! whoami?"; exit 1; fi
#  export F90CACHE_DIR=$(cd madvfold/DEV/F90CACHE/f90cache; pwd)/$(whoami)/${PROC}
#  mkdir -p ${F90CACHE_DIR}
#  if [ "${FC}" != "" ]; then
#    export FC="f90cache ${FC}"
#  else
#    export FC="f90cache gfortran"
#  fi
#  export F90CACHE_LOGFILE=$(cd madvfold/DEV/${PROC}; pwd)/f90cache_cudacpp.log
#fi
#popd > /dev/null

# Set up CUDA
cuda_home=/usr/local/cuda-12.0
if [ -d $cuda_home ]; then
  export CUDA_HOME=$(cd $cuda_home; pwd -P)
  export PATH=${CUDA_HOME}/bin:${PATH}
fi

# Build cudacpp Born libraries
ccBornSa=$(ccProcDir ${ccBornProc} vfold.sa)
echo "INFO:   build the Born library ${ccBornSa} (all backends) at $(date +%T)"
pushd $(ls -d ${ccBornSa}/SubProcesses/P1_*) > /dev/null # P1_<ccproc> (.mad) or P1_Sigma_<model>_<ccproc> (.vfold.sa)
if [ "$force" == "2" ]; then
  make -f cudacpp.mk distclean
elif [ "$force" == "1" ]; then
  make -f cudacpp.mk cleanall
fi
if [ "${usegtest}" == "1" ]; then make -f cudacpp.mk gtestlibs; fi
nj=$(( $(nproc)-1 ))
make -f cudacpp.mk -j${nj} bldall
popd > /dev/null

# Build cudacpp Real libraries
for tag in ${ccRealTags}; do
  ccRealProc=$(ccProcForTag ${tag})
  ccRealSa=$(ccProcDir ${ccRealProc} sa)
  echo "INFO:   build the Real library ${ccRealSa} (${tag}, all backends) at $(date +%T)"
  pushd $(ls -d ${ccRealSa}/SubProcesses/P1_*) > /dev/null # P1_<ccproc> (.mad) or P1_Sigma_<model>_<ccproc> (.sa)
  if [ "$force" == "1" ]; then
    make -f cudacpp.mk cleanall
  fi
  make -f cudacpp.mk -j${nj} bldall
  popd > /dev/null
done

echo "INFO:   cudacpp libraries built at $(date +%T)"

# Sanity check
echo FC=${FC}
echo F90CACHE_DIR=${F90CACHE_DIR}
echo F90CACHE_LOGFILE=${F90CACHE_LOGFILE}
