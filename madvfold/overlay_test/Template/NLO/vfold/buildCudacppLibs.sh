#!/bin/bash
# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Aug 2026) for project MadVFold (arXiv:2609.23500).

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
# NB! Need .mad libraries because .sa libraries are built without multichannel support
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

# Set up build configuration
cd ../CUDACPP
export USECCACHE=1
export CCACHE_DIR=$(pwd -P)/ccache
echo CCACHE_DIR=${CCACHE_DIR}
export FC=gfortran
export FPTYPE=d
export HASCURAND=hasNoCurand

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
ccBornMad=${ccBornProc}.mad
pushd ${ccBornMad}/SubProcesses/P1_${ccBornProc} > /dev/null
if [ "$force" == "2" ]; then
  make -f cudacpp.mk distclean
elif [ "$force" == "1" ]; then
  make -f cudacpp.mk cleanall
fi
make -f cudacpp.mk gtestlibs
nj=$(( $(nproc)-1 ))
make -f cudacpp.mk -j${nj} bldall
popd > /dev/null

# Build cudacpp Real libraries
for tag in ${ccRealTags}; do
  ccRealProc=$(ccProcForTag ${tag})
  ccRealMad=${ccRealProc}.mad
  pushd ${ccRealMad}/SubProcesses/P1_${ccRealProc} > /dev/null
  if [ "$force" == "1" ]; then
    make -f cudacpp.mk cleanall
  fi
  make -f cudacpp.mk -j${nj} bldall
  popd > /dev/null
done

# Sanity check
echo FC=${FC}
echo F90CACHE_DIR=${F90CACHE_DIR}
echo F90CACHE_LOGFILE=${F90CACHE_LOGFILE}
