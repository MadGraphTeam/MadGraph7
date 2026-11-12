#!/bin/bash
# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Aug 2026) for project MadVfold (arXiv:2609.23500).

set -e
set -o pipefail

cd $(dirname $0)/..
PROC=$(basename $(pwd))

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

# Rename+re-soname a group's own common.so to a name unique to that group, and
# repoint the group's matrix-element .so's own DT_NEEDED at the renamed name.
# Motivation: plain dlopen() resolves a same-named transitive dependency
# (DT_NEEDED) by a process-global name match, not by which directory asked for
# it: so if Born/Real1/Real2 all kept the same common.so basename, loading more
# than one of them in the same process would silently share whichever loaded
# first, even though each has its own (possibly different) copy on disk. This
# was the first issue identified in the real2/uux coupling-corruption bug).
# Giving each group's common.so a distinct name+soname lets plain dlopen() keep
# them genuinely separate, with no namespace tricks needed -- unlike
# dlmopen(LM_ID_NEWLM), which also isolates same-named dependencies but does so
# by creating a wholly separate linker namespace, which breaks the cuda backend
# (each namespace gets its own independent copy of anything cuda-related, and
# two such copies coexisting in one process is unsupported by NVIDIA: this was
# the second issue found during eebb_vfalt GPU debugging). Plain dlopen() by
# unique name has none of that risk: it is the same mechanism already proven
# CUDA-compatible for every single-Real-library process.
function renameCommonLib {
  local dir=$1
  local tag=$2
  local meLib=$3 # basename of this group's own matrix-element .so, e.g. libmg5amc_epem_bbxmummupg_cuda.so
  pushd ${dir} > /dev/null
  local oldCommonLib=$(basename $(ls libmg5amc_common_*.so))
  local newCommonLib=libmg5amc_cudacpp_common_${tag}.so
  mv ${oldCommonLib} ${newCommonLib}
  patchelf --set-soname ${newCommonLib} ${newCommonLib}
  patchelf --replace-needed ${oldCommonLib} ${newCommonLib} ${meLib}
  popd > /dev/null
}

# The cudacpp process directories are in CUDACPP (created by vfold/genCudacpp.sh)
# or else in ../CUDACPP (the layout of the hand-written madvfold/DEV/<proc>_vfalt)
if [ -d CUDACPP ]; then ccDir=CUDACPP; else ccDir=../CUDACPP; fi
echo "Copy the cudacpp libraries from $(cd ${ccDir}; pwd)"

# Copy cudacpp libraries
targetDir=lib/cudacpp
rm -rf ${targetDir}
ccBornSa=$(cd ${ccDir}; ccProcDir ${ccBornProc} vfold.sa)
ccBornDir=${ccDir}/${ccBornSa}/lib
for bld in ${ccBornDir}/build.*; do
  bld=$(basename ${bld})
  mkdir -p ${targetDir}/${bld}
  # Born
  tag=born
  mkdir ${targetDir}/${bld}/${tag}
  cp -dpr ${ccBornDir}/${bld}/libmg5amc_common_*.so ${targetDir}/${bld}/${tag}
  cp -dpr ${ccBornDir}/${bld}/libmg5amc_${ccBornProc}_*.so ${targetDir}/${bld}/${tag}
  pushd ${targetDir}/${bld}/${tag} > /dev/null
  ln -sf libmg5amc_${ccBornProc}_*.so libmg5amc_cudacpp_born.so
  popd > /dev/null
  renameCommonLib ${targetDir}/${bld}/${tag} ${tag} libmg5amc_cudacpp_born.so
  echo ${targetDir}/${bld}/${tag}
  ls -l ${targetDir}/${bld}/${tag}
  # Real
  for tag in ${ccRealTags}; do
    ccRealProc=$(ccProcForTag ${tag})
    ccRealSa=$(cd ${ccDir}; ccProcDir ${ccRealProc} sa)
    ccRealDir=${ccDir}/${ccRealSa}/lib
    mkdir ${targetDir}/${bld}/${tag}
    cp -dpr ${ccRealDir}/${bld}/libmg5amc_common_*.so ${targetDir}/${bld}/${tag}
    cp -dpr ${ccRealDir}/${bld}/libmg5amc_${ccRealProc}_*.so ${targetDir}/${bld}/${tag}
    pushd ${targetDir}/${bld}/${tag} > /dev/null
    ln -sf libmg5amc_${ccRealProc}_*.so libmg5amc_cudacpp_${tag}.so
    popd > /dev/null
    renameCommonLib ${targetDir}/${bld}/${tag} ${tag} libmg5amc_cudacpp_${tag}.so
    echo ${targetDir}/${bld}/${tag}
    ls -l ${targetDir}/${bld}/${tag}
  done
  echo
done
