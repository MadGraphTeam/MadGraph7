#!/bin/bash
# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Sep 2026) for project MadVFold (arXiv:2609.23500).

###set -e
###set -o pipefail

cd $(dirname $0)/..
PROC=$(basename $(pwd))

# Default: build the code only in the first F0 test then use the existing binaries
xOpt=-x

# Command line arguments
ARGS="$*"
mode=
oOpt=
step=
refstep=
cudascan=0
ninjaAndCollier=0
noopt7=0
onlyNoopt7=0
usage() {
  echo "Usage: $0 [--ninja-and-collier] [--noopt7|--only-noopt7] <--launch|--launch-o|--launch-o-cudascan> <step>"
  echo "       (--ninja-and-collier: also run F1 with the other MadLoop reduction library)"
  echo "       (--noopt7: also run F1 without the stock MadGraph7 Born cache and gscale, in <step>Su)"
  echo "       (--only-noopt7: only add <step>Su to the existing runs of <step>)"
  echo "Usage: $0 <--diff|--diff-o> <step>"
  echo "Usage: $0 <--diff2|--diff2-o> <refstep> <step>"
  exit 1
}
while [ "$1" != "" ]; do
  if [ "$1" == "--ninja-and-collier" ]; then
    ninjaAndCollier=1; shift
  elif [ "$1" == "--noopt7" ]; then
    noopt7=1; shift
  elif [ "$1" == "--only-noopt7" ]; then
    onlyNoopt7=1; shift
  elif [ "$mode" == "" ] && [ "$1" == "--launch" ]; then
    mode=$1; shift
  elif [ "$mode" == "" ] && [ "$1" == "--launch-o" ]; then
    mode=--launch; oOpt=-o; shift
  elif [ "$mode" == "" ] && [ "$1" == "--launch-o-cudascan" ]; then
    mode=--launch; oOpt=-o; cudascan=1; shift
  elif [ "$mode" == "" ] && [ "$1" == "--diff" ]; then
    mode=$1; shift
  elif [ "$mode" == "" ] && [ "$1" == "--diff-o" ]; then
    mode=--diff; oOpt=-o; shift
  elif [ "$mode" == "" ] && [ "$1" == "--diff2" ]; then
    mode=$1; refstep=$2; shift; shift
  elif [ "$mode" == "" ] && [ "$1" == "--diff2-o" ]; then
    mode=--diff2; oOpt=-o; refstep=$2; shift; shift
  elif [ "$step" == "" ]; then
    step=$1; shift
  else
    usage
  fi
done
if [ "$mode" == "" ] || [ "$step" == "" ]; then
  echo mode=${mode}
  echo step=${step}
  usage
fi
if [ "$step" == "$refstep" ]; then
  echo step=${step}
  echo refstep=${step}
  usage
fi
if [ "$noopt7" == "1" ] && [ "$onlyNoopt7" == "1" ]; then usage; fi
if [ "$noopt7$onlyNoopt7" != "00" ] && [ "$mode" != "--launch" ]; then usage; fi
###echo mode=${mode}
###echo step=${step}

# Launch or diff steps
if [ "$mode" == "--launch" ]; then

  events=Events/${step}
  if [ "$onlyNoopt7" == "1" ]; then
    # Step 15e: add the F1 run without the MadGraph7 Born reuse to the existing runs of this step
    if [ ! -d ${events}S ]; then
      echo "ERROR! ${events}S does not exist: --only-noopt7 adds ${events}Su to an existing test"
      exit 1
    fi
    if [ -e ${events}Su ]; then
      echo "ERROR! ${events}Su exists"
      exit 1
    fi
  elif [ "$(ls -d1 ${events}* 2>/dev/null | grep -v '.txt$')" != "" ]; then
    echo "ERROR! ${events}* directories exist"
    ls -d1 ${events}* | grep -v '.txt$'
    exit 1
  fi
  mkdir -p $(dirname ${events})

  if [ "$(ls -d1 Events/run* 2>/dev/null | grep -v run_gen)" != "" ]; then
    echo "ERROR! Events/run* directories exist"
    ls -ld Events/run* | grep -v run_gen
    exit 1
  fi

  set -e
  ###set -x

  # Step 15e: with --only-noopt7, this is the only run (without -x: this is also the only build)
  if [ "$onlyNoopt7" == "1" ]; then
    ENV_VFOLD_MG7BORNNOOPT=1 ENV_VFOLD_ME_MODE=F1 ./vfold/myLaunch.sh ${oOpt}
    mv Events/run_01 ${events}Su
    exit 0
  fi

  ENV_VFOLD_ME_MODE=F0 ./vfold/myLaunch.sh ${oOpt}
  mv Events/run_01 ${events}Z

  ENV_VFOLD_ME_MODE=F1 ./vfold/myLaunch.sh ${oOpt} ${xOpt}
  mv Events/run_01 ${events}S

  # Step 14h: also F1 with the other MadLoop reduction library, for the timing table only
  # (in ${events}Sn or ${events}Sc: F1[n] and F1[c], the results are not expected to be the same)
  if [ "$ninjaAndCollier" == "1" ]; then
    if [ "$(grep -m1 ^ENV_VFOLDTEST_MLREDUCTION= ${events}S/LOG.txt)" == "ENV_VFOLDTEST_MLREDUCTION=ninja" ]; then
      other=collier; otherS=Sc
    else
      other=ninja; otherS=Sn
    fi
    ENV_VFOLDTEST_MLREDUCTION=${other} ENV_VFOLD_ME_MODE=F1 ./vfold/myLaunch.sh ${oOpt} ${xOpt}
    mv Events/run_01 ${events}${otherS}
  fi

  # Step 15e: also F1 without the stock MadGraph7 Born cache and gscale, for the timing table
  if [ "$noopt7" == "1" ]; then
    ENV_VFOLD_MG7BORNNOOPT=1 ENV_VFOLD_ME_MODE=F1 ./vfold/myLaunch.sh ${oOpt} ${xOpt}
    mv Events/run_01 ${events}Su
  fi

  ENV_VFOLD_ME_MODE=F2 ./vfold/myLaunch.sh ${oOpt} ${xOpt}
  mv Events/run_01 ${events}T

  ENV_VFOLD_ME_MODE=F ./vfold/myLaunch.sh ${oOpt} ${xOpt}
  mv Events/run_01 ${events}F

  ENV_VFOLD_CUDACPP_BACKEND=none ENV_VFOLD_ME_MODE=C ./vfold/myLaunch.sh ${oOpt} ${xOpt}
  mv Events/run_01 ${events}Cnone

  ENV_VFOLD_CUDACPP_BACKEND=sse4 ENV_VFOLD_ME_MODE=C ./vfold/myLaunch.sh ${oOpt} ${xOpt}
  mv Events/run_01 ${events}Csse4

  ENV_VFOLD_CUDACPP_BACKEND=avx2 ENV_VFOLD_ME_MODE=C ./vfold/myLaunch.sh ${oOpt} ${xOpt}
  mv Events/run_01 ${events}Cavx2

  if [ "$(grep -m1 -c Gold /proc/cpuinfo)" == "1" ]; then
    ENV_VFOLD_CUDACPP_BACKEND=512y ENV_VFOLD_ME_MODE=C ./vfold/myLaunch.sh ${oOpt} ${xOpt}
    mv Events/run_01 ${events}C512y
    ENV_VFOLD_CUDACPP_BACKEND=512z ENV_VFOLD_ME_MODE=C ./vfold/myLaunch.sh ${oOpt} ${xOpt}
    mv Events/run_01 ${events}C512z
  fi

  if nvidia-smi -L >& /dev/null; then
    unset CUDACPP_RUNTIME_BLASCOLORSUM CUDACPP_RUNTIME_BLASCOLORSUM_VFOLDBORN
    if [ "$cudascan" == "1" ]; then
      # With step9l parameters stop at vecsize=1024 for kbatch=32
      # With step10b parameters stop at vecsize=2048/4096 for kbatch=64/128
      if [ "${ENV_VFOLD_KPOINT_BATCH}" == "256" ] && [ "${ENV_VFOLD_SBORN_GSCALE}" == "1" ]; then
        vecsizes="32 64 128 256 512 1024 2048 4096 8192"
        vecsizesBlas="4096 8192"
      elif [ "${ENV_VFOLD_KPOINT_BATCH}" == "128" ] && [ "${ENV_VFOLD_SBORN_GSCALE}" == "1" ]; then
        vecsizes="32 64 128 256 512 1024 2048 4096"
        vecsizesBlas="2048 4096"
      elif [ "${ENV_VFOLD_KPOINT_BATCH}" == "64" ] && [ "${ENV_VFOLD_SBORN_GSCALE}" == "1" ]; then
        vecsizes="32 64 128 256 512 1024 2048"
        vecsizesBlas="1024 2048"
      elif [ "${ENV_VFOLD_KPOINT_BATCH}" == "32" ] && [ "${ENV_VFOLD_SBORN_GSCALE}" == "1" ]; then
        vecsizes="32 64 128 256 512 1024"
        vecsizesBlas="512 1024"
      else
        vecsizes="32 64 128 256 512 1024"
        vecsizesBlas="512 1024"
      fi
      for vecsize in $vecsizes; do
        ENV_VFOLD_VECSIZE_USED=${vecsize} ENV_VFOLD_CUDACPP_BACKEND=cuda ENV_VFOLD_ME_MODE=C ./vfold/myLaunch.sh ${oOpt} ${xOpt}
        mv Events/run_01 ${events}Ccuda${vecsize}
      done
      ###for vecsize in $vecsizesBlas; do
      ###  # BLAS for Born and Real libraries
      ###  CUDACPP_RUNTIME_BLASCOLORSUM=1 \
      ###    ENV_VFOLD_VECSIZE_USED=${vecsize} ENV_VFOLD_CUDACPP_BACKEND=cuda ENV_VFOLD_ME_MODE=C ./vfold/myLaunch.sh ${oOpt} ${xOpt}
      ###  mv Events/run_01 ${events}Ccudb${vecsize}
      ###done
      for vecsize in $vecsizesBlas; do
        # Step 14n: BLAS for the Real libraries only
        CUDACPP_RUNTIME_BLASCOLORSUM=1 CUDACPP_RUNTIME_BLASCOLORSUM_VFOLDBORN= \
          ENV_VFOLD_VECSIZE_USED=${vecsize} ENV_VFOLD_CUDACPP_BACKEND=cuda ENV_VFOLD_ME_MODE=C ./vfold/myLaunch.sh ${oOpt} ${xOpt}
        mv Events/run_01 ${events}Ccudbr${vecsize}
      done
    else
      ENV_VFOLD_CUDACPP_BACKEND=cuda ENV_VFOLD_ME_MODE=C ./vfold/myLaunch.sh ${oOpt} ${xOpt}
      mv Events/run_01 ${events}Ccuda
    fi
  fi

elif [ "$mode" == "--diff" ]; then

  events=${step}
  cmdtmp=$(mktemp)
  notcmp=

  # Step 14s: compare all the runs of this step that exist (before, only the
  # backends none, sse4, avx2, 512y, 512z and a single cuda run were compared:
  # the cuda scans Ccuda<N> and Ccudbr<N> and F1 with the other MadLoop library
  # were silently not compared). Each line also records a failure in ${status}.
  # F1/S and F2/T are identical to F, and so is F1 with the MadLoop library of F
  # (Sn or Sc from --ninja-and-collier; with the other library it is not compared)
  # (step 14s part 10/10: the variable was ENV_VFOLD_MLREDUCTION in older runs)
  mlF=$(grep -m1 -E '^ENV_VFOLD(TEST)?_MLREDUCTION=' Events/${events}F/LOG.txt 2>/dev/null | cut -d= -f2)
  mlF=${mlF:-ninja} # older runs do not print it, they all used ninja
  for sfx in S T Sn Sc; do
    if [ ! -d Events/${events}${sfx} ]; then
      if [ "${sfx}" == "S" ] || [ "${sfx}" == "T" ]; then notcmp="${notcmp} ${events}${sfx}(missing)"; fi
    elif [ "${sfx}" == "Sn" ] || [ "${sfx}" == "Sc" ] && [ "${sfx}" != "S${mlF:0:1}" ]; then
      notcmp="${notcmp} ${events}${sfx}(F uses ${mlF})"
    else
      echo "  ./vfold/myDiffRun.sh ${oOpt} ${events}${sfx} ${events}F || status=1" >> ${cmdtmp}
    fi
  done
  # Step 15e: F1 without the MadGraph7 Born reuse is equal to F within tolerance (gscale rescales amplitudes)
  if [ -d Events/${events}Su ]; then
    echo "  ./vfold/myDiffRun.sh ${oOpt} -nd5 ${events}F ${events}Su || status=1" >> ${cmdtmp}
  fi
  # Each C (all CPU backends, all cuda and cudb/cudbr vecsizes found) is equal to F within tolerance
  for d in $(for d in Events/${events}C*; do if [ -d ${d} ]; then echo ${d#Events/}; fi; done | sort -V); do
    echo "  ./vfold/myDiffRun.sh ${oOpt} -nd5 ${events}F ${d} || status=1" >> ${cmdtmp}
  done

  status=0
  source ${cmdtmp}

  echo
  echo "Compared results using"
  echo "  $0 ${ARGS}"
  echo "Modes F1/S, F2/T (and F1 with the MadLoop library of F) and F give identical results"
  echo "Modes F and all Cs (and F1 without the MadGraph7 Born reuse, if any) are equal within tolerance"
  sed 's/ || status=1$//' ${cmdtmp}
  if [ "${notcmp}" != "" ]; then echo "NOT compared:${notcmp}"; fi
  echo "STATUS=${status} (all comparisons above)"
  rm ${cmdtmp}
  exit ${status}

elif [ "$mode" == "--diff2" ]; then

  refevt=${refstep}
  events=${step}
  cmdtmp=$(mktemp)
  lnktmp=$(mktemp)
  notcmp=

  # Step 14s: compare each run of this step that exists to the same run of the
  # reference step (S and T to F of the reference if this has no S or T), also
  # the cuda scans Ccuda<N> and Ccudbr<N> and F1 with the other MadLoop library
  # (before, only Z, S, T, F, the CPU backends and a single cuda run were compared)
  for d in $(for d in Events/${events}*; do if [ -d ${d} ]; then echo ${d#Events/}; fi; done | sort -V); do
    sfx=${d#${events}}
    if [ "${sfx}" == "" ] || [[ "${sfx}" =~ / ]]; then continue; fi
    ref=${refevt}${sfx}
    if [ ! -d Events/${ref} ] && { [ "${sfx}" == "S" ] || [ "${sfx}" == "T" ]; }; then ref=${refevt}F; fi
    if [ -d Events/${ref} ]; then
      echo "  ./vfold/myDiffRun.sh ${oOpt} ${ref} ${d} || status=1" >> ${cmdtmp}
    else
      notcmp="${notcmp} ${d}"
    fi
  done

  sed -e "s|./vfold/myDiffRun.sh|../linkDumps.sh|" -e "s/ || status=1$//" ${cmdtmp} > ${lnktmp}

  status=0
  source ${cmdtmp}

  echo
  echo "Compared results using"
  echo "  $0 ${ARGS}"
  echo "Each run is identical to the same run of the reference ${refstep}"
  echo "(S and T to F of the reference if this has no S or T)"
  sed 's/ || status=1$//' ${cmdtmp}
  if [ "${notcmp}" != "" ]; then echo "NOT compared (not in the reference):${notcmp}"; fi
  echo "STATUS=${status} (all comparisons above)"
  rm ${cmdtmp}

  if [ "${oOpt}" == "" ]; then
    echo
    echo "Link dumpfiles"
    cat ${lnktmp}
    echo
  else
    ###echo "There are no dumpfiles to compare or link (-o option)"
    echo
  fi
  rm ${lnktmp}
  exit ${status}

else

  echo "ERROR! Invalid mode=${mode}"
  exit 1

fi
