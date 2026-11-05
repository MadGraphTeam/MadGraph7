#!/bin/bash
# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Aug 2026) for project MadVFold (arXiv:2609.23500).

set -e
set -o pipefail

cd $(dirname $0)/..
PROC=$(basename $(pwd))

# Command line arguments
ARGS="$*"
oOpt=
xOpt=
while [ "$1" != "" ]; do
  if [ "$1" == "-o" ]; then
    oOpt=-o # bin/generate_events -o: skip MINT0/1, execute only MINT2 from existing grids
  elif [ "$1" == "-x" ]; then
    xOpt=-x # bin/generate_events -s: skip compilation, use existing binaries
  else
    echo "Usage:   $0 [-o] [-x]"
    echo "Example: ENV_VFOLD_ME_MODE=F $0"
    echo "Example: ENV_VFOLD_CUDACPP_BACKEND=avx2 ENV_VFOLD_ME_MODE=C $0"
    echo "Example: ENV_NEVENTS=10 ENV_FOLDING='8,4,1' ENV_VFOLD_ME_MODE=F $0 -o"
    echo "Example: ENV_MCATNLODELTA=1 ENV_VFOLD_ME_MODE=F0 $0"
    echo "Example: ENV_PHYSICS=1 ENV_REQACC=0.01 ENV_VFOLD_ME_MODE=F0 $0"
    echo "Example: ENV_ISEED=84101 ENV_VFOLD_ME_MODE=F0 $0"
    echo "Example: ENV_VFOLD_MLREDUCTION=ninja ENV_VFOLD_ME_MODE=F1 $0"
    echo "Example: ENV_VFOLD_POLYCACHE=0 ENV_VFOLD_ME_MODE=F1 $0"
    echo "Example: CUDACPP_RUNTIME_BLASCOLORSUM=1 CUDACPP_RUNTIME_BLASCOLORSUM_VFOLDBORN= ENV_VFOLD_CUDACPP_BACKEND=cuda ENV_VFOLD_ME_MODE=C $0"
    exit 1
  fi
  shift
done

# Create the logfile
out=$(pwd)/LOG.txt
start=`date`
status=0
echo "START:    $start" | tee $out
echo "HOSTNAME: $(hostname)" | tee -a $out
echo "PROC:     ${PROC}" | tee -a $out
echo "" | tee -a $out
echo "EXECUTING: $0 ${ARGS}" | tee -a $out
echo "" | tee -a $out

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
#  export F90CACHE_LOGFILE=$(cd madvfold/DEV/${PROC}; pwd)/f90cache.log
#fi
#popd > /dev/null
echo FC=${FC} | tee -a $out
echo F90CACHE_DIR=${F90CACHE_DIR} | tee -a $out
echo F90CACHE_LOGFILE=${F90CACHE_LOGFILE} | tee -a $out

# Dump software versions
vout=VERSIONS.txt
python3 --version | head -1 | tee $vout
gcc --version | head -1 | tee -a $vout
gfortran --version | head -1 | tee -a $vout
echo "FC=${FC}" | tee -a $vout
echo "MG5_VERSION $(cat MGMEVersion.txt)" | tee -a $vout
if f90cache -V >& /dev/null; then
  f90cache -V | head -1 | tee -a $vout
else
  echo "f90cache is not installed" | tee -a $vout
fi
if nvcc --version >& /dev/null; then
  nvcc --version | grep 'Cuda compilation tools' | tee -a $vout
else
  echo "nvcc is not installed" | tee -a $vout
fi
more ../../../HEPTools/pythia8/include/Pythia8/Pythia.h | \grep 'PYTHIA_VERSION ' | awk '{print $2,$3}' | tee -a $vout

# Only modes F0, F1, F2, F and C are supported
if [ "${ENV_VFOLD_ME_MODE}" != "F0" ] && [ "${ENV_VFOLD_ME_MODE}" != "F1" ] && [ "${ENV_VFOLD_ME_MODE}" != "F2" ] && [ "${ENV_VFOLD_ME_MODE}" != "F" ] && [ "${ENV_VFOLD_ME_MODE}" != "C" ]; then
  echo "ERROR! Invalid ENV_VFOLD_ME_MODE=${ENV_VFOLD_ME_MODE}"
  exit 1
fi

# Reduce ENV_FVZ_NDUMP from 50 (eebb) to 5 (eebbbb, eemumubb) to reduce dump sizes
if [ "${PROC}" != "${PROC/eebb_}" ]; then
  if [ "${ENV_FVZ_NDUMP}" == "" ]; then export ENV_FVZ_NDUMP=50; fi # dump 50 events as in myLaunch.sh (eebb)
else
  if [ "${ENV_FVZ_NDUMP}" == "" ]; then export ENV_FVZ_NDUMP=5; fi # dump 5 events as in myLaunch.sh (eebbbb, eemumubb)
fi

# Folding
if [ "${ENV_FOLDING}" == "" ]; then ENV_FOLDING="8, 4, 1"; fi # default 8x4x1

# MC@NLO-Delta matching (1: set mcatnlo_delta=True in the run_card; unset or 0: keep the run_card as is)
if [ "${ENV_MCATNLODELTA}" != "" ] && [ "${ENV_MCATNLODELTA}" != "0" ] && [ "${ENV_MCATNLODELTA}" != "1" ]; then
  echo "ERROR! Invalid ENV_MCATNLODELTA=${ENV_MCATNLODELTA}"
  exit 1
fi
if [ "${ENV_MCATNLODELTA}" == "1" ] && [ "$xOpt" == "-x" ]; then
  echo "WARNING! ENV_MCATNLODELTA=1 with -x: existing binaries are used, check that they were built with mcatnlo_delta=True"
fi

# Physics runs (1: stock MINT0/MINT1 parameters and all channels, i.e. keep ENV_VFOLD_MINT* and ENV_VFOLD_CHANNELS
# unset instead of the faster test settings below; unset or 0: test settings)
if [ "${ENV_PHYSICS}" != "" ] && [ "${ENV_PHYSICS}" != "0" ] && [ "${ENV_PHYSICS}" != "1" ]; then
  echo "ERROR! Invalid ENV_PHYSICS=${ENV_PHYSICS}"
  exit 1
fi
if [ "${ENV_PHYSICS}" == "1" ] && [ "$oOpt" == "-o" ]; then
  echo "ERROR! ENV_PHYSICS=1 is not supported with -o (physics tests rerun all of MINT0/1/2)"
  exit 1
fi

# Required accuracy (req_acc in the run_card; unset: keep the run_card value, -1 = auto from nevents)
if [ "${ENV_REQACC}" != "" ] && ! [[ "${ENV_REQACC}" =~ ^(-1|0?\.[0-9]+|1)$ ]]; then
  echo "ERROR! Invalid ENV_REQACC=${ENV_REQACC} (must be -1 or in ]0,1])"
  exit 1
fi

# Random seed (iseed in the run_card; unset: default 84100 below)
if [ "${ENV_ISEED}" != "" ] && ! [[ "${ENV_ISEED}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR! Invalid ENV_ISEED=${ENV_ISEED} (must be a positive integer)"
  exit 1
fi

# Define folding-specific reference for MINT2-only generation (-o)
filesK="res_1.dat mint_grids res_1 grid.MC_integer results.dat"
ofld=
if [ "$oOpt" == "-o" ]; then
  if [ "${ENV_FOLDING}" == "8, 4, 1" ] || [ "${ENV_FOLDING}" == "8,4,1" ]; then
    ofld=841
  elif [ "${ENV_FOLDING}" == "1, 1, 1" ] || [ "${ENV_FOLDING}" == "1,1,1" ]; then
    ofld=111
  else
    echo "ERROR! Unknown ENV_FOLDING=${ENV_FOLDING} in -o mode"
    exit 1
  fi
  # MINT0/1 grids for this folding (see vfold/test/grids/README)
  # With Delta, use MINT0/1 grids from a Delta physics run in vfold5381 (MG5 3.8.1, aS 0.118):
  # the MINT1 upper envelope of the v363 grids was computed without Delta
  refgen=vfold/test/grids/folding_$(echo ${ENV_FOLDING} | sed 's/ //g; s/,/_/g')
  if [ "${ENV_MCATNLODELTA}" == "1" ]; then
    refgen=${refgen}_Delta
  fi
  if [ ! -d ${refgen} ]; then
    echo "ERROR! Missing directory refgen=${refgen} in -o mode"
    exit 1
  fi
  for f in job_status.pkl aS.txt; do
    if [ ! -f ${refgen}/${f} ]; then
      echo "ERROR! Missing file ${refgen}/${f} in -o mode"
      exit 1
    fi
  done
  # Keep the MINT2 order of events in events.lhe (MG5 3.8.1 collect_events.py shuffles them)
  export ENV_VFOLD_NOSHUFFLE=1
else
  # Let MG5 3.8.1 collect_events.py reshuffle the MINT2 order of events in events.lhe
  unset ENV_VFOLD_NOSHUFFLE
fi

# Tweak MINT parameters for faster tests
# This is a necessity for more complex processes (more channels, slower MEs)
# like eebbbb and eemumubb, but is useful also for the simplest eebb process
# These are (functional/timing) software tests, physics accuracy is not the goal
# See create_jobs_to_run and write_input_file in amcatnlo_run_interface.py
# Increase ENV_NEVENTS for slower MINT2 generation for more accurate timing tests
randinit=84100
if [ "${ENV_ISEED}" != "" ]; then randinit=${ENV_ISEED}; fi
if [ "${PROC}" != "${PROC/eebb_}" ]; then
  if [ "${ENV_NEVENTS}" == "" ]; then ENV_NEVENTS=10; fi # default 10
  export ENV_VFOLD_MINT0_NITERS=2    # MINT0 (grid setup) iterations [DEFAULT: 12]
  export ENV_VFOLD_MINT0_ACC=0.5     # MINT0 accuracy target [DEFAULT: 0.03]
  export ENV_VFOLD_MINT1_NITERS=2    # MINT1 (upper envelope) iterations [DEFAULT: 12]
  export ENV_VFOLD_MINT0_NPOINTS=500 # MINT0 points/iter [DEFAULT: -1=auto, can grow to 1000s]
  export ENV_VFOLD_MINT1_NPOINTS=500 # MINT1 points/iter [DEFAULT: -1=auto, can grow to 1000s]
  ###unset ENV_VFOLD_CHANNELS        # GF channels (space-separated) [DEFAULT: all channels]
  if [ "${ENV_VFOLD_CHANNELS}" == "" ]; then export ENV_VFOLD_CHANNELS="2.0"; fi # GF channels (space-separated) [DEFAULT: all channels]
elif [ "${PROC}" != "${PROC/eebbbb_}" ]; then
  if [ "${ENV_NEVENTS}" == "" ]; then
    if [ "${ofld}" == "841" ]; then
      ###randinit="84100" # 84100/1/2/3/4 give virtual=0 with 10 events
      ENV_NEVENTS=20 # 841 grids: 10 in ~22s (no virtual!), 20-50-100 in ~33-70-135s (virtual)
    elif [ "${ofld}" == "111" ]; then
      ENV_NEVENTS=200 # 111 grids: 50 in ~2.5s, 100 in ~5s, 200 in ~10s
    else
      ENV_NEVENTS=10 # default 10
    fi
  fi
  export ENV_VFOLD_MINT0_NITERS=2    # MINT0 (grid setup) iterations [DEFAULT: 12]
  export ENV_VFOLD_MINT0_ACC=0.5     # MINT0 accuracy target [DEFAULT: 0.03]
  export ENV_VFOLD_MINT1_NITERS=2    # MINT1 (upper envelope) iterations [DEFAULT: 12]
  export ENV_VFOLD_MINT0_NPOINTS=50  # MINT0 points/iter [DEFAULT: -1=auto, can grow to 1000s]
  export ENV_VFOLD_MINT1_NPOINTS=50  # MINT1 points/iter [DEFAULT: -1=auto, can grow to 1000s]
  # NB: may need to keep ENV_VFOLD_CHANNELS unset to investigate bugs in channel mapping
  if [ "${ENV_VFOLD_CHANNELS}" == "" ]; then export ENV_VFOLD_CHANNELS="11.0"; fi   # GF channels (space-separated) [DEFAULT: all channels]
elif [ "${PROC}" != "${PROC/eemumubb_}" ]; then
  if [ "${ENV_NEVENTS}" == "" ]; then ENV_NEVENTS=10; fi # default 10
  export ENV_VFOLD_MINT0_NITERS=2    # MINT0 (grid setup) iterations [DEFAULT: 12]
  export ENV_VFOLD_MINT0_ACC=0.5     # MINT0 accuracy target [DEFAULT: 0.03]
  export ENV_VFOLD_MINT1_NITERS=2    # MINT1 (upper envelope) iterations [DEFAULT: 12]
  export ENV_VFOLD_MINT0_NPOINTS=50  # MINT0 points/iter [DEFAULT: -1=auto, can grow to 1000s]
  export ENV_VFOLD_MINT1_NPOINTS=50  # MINT1 points/iter [DEFAULT: -1=auto, can grow to 1000s]
  # NB: may need to keep ENV_VFOLD_CHANNELS unset to investigate bugs in channel mapping
  if [ "${ENV_VFOLD_CHANNELS}" == "" ]; then export ENV_VFOLD_CHANNELS="11.0"; fi   # GF channels (space-separated) [DEFAULT: all channels]
elif [ "${PROC}" != "${PROC/eemumubbg_}" ]; then
  if [ "${ENV_NEVENTS}" == "" ]; then ENV_NEVENTS=10; fi # default 10
  # Step 14h: COLLIER by default, except in physics tests (Ninja amplifies rounding, see ENV_VFOLD_MLREDUCTION below)
  if [ "${ENV_VFOLD_MLREDUCTION}" == "" ] && [ "${ENV_PHYSICS}" != "1" ]; then export ENV_VFOLD_MLREDUCTION=collier; fi
  export ENV_VFOLD_MINT0_NITERS=2    # MINT0 (grid setup) iterations [DEFAULT: 12]
  export ENV_VFOLD_MINT0_ACC=0.5     # MINT0 accuracy target [DEFAULT: 0.03]
  export ENV_VFOLD_MINT1_NITERS=2    # MINT1 (upper envelope) iterations [DEFAULT: 12]
  export ENV_VFOLD_MINT0_NPOINTS=50  # MINT0 points/iter [DEFAULT: -1=auto, can grow to 1000s]
  export ENV_VFOLD_MINT1_NPOINTS=50  # MINT1 points/iter [DEFAULT: -1=auto, can grow to 1000s]
  # NB: may need to keep ENV_VFOLD_CHANNELS unset to investigate bugs in channel mapping
  # Step 14q: GF58.0 (13% of the ABS integral) instead of GF11.0 (0.1%), see vfold/test/grids/README
  if [ "${ENV_VFOLD_CHANNELS}" == "" ]; then export ENV_VFOLD_CHANNELS="58.0"; fi # GF channels (space-separated) [DEFAULT: all channels]
elif [ "${PROC}" != "${PROC/ggttg_}" ]; then
  if [ "${ENV_NEVENTS}" == "" ]; then ENV_NEVENTS=10; fi # default 10
  # Step 14n: these can be set in the environment, e.g. to generate the -o test grids
  if [ "${ENV_VFOLD_MINT0_NITERS}" == "" ]; then export ENV_VFOLD_MINT0_NITERS=2; fi    # MINT0 (grid setup) iterations [DEFAULT: 12]
  if [ "${ENV_VFOLD_MINT0_ACC}" == "" ]; then export ENV_VFOLD_MINT0_ACC=0.5; fi        # MINT0 accuracy target [DEFAULT: 0.03]
  if [ "${ENV_VFOLD_MINT1_NITERS}" == "" ]; then export ENV_VFOLD_MINT1_NITERS=2; fi    # MINT1 (upper envelope) iterations [DEFAULT: 12]
  if [ "${ENV_VFOLD_MINT0_NPOINTS}" == "" ]; then export ENV_VFOLD_MINT0_NPOINTS=50; fi # MINT0 points/iter [DEFAULT: -1=auto, can grow to 1000s]
  if [ "${ENV_VFOLD_MINT1_NPOINTS}" == "" ]; then export ENV_VFOLD_MINT1_NPOINTS=50; fi # MINT1 points/iter [DEFAULT: -1=auto, can grow to 1000s]
  # NB: may need to keep ENV_VFOLD_CHANNELS unset to investigate bugs in channel mapping
  # Step 14o: GF10.0 is the largest channel with virtuals (the -o test grids are for this channel)
  if [ "${ENV_VFOLD_CHANNELS}" == "" ]; then export ENV_VFOLD_CHANNELS="10.0"; fi # GF channels (space-separated) [DEFAULT: all channels]
else
  echo "ERROR! Unknown PROC=${PROC}"
  exit 1
fi
if [ "${ENV_PHYSICS}" == "1" ]; then
  unset ENV_VFOLD_MINT0_NITERS ENV_VFOLD_MINT0_ACC ENV_VFOLD_MINT1_NITERS ENV_VFOLD_MINT0_NPOINTS ENV_VFOLD_MINT1_NPOINTS ENV_VFOLD_CHANNELS
  # No dumps: the extra MINT1 dump iteration is included in the MINT1 results and biases them
  export ENV_FVZ_NDUMP=0
fi
# Check that the -o grids exist for the selected channels (before changing any card)
if [ "$oOpt" == "-o" ] && [ "${ENV_VFOLD_CHANNELS}" != "" ]; then
  for Pdir in $(cd SubProcesses; ls -d P*); do
    for gf in ${ENV_VFOLD_CHANNELS}; do
      if [ ! -d ${refgen}/${Pdir}/GF${gf} ]; then
        echo "ERROR! No -o grids for channel GF${gf} (missing directory ${refgen}/${Pdir}/GF${gf})"
        echo "Available channels: $(ls ${refgen}/${Pdir} 2>/dev/null | tr '\n' ' ')"
        exit 1
      fi
    done
  done
fi
cp Cards/run_card.dat Cards/run_card.dat.original
cp Source/run_card.inc Source/run_card.inc.original
cp SubProcesses/randinit SubProcesses/randinit.original
cp Cards/param_card.dat Cards/param_card.dat.original
cp Source/param_card.inc Source/param_card.inc.original
if [ "$oOpt" == "-o" ]; then # fort.77 (soft/collinear checks in fks_singular.f depends on aS)
  for f in SubProcesses/P*/fort.77; do if [ -f ${f} ]; then cp -p ${f} ${f}.original; fi; done # (not in a new directory before its first build)
fi
sed -i "s/.*= nevents/  ${ENV_NEVENTS} = nevents/" Cards/run_card.dat
sed -i "s/.*= folding/  ${ENV_FOLDING} = folding/" Cards/run_card.dat
sed -i "s/.*= iseed/  ${randinit} = iseed/" Cards/run_card.dat
if [ "${ENV_REQACC}" != "" ]; then
  sed -i "s/.*= req_acc /  ${ENV_REQACC} = req_acc /" Cards/run_card.dat
fi
# Use in MINT2 the aS(MZ) of the MINT0/1 grids (aS.txt, see vfold/test/grids/README):
# the v363 grids were computed with aS(MZ)=0.119 (MG5 3.6.3 took it from pdlabel nn23nlo
# even if lpp=0), while the MG5 3.8.1 param_card has 0.118
if [ "$oOpt" == "-o" ]; then
  asmz=$(printf "%e" $(cat ${refgen}/aS.txt))
  sed -i "s/^ *3 [0-9.e+-]* # aS /    3 ${asmz} # aS /" Cards/param_card.dat
  if ! grep -q "^    3 ${asmz} # aS " Cards/param_card.dat; then
    echo "ERROR! Could not set aS(MZ)=${asmz} in Cards/param_card.dat"
    for f in Cards/run_card.dat Source/run_card.inc SubProcesses/randinit Cards/param_card.dat Source/param_card.inc SubProcesses/P*/fort.77; do if [ -f ${f}.original ]; then mv ${f}.original ${f}; fi; done
    exit 1
  fi
fi
if [ "${ENV_MCATNLODELTA}" == "1" ]; then
  sed -i "s/.*= mcatnlo_delta/  True = mcatnlo_delta/I" Cards/run_card.dat # case-insensitive: MCatNLO_DELTA in the 3.8.1 run_card
  cp SubProcesses/pythia8_opts SubProcesses/pythia8_opts.original
  cp SubProcesses/pythia8_control_setup.inc SubProcesses/pythia8_control_setup.inc.original
fi

# Delta needs Pythia8: set pythia8_path relative to the process dir
# (the mg5 dir is ../../../), which the MG5 run interface can resolve
sed -i "s|^pythia8_path = .*|pythia8_path = ../../../HEPTools/pythia8|" Cards/amcatnlo_configuration.txt

# Copy files that are needed for MINT2-only generation (-o)
if [ "$oOpt" == "-o" ]; then
  if ls SubProcesses/P*/GF* >& /dev/null; then
    echo "WARNING! removing existing SubProcesses/P*/GF*"
    rm -rf SubProcesses/P*/GF*
  fi
  # job_status.pkl from MG5 3.8.1 (MG5 replaces its hardcoded avalassi paths)
  cp -p ${refgen}/job_status.pkl SubProcesses/job_status.pkl
  Pdirs=$(cd SubProcesses; ls -d P*)
  for Pdir in ${Pdirs}; do
    if [ "${ENV_VFOLD_CHANNELS}" != "" ]; then
      gfs="$(for gf in ${ENV_VFOLD_CHANNELS}; do echo GF${gf}; done)"
    else
      gfs=$(cd ${refgen}/${Pdir}; ls -d GF*)
    fi
    for gf in $gfs; do
      for f in ${filesK}; do
        mkdir -p SubProcesses/${Pdir}/${gf}
        cp -p ${refgen}/${Pdir}/${gf}/${f} SubProcesses/${Pdir}/${gf}
      done
    done
  done
fi

# Cudacpp: use avx2 by default rather than 512y to ease portability (e.g. to lbbuildinter02)
# Reduce VECSIZE_USED which otherwise would use the full VECSIZE_MEMMAX (16384)
export ENV_VFOLD_CUDACPP_BASE=$(pwd -P)/lib/cudacpp
if [ "${ENV_VFOLD_ME_MODE}" == "C" ]; then
  if [ "${ENV_VFOLD_CUDACPP_BACKEND}" == "" ]; then export ENV_VFOLD_CUDACPP_BACKEND=avx2; fi
  if [ "${ENV_VFOLD_CUDACPP_BACKEND}" == "512z" ]; then
    export ENV_VFOLD_VECSIZE_USED=8 # 8 doubles
  elif [ "${ENV_VFOLD_CUDACPP_BACKEND}" == "512y" ]; then
    export ENV_VFOLD_VECSIZE_USED=4 # 4 doubles
  elif [ "${ENV_VFOLD_CUDACPP_BACKEND}" == "avx2" ]; then
    export ENV_VFOLD_VECSIZE_USED=4 # 4 doubles
  elif [ "${ENV_VFOLD_CUDACPP_BACKEND}" == "sse4" ]; then
    export ENV_VFOLD_VECSIZE_USED=2 # 2 doubles
  elif [ "${ENV_VFOLD_CUDACPP_BACKEND}" == "none" ]; then
    export ENV_VFOLD_VECSIZE_USED=1 # 1 double
  elif [ "${ENV_VFOLD_CUDACPP_BACKEND}" == "cuda" ]; then
    if [ "${ENV_VFOLD_VECSIZE_USED}" == "" ]; then
      export ENV_VFOLD_VECSIZE_USED=32 # 32 threads per warp
    fi
  else
    echo "ERROR! Invalid ENV_VFOLD_CUDACPP_BACKEND=${ENV_VFOLD_CUDACPP_BACKEND}"
    exit 1
  fi
else
  export ENV_VFOLD_CUDACPP_BACKEND=
  export ENV_VFOLD_VECSIZE_USED=
fi

# Define path to libninja.so.0 (alternative to using rpath in the makefiles)
export LD_LIBRARY_PATH=$(cd ../../..; pwd -P)/HEPTools/lib:${LD_LIBRARY_PATH}

# First copy the cudacpp libraries if needed (check if mode is C)
# Only warn if there is no vfold/copyCudacppLibs.sh (mode C may not use cudacpp yet)
# Step 14p: only copy them if this has not been done (they are copied at generation)
# or if they have been rebuilt since then (e.g. with vfold/buildCudacppLibs.sh)
if [ "${ENV_VFOLD_ME_MODE}" == "C" ]; then
  if [ -f ./vfold/copyCudacppLibs.sh ]; then
    if [ ! -f lib/cudacpp/.copied ] || [ -n "$(find CUDACPP -name 'libmg5amc_*.so' -newer lib/cudacpp/.copied -print -quit 2>/dev/null)" ]; then
      ./vfold/copyCudacppLibs.sh | tee -a $out
      touch lib/cudacpp/.copied
    fi
  else
    echo "WARNING! ./vfold/copyCudacppLibs.sh not found: the cudacpp libraries are not copied" | tee -a $out
  fi
fi

# Road B step 9a: EVENT (kpoint) batching -- how many kpoints are gathered
# before their Born/Real/SDE matrix elements are evaluated in one batch.
# Default 1 = previous behaviour (batch width = number of folds only).
# Capped internally so that nkpoints*nfolds fits the 8192-entry batch
# buffers (8192 as of step 11g, 4096 as of step 9l, 128 before), for
# instance 8192 at folding 1,1,1 and 256 at folding 8,4,1.
if [ "${ENV_VFOLD_KPOINT_BATCH}" == "" ]; then export ENV_VFOLD_KPOINT_BATCH=1; fi

# Step 14h: MadLoop reduction library, for all modes: ninja (MLReductionLib=6|7|1, the MG5
# default: Ninja, then COLLIER and CutTools if Ninja is not stable) or collier (MLReductionLib=7).
# The default is ninja, except for the non-physics tests of the processes which set collier above. With ninja, rounding
# differences change the virtual, up to 1e-4 and more at some points, as they change which library
# is used: in eemumubbg, F and each C backend differ. It is also slower if Ninja is often not stable.
if [ "${ENV_VFOLD_MLREDUCTION}" == "" ]; then export ENV_VFOLD_MLREDUCTION=ninja; fi
if [ "${ENV_VFOLD_MLREDUCTION}" == "ninja" ]; then
  mlreductionlib="6|7|1"
elif [ "${ENV_VFOLD_MLREDUCTION}" == "collier" ]; then
  mlreductionlib="7"
else
  echo "ERROR! Invalid ENV_VFOLD_MLREDUCTION=${ENV_VFOLD_MLREDUCTION} (must be ninja or collier)"
  exit 1
fi
# (both files: the compilation, i.e. without -x, updates the SubProcesses one from the Cards one)
for f in Cards/MadLoopParams.dat SubProcesses/MadLoopParams.dat; do
  cp ${f} ${f}.original
  sed -i "/^#MLReductionLib/{n;s/.*/${mlreductionlib}/}" ${f}
done

# Road B step 9i/j: SBORN served from cudacpp (mode C only)
# Road B step 11y: no fused BSF kernel, the colour-linked Borns are also served from cudacpp
if [ "${ENV_VFOLD_SBORN_BRIDGE}" == "" ]; then export ENV_VFOLD_SBORN_BRIDGE=1; fi

# Road B step 11a: MG5 timers (Fortran cpu_time) via vfold_cpu_time.c
# 0=getrusage syscall (as libgfortran), 1=vDSO monotonic clock, 2=rdtsc
# Modes 1 and 2 measure wall time (not process CPU time) since the first call
# Road B step 11b: -1 (-2) = 0 for the reference modes F0/F1, 1 (2) for F2/F/C
if [ "${ENV_VFOLD_CPUTIME}" == "" ]; then export ENV_VFOLD_CPUTIME=-1; fi

# Road B step 11d: with ENV_VFOLD_KPOINT_BATCH>1, park the ME batch of a MINT2
# queue (virt/novi) when another queue overwrites the buffers, instead of
# regathering it later (0=regather as before, 1=park)
if [ "${ENV_VFOLD_BATCH_PERVN}" == "" ]; then export ENV_VFOLD_BATCH_PERVN=1; fi

# Step 14m: cache of the compilation of the MadLoop polynomial.f, by far the slowest Fortran file
# (9 of the 13 minutes of the compilation of ggttg), for the regeneration of a process already built.
# Only this file is cached: it uses no vfold code and its module POLYNOMIAL_CONSTANTS is only used
# in its V0 directory. The key includes all it depends on, so that an entry can never be stale. Its
# polynomial.o and polynomial_constants.mod are stored together, in madvfold/DEV/fcache (untracked).
# Disable it with ENV_VFOLD_POLYCACHE=0. It is only used in directories in madvfold/DEV, without -x.
polyCacheDir=$(cd ..; pwd)/fcache
declare -A polyCacheKeys
polyCacheStamp=
polyCacheRestore() {
  if [ "${ENV_VFOLD_POLYCACHE}" == "0" ] || [ "$xOpt" == "-x" ] || [ "$(basename $(cd ..; pwd))" != "DEV" ]; then return; fi
  local v key entry
  mkdir -p ${polyCacheDir}
  for v in SubProcesses/P*/V0_*; do
    if [ ! -f ${v}/polynomial.f ]; then continue; fi
    # (the key is computed before the build, which changes Source/make_opts)
    key=$( (cat ${v}/polynomial.f ${v}/coef_specs.inc ${v}/loop_max_coefs.inc ${v}/makefile Source/make_opts; echo "FC=${FC} FFLAGS=${FFLAGS}"; ${FC:-gfortran} --version | head -1) | md5sum | cut -d' ' -f1)
    polyCacheKeys[${v}]=${key}
    entry=${polyCacheDir}/polynomial_${key}
    if [ -f ${entry}/polynomial.o ] && [ -f ${entry}/polynomial_constants.mod ]; then
      if [ ! -f ${v}/polynomial.o ] || [ ${v}/polynomial.f -nt ${v}/polynomial.o ]; then
        cp ${entry}/polynomial.o ${entry}/polynomial_constants.mod ${v}/
        touch ${v}/polynomial.o ${v}/polynomial_constants.mod
        echo "FCACHE: polynomial.o restored for ${v} (key ${key})" | tee -a $out
      fi
    fi
  done
  polyCacheStamp=$(mktemp ${polyCacheDir}/stamp.XXXXXX)
}
polyCacheStore() {
  if [ "${polyCacheStamp}" == "" ]; then return; fi
  local v entry tmp
  sleep 1
  for v in "${!polyCacheKeys[@]}"; do
    entry=${polyCacheDir}/polynomial_${polyCacheKeys[${v}]}
    # Only what has been compiled by this build, written atomically
    if [ ! -d ${entry} ] && [ ${v}/polynomial.o -nt ${polyCacheStamp} ] && [ -f ${v}/polynomial_constants.mod ]; then
      tmp=$(mktemp -d ${polyCacheDir}/tmp.XXXXXX)
      cp ${v}/polynomial.o ${v}/polynomial_constants.mod ${tmp}/
      mv -T ${tmp} ${entry} 2>/dev/null || rm -rf ${tmp}
      echo "FCACHE: polynomial.o stored for ${v} (key ${polyCacheKeys[${v}]})" | tee -a $out
    fi
  done
  rm -f ${polyCacheStamp}
}

# Dump function (used twice: before and after generation)
dumpENV() {
  echo "ENV_NEVENTS=${ENV_NEVENTS}" | tee -a $out
  echo "ENV_FOLDING=${ENV_FOLDING}" | tee -a $out
  echo "ENV_VFOLD_CUDACPP_BASE=${ENV_VFOLD_CUDACPP_BASE}" | tee -a $out
  echo "ENV_VFOLD_CUDACPP_BACKEND=${ENV_VFOLD_CUDACPP_BACKEND}" | tee -a $out
  echo "ENV_VFOLD_ME_MODE=${ENV_VFOLD_ME_MODE}" | tee -a $out
  echo "ENV_VFOLD_VECSIZE_USED=${ENV_VFOLD_VECSIZE_USED}" | tee -a $out
  echo "ENV_VFOLD_KPOINT_BATCH=${ENV_VFOLD_KPOINT_BATCH}" | tee -a $out
  echo "ENV_VFOLD_SBORN_BRIDGE=${ENV_VFOLD_SBORN_BRIDGE}" | tee -a $out
  echo "ENV_VFOLD_SBORN_GSCALE=${ENV_VFOLD_SBORN_GSCALE}" | tee -a $out
  echo "ENV_VFOLD_MLREDUCTION=${ENV_VFOLD_MLREDUCTION}" | tee -a $out
  echo "ENV_VFOLD_POLYCACHE=${ENV_VFOLD_POLYCACHE}" | tee -a $out
  echo "ENV_VFOLD_CPUTIME=${ENV_VFOLD_CPUTIME}" | tee -a $out
  echo "ENV_VFOLD_BATCH_PERVN=${ENV_VFOLD_BATCH_PERVN}" | tee -a $out
  echo "ENV_VFOLD_MINT0_ACC=${ENV_VFOLD_MINT0_ACC}" | tee -a $out
  echo "ENV_VFOLD_MINT0_NITERS=${ENV_VFOLD_MINT0_NITERS}" | tee -a $out
  echo "ENV_VFOLD_MINT0_NPOINTS=${ENV_VFOLD_MINT0_NPOINTS}" | tee -a $out
  echo "ENV_VFOLD_MINT1_NITERS=${ENV_VFOLD_MINT1_NITERS}" | tee -a $out
  echo "ENV_VFOLD_MINT1_NPOINTS=${ENV_VFOLD_MINT1_NPOINTS}" | tee -a $out
  echo "ENV_VFOLD_CHANNELS=${ENV_VFOLD_CHANNELS}" | tee -a $out
  echo "ENV_VFOLD_NOSHUFFLE=${ENV_VFOLD_NOSHUFFLE}" | tee -a $out
  echo "ENV_MCATNLODELTA=${ENV_MCATNLODELTA}" | tee -a $out
  echo "ENV_PHYSICS=${ENV_PHYSICS}" | tee -a $out
  echo "ENV_REQACC=${ENV_REQACC}" | tee -a $out
  echo "ENV_ISEED=${ENV_ISEED}" | tee -a $out
  echo "CUDACPP_RUNTIME_BLASCOLORSUM=${CUDACPP_RUNTIME_BLASCOLORSUM}" | tee -a $out
  # Step 14n: the overrides for each library, only if defined (empty is not the same as unset)
  (env | grep -E '^CUDACPP_RUNTIME_(BLASCOLORSUM|CUBLASTF32TENSOR)_VFOLD' || true) | sort | tee -a $out
  echo "randinit=${randinit}" | tee -a $out
}

# Dump before generation
dumpENV

# Launch the process with no parton showers (-p) and no interactive dialog (-f)
# Optionally (-o) skip MINT0/MINT1 and go straight to MINT2 generation
if ls run_*_debug.log >& /dev/null; then echo "WARNING! removing existing run_xx_debug.log files"; rm run_*_debug.log; fi
if [ -f ME5_debug ]; then echo "WARNING! removing existing ME5_debug file"; rm ME5_debug; fi
polyCacheRestore
if ! ./bin/generate_events -f -p ${oOpt} ${xOpt} | tee -a $out; then status=1; fi
if ls run_*_debug.log >& /dev/null; then echo "ERROR! run_*_debug.log file has been created"; status=1; fi
if [ -f ME5_debug ]; then echo "ERROR! ME5_debug file has been created"; status=1; fi
if [ "${status}" == "0" ]; then polyCacheStore; fi

# Dump after generation
dumpENV
end=`date`
echo "START:    $start" | tee -a $out
echo "END:      $end" | tee -a $out
echo "ELAPSED:  "$((`date +%s -d "$end"`-`date +%s -d"$start"`))" seconds" | tee -a $out
echo "" | tee -a $out
echo "STATUS=${status}" | tee -a $out

# Move LOG.txt and other result files from SubProcesses to Events/run_xx
# (move not copy: this ensures that they do not show up in git status)
# As of the move from fortran to cudacpp, also move grids and LHE files
runDir=$(ls -1trd Events/run_[0-9]* 2>/dev/null | tail -1)
# If the run failed before MG5 created Events/run_xx, create run_01 to keep the logs
if [ "${runDir}" == "" ]; then runDir=Events/run_01; mkdir -p ${runDir}; fi
mv ${vout} ${runDir}/
mv ${out} ${runDir}/
if ls run_*_debug.log >& /dev/null; then cp -p run_*_debug.log ${runDir}; fi
if [ -f ME5_debug ]; then cp -p ME5_debug ${runDir}; fi
files="nevents_unweighted results.dat"
for f in ${files}; do
  if [ -f SubProcesses/${f} ]; then mv SubProcesses/${f} ${runDir}; fi
done
files="${filesK} events.lhe\
 input_app_MINT0.txt input_app_MINT1.txt input_app_MINT2.txt input_app.txt\
 log.txt log_MINT0.txt log_MINT1.txt log_MINT2.txt\
 mint_grids_MINT0 mint_grids_MINT1 mint_grids_MINT2 nevts\
 res_0 res.dat res_0.dat res_2.dat results.dat"
Pdirs=$(cd SubProcesses; ls -d P*)
for Pdir in ${Pdirs}; do
  for gf in $(cd SubProcesses/${Pdir}; ls -d GF*); do
    mkdir -p ${runDir}/${Pdir}/${gf}
    for f in ${files}; do
      # MINT2 files and event.lhe may be missing if no events are generated in this channel
      if [ -f SubProcesses/${Pdir}/${gf}/${f} ]; then
        cp -p SubProcesses/${Pdir}/${gf}/${f} ${runDir}/${Pdir}/${gf}
      fi
    done
  done
done
if [ -f ${runDir}/events.lhe.gz ]; then gunzip ${runDir}/events.lhe.gz; fi
rm -f ${runDir}/RunMaterial.tar.gz
rm -f ${runDir}/*html
mv Cards/MadLoopParams.dat.original Cards/MadLoopParams.dat
mv SubProcesses/MadLoopParams.dat.original SubProcesses/MadLoopParams.dat
mv Cards/run_card.dat.original Cards/run_card.dat
mv Source/run_card.inc.original Source/run_card.inc
mv SubProcesses/randinit.original SubProcesses/randinit
mv Cards/param_card.dat.original Cards/param_card.dat
mv Source/param_card.inc.original Source/param_card.inc
# The libraries built for -o have the aS(MZ) of the grids, and the restored files are older
# than them: force a rebuild in the next run (else a plain run uses that aS)
if [ "$oOpt" == "-o" ]; then touch Source/param_card.inc; fi
if [ "$oOpt" == "-o" ]; then
  for f in SubProcesses/P*/fort.77; do if [ -f ${f}.original ]; then mv ${f}.original ${f}; fi; done
fi
if [ "${ENV_MCATNLODELTA}" == "1" ]; then
  mv SubProcesses/pythia8_opts.original SubProcesses/pythia8_opts
  mv SubProcesses/pythia8_control_setup.inc.original SubProcesses/pythia8_control_setup.inc
  rm -f SubProcesses/P*/LHAFortran_aMCatNLO.h.gch
fi
# The MG5 run interface rewrites PYTHIA8_PATH in make_opts if it does not
# find pythia8-config: undo only this change (keep any other change visible)
sed -i "s/^PYTHIA8_PATH=NotInstalled$/PYTHIA8_PATH=this_variable_is_not_used/" Source/make_opts
exit $status
