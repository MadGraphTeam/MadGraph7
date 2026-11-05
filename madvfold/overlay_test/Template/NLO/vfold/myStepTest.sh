#!/bin/bash
# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Sep 2026) for project MadVFold (arXiv:2609.23500).

set -e
set -o pipefail

cd $(dirname $0)/..
PROC=$(basename $(pwd))

# Command line arguments
usage() {
  echo "Usage:   $0 --ref <REF<n>/<tag>> [--diffonly] [--tsts '<tst1 tst2...>'] [--ninja-and-collier] <step>"
  echo "         (--ref: results in Events/REF<n>/<tag>, e.g. --ref REF4/GOLD; no default)"
  echo "         (--diffonly: no launch, only diff and timing table of existing results in Events/REF<n>/<tag>)"
  echo "         (--ninja-and-collier: also run F1 with the other MadLoop reduction library, see myLaunch.sh)"
  echo "Example: $0 --ref REF4/GOLD step14s"
  exit 1
}
ref=
diffonly=0
tag=
tsts=
ninjaAndCollier=
while [ "$1" != "" ]; do
  if [ "$1" == "--ninja-and-collier" ]; then
    ninjaAndCollier=--ninja-and-collier; shift
  elif [ "$1" == "--ref" ] && [ "$2" != "" ] && [ "$ref" == "" ]; then
    ref=$2; shift; shift
  elif [ "$1" == "--diffonly" ] && [ "$diffonly" == "0" ]; then
    diffonly=1; shift
  elif [ "$1" == "--tsts" ] && [ "$2" != "" ] && [ "$tsts" == "" ]; then
    tsts=$2; shift; shift
  elif [ "$step" == "" ]; then
    step=$1; shift
  else
    usage
  fi
done

# Check if this is part of step8, step9, step10, step11 or a later step
if [ "${step}" == "--diffonly" ] || [ "${step}" == "--tsts" ] || [ "${step}" == "--ref" ]; then
  usage
elif [ "${step}" != "${step#step8}" ]; then
  ###stepN=step8
  tstsDef="n200f111k1 n20f841k1"
  launch="--launch-o"
elif [ "${step}" != "${step#step9}" ]; then
  ###stepN=step9
  tstsDef="n200f111k1 n200f111k32 n20f841k1 n20f841k32"
  launch="--launch-o-cudascan"
elif [ "${step}" != "${step#step10}" ]; then
  ###stepN=step10
  tstsDef="n200f111k1 n200f111k32 n200f111k32g1 n20f841k1 n20f841k32 n20f841k32g1"
  launch="--launch-o-cudascan"
elif [ "${step}" != "${step#step11}" ] || [[ "${step}" =~ ^step[0-9]+ ]]; then
  ###stepN=step11 (or later, e.g. step12x, step13x for the generated process directories)
  tstsDef="n200f111k32g1 n200f111k512g1 n200f841k1g1 n200f841k64g1"
  launch="--launch-o-cudascan"
else
  echo "ERROR! Invalid step=${step} must be of the form step<n>x (e.g. step8x, step11x, step13x)"
  exit 1
fi

# The reference set and the tag of the results directory Events/REF<n>/<tag> (step 14s: no default,
# e.g. REF4/GOLD; before, the tag was derived from the names of the developers' machines)
if [ "${ref}" == "" ]; then
  echo "ERROR! --ref REF<n>/<tag> is required (e.g. --ref REF4/GOLD)"
  usage
fi
if ! [[ "${ref}" =~ ^REF[0-9]*/[A-Za-z0-9_.-]+$ ]]; then
  echo "ERROR! Invalid --ref ${ref} must be REF<n>/<tag> (e.g. REF4/GOLD)"
  exit 1
fi
tag=${ref#*/}
ref=${ref%%/*}
if [ "${diffonly}" == "1" ] && [ ! -d Events/${ref}/${tag} ]; then
  echo "ERROR! Directory Events/${ref}/${tag} does not exist"
  exit 1
fi

# Define tsts if not yet defined
if [ "${tsts}" == "" ]; then tsts=${tstsDef}; fi

# Loop over all tests
status=0
START=`date`
echo "==============================================================================="
for tst in ${tsts}; do
  start=`date`
  refstep=${ref}/${tag}/${tst}/${step}
  outL=Events/${ref}/${tag}/${tst}/log_${step}_launch.txt
  outD=Events/${ref}/${tag}/${tst}/log_${step}_diff.txt
  table=Events/${ref}/${tag}/${tst}/timingTable_${step}.txt
  step2=step8m # NB: tst2 and step2 are not used at the moment (comparison to step8m/step8n disabled below)
  if [ "${tst}" == "n200f111k1" ]; then
    env="ENV_NEVENTS=200 ENV_FOLDING='1,1,1' ENV_VFOLD_KPOINT_BATCH=1"
    tst2=${tst}
  elif [ "${tst}" == "n200f111k32" ]; then
    env="ENV_NEVENTS=200 ENV_FOLDING='1,1,1' ENV_VFOLD_KPOINT_BATCH=32"
    tst2=${tst/k32/k1}
  elif [ "${tst}" == "n200f111k32g1" ]; then
    env="ENV_NEVENTS=200 ENV_FOLDING='1,1,1' ENV_VFOLD_KPOINT_BATCH=32 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k32g1/k1}
  elif [ "${tst}" == "n2000f111k32g1" ]; then
    # 10x more events
    env="ENV_NEVENTS=2000 ENV_FOLDING='1,1,1' ENV_VFOLD_KPOINT_BATCH=32 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k32g1/k1}
  elif [ "${tst}" == "n200f111k512" ]; then
    env="ENV_NEVENTS=200 ENV_FOLDING='1,1,1' ENV_VFOLD_KPOINT_BATCH=512"
    tst2=${tst/k512/k1}
  elif [ "${tst}" == "n200f111k512g1" ]; then
    env="ENV_NEVENTS=200 ENV_FOLDING='1,1,1' ENV_VFOLD_KPOINT_BATCH=512 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k512g1/k1}
  elif [ "${tst}" == "n200f111k1024g1" ]; then
    env="ENV_NEVENTS=200 ENV_FOLDING='1,1,1' ENV_VFOLD_KPOINT_BATCH=1024 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k1024g1/k1}
  elif [ "${tst}" == "n100f111k512g1" ]; then
    # Step 14p: 5x more events than n20f111k512g1, to compare to n100f841k64g1 (e.g. ggttg: about 2 virtual events)
    env="ENV_NEVENTS=100 ENV_FOLDING='1,1,1' ENV_VFOLD_KPOINT_BATCH=512 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k512g1/k1}
  elif [ "${tst}" == "n50f111k512g1" ]; then
    # Step 14s: 2x fewer events than n100f111k512g1, to compare to n50f841k64g1 (e.g. ggttg with the step 14t grids)
    env="ENV_NEVENTS=50 ENV_FOLDING='1,1,1' ENV_VFOLD_KPOINT_BATCH=512 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k512g1/k1}
  elif [ "${tst}" == "n20f111k512g1" ]; then
    # 10x fewer events: match n20f841 for eemumubbg tests
    env="ENV_NEVENTS=20 ENV_FOLDING='1,1,1' ENV_VFOLD_KPOINT_BATCH=512 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k512g1/k1}
  elif [ "${tst}" == "n2000f111k512g1" ]; then
    # 10x more events: the fixed GPU init cost becomes negligible and the virtual sample larger
    env="ENV_NEVENTS=2000 ENV_FOLDING='1,1,1' ENV_VFOLD_KPOINT_BATCH=512 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k512g1/k1}
  elif [ "${tst}" == "n200f111k32g1DELTA" ]; then
    env="ENV_NEVENTS=200 ENV_FOLDING='1,1,1' ENV_VFOLD_KPOINT_BATCH=32 ENV_VFOLD_SBORN_GSCALE=1 ENV_MCATNLODELTA=1"
    tst2=${tst/k32g1DELTA/k1}
  elif [ "${tst}" == "n200f111k512g1DELTA" ]; then
    env="ENV_NEVENTS=200 ENV_FOLDING='1,1,1' ENV_VFOLD_KPOINT_BATCH=512 ENV_VFOLD_SBORN_GSCALE=1 ENV_MCATNLODELTA=1"
    tst2=${tst/k512g1DELTA/k1}
  elif [ "${tst}" == "n20f841k1" ]; then
    env="ENV_NEVENTS=20 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=1"
    tst2=${tst}
  elif [ "${tst}" == "n20f841k32" ]; then
    env="ENV_NEVENTS=20 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=32"
    tst2=${tst/k32/k1}
  elif [ "${tst}" == "n20f841k32g1" ]; then
    env="ENV_NEVENTS=20 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=32 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k32g1/k1}
  elif [ "${tst}" == "n20f841k64g1" ]; then
    env="ENV_NEVENTS=20 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=64 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k64g1/k1}
  elif [ "${tst}" == "n20f841k128" ]; then
    env="ENV_NEVENTS=20 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=128"
    tst2=${tst/k128/k1}
  elif [ "${tst}" == "n20f841k128g1" ]; then
    env="ENV_NEVENTS=20 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=128 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k128g1/k1}
  elif [ "${tst}" == "n50f841k64g1" ]; then
    # Step 14s: 2x fewer events than n100f841k64g1, to compare to n50f111k512g1 (e.g. ggttg with the step 14t grids)
    env="ENV_NEVENTS=50 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=64 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k64g1/k1}
  elif [ "${tst}" == "n100f841k1" ]; then
    env="ENV_NEVENTS=100 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=1"
    tst2=${tst}
  elif [ "${tst}" == "n100f841k32g1" ]; then
    env="ENV_NEVENTS=100 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=32 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k32g1/k1}
  elif [ "${tst}" == "n100f841k64g1" ]; then
    env="ENV_NEVENTS=100 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=64 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k64g1/k1}
  elif [ "${tst}" == "n200f841k1" ]; then
    env="ENV_NEVENTS=200 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=1"
    tst2=${tst}
    step2=step8n
  elif [ "${tst}" == "n200f841k1g1" ]; then
    env="ENV_NEVENTS=200 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=1 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k1g1/k1}
    step2=step8n
  elif [ "${tst}" == "n200f841k16g1" ]; then
    env="ENV_NEVENTS=200 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=16 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k16g1/k1}
    step2=step8n
  elif [ "${tst}" == "n200f841k32g1" ]; then
    env="ENV_NEVENTS=200 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=32 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k32g1/k1}
    step2=step8n
  elif [ "${tst}" == "n100f841k64g1" ]; then
    # Step 14p: 5x more events than n20f841k64g1, to compare to n100f111k512g1 (e.g. ggttg: about 2 virtual events)
    env="ENV_NEVENTS=100 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=64 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k64g1/k1}
  elif [ "${tst}" == "n200f841k64g1" ]; then
    env="ENV_NEVENTS=200 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=64 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k64g1/k1}
    step2=step8n
  elif [ "${tst}" == "n200f841k128g1" ]; then
    env="ENV_NEVENTS=200 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=128 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k128g1/k1}
    step2=step8n
  elif [ "${tst}" == "n200f841k1g1DELTA" ]; then
    env="ENV_NEVENTS=200 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=1 ENV_VFOLD_SBORN_GSCALE=1 ENV_MCATNLODELTA=1"
    tst2=${tst/k1g1DELTA/k1}
    step2=step8n
  elif [ "${tst}" == "n200f841k64g1DELTA" ]; then
    env="ENV_NEVENTS=200 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=64 ENV_VFOLD_SBORN_GSCALE=1 ENV_MCATNLODELTA=1"
    tst2=${tst/k64g1DELTA/k1}
    step2=step8n
  elif [ "${tst}" == "n200f841k256g1" ]; then
    env="ENV_NEVENTS=200 ENV_FOLDING='8,4,1' ENV_VFOLD_KPOINT_BATCH=256 ENV_VFOLD_SBORN_GSCALE=1"
    tst2=${tst/k256g1/k1}
    step2=step8n
  else
    echo "ERROR! Unknown tst=${tst}" |& tee -a $outL
    exit 1
  fi
  #====================
  # PART 1: LAUNCH
  #====================
  if [ "$diffonly" == "0" ]; then
    mkdir -p $(dirname ${outL})
    echo | tee $outL
    echo "START:    $start" | tee -a $outL
    echo "HOSTNAME: $(hostname)" | tee -a $outL
    echo "PROC:     ${PROC}" | tee -a $outL
    echo | tee -a $outL
    echo "-------------------------------------------------------------------------------" | tee -a $outL
    echo | tee -a $outL
    echo "Launch ${PROC} tests on $(hostname):" | tee -a $outL
    cmd="./vfold/myBackends.sh ${ninjaAndCollier} ${launch} ${refstep}"
    echo "  ${env} \\" | tee -a $outL
    echo "    ${cmd}" | tee -a $outL
    echo | tee -a $outL
    if ! eval "${env} ${cmd}" |& tee -a $outL; then status=1; fi
    echo | tee -a $outL
    echo "-------------------------------------------------------------------------------" | tee -a $outL
    echo | tee -a $outL
    end=`date`
    echo "START:    $start" | tee -a $outL
    echo "END:      $end" | tee -a $outL
    echo "ELAPSED:  "$((`date +%s -d "$end"`-`date +%s -d"$start"`))" seconds" | tee -a $outL
    echo | tee -a $outL
    echo "-------------------------------------------------------------------------------" | tee -a $outL
    echo | tee -a $outL
    echo "STATUS=${status} (launch ${refstep})" | tee -a $outL
    echo | tee -a $outL
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
  fi
  #====================
  # PART 2: DIFF/TABLE
  #====================
  if ! ls -ld Events/${ref}/${tag}/${tst}/${step}* > /dev/null; then status=1; break; fi
  mkdir -p $(dirname ${outD})
  echo | tee $outD
  echo "ls -ld Events/${ref}/${tag}/${tst}/${step}*" | tee $outD
  ls -ld Events/${ref}/${tag}/${tst}/${step}* | tee $outD
  echo | tee -a $outD
  echo "-------------------------------------------------------------------------------" | tee -a $outD
  echo | tee -a $outD
  cmd="./vfold/myBackends.sh --diff-o ${refstep}"
  echo "Compare results using:" | tee -a $outD
  echo "  ${cmd}" | tee -a $outD
  echo | tee -a $outD
  if ! ${cmd} |& tee -a $outD; then status=1; fi
  echo | tee -a $outD
  echo "-------------------------------------------------------------------------------" | tee -a $outD
  echo | tee -a $outD
  # The comparison to the reference ${ref}/${tst2}/${step2} (step8m/step8n) is NOT supported at the moment:
  # these references do not exist in REF3, and myDiffRun.sh only prints 'Directory not found' for them,
  # so the comparison was a silent no-op that still reported 'identical' (tst2 and step2 are kept for later)
  cmd="./vfold/myBackends.sh --diff2-o ${ref}/${tst2}/${step2} ${refstep}"
  echo "SKIPPED (not supported at the moment, references do not exist in ${ref}):" | tee -a $outD
  echo "  ${cmd}" | tee -a $outD
  echo | tee -a $outD
  echo "-------------------------------------------------------------------------------" | tee -a $outD
  echo | tee -a $outD
  cmd="./vfold/myTimingTable.py -f txt Events/${refstep}"
  echo "Prepare timing table using:" | tee -a $outD
  echo "  ${cmd}" | tee -a $outD
  echo | tee -a $outD
  ${cmd} |& tee -a $outD # this writes the table to a file (only errors to stdout)
  cat $(dirname Events/${refstep})/timingTable_$(basename ${refstep}).txt | tee -a $outD
  echo | tee -a $outD
  echo "-------------------------------------------------------------------------------" | tee -a $outD
  echo | tee -a $outD
  echo "STATUS=${status} (diff ${refstep})" | tee -a $outD
  echo | tee -a $outD
  echo "==============================================================================="
done
echo
END=`date`
echo "START:    $START"
echo "END:      $END"
echo "ELAPSED:  "$((`date +%s -d "$END"`-`date +%s -d"$START"`))" seconds"
echo
echo "STATUS=${status} (overall)"
