#!/bin/bash
# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Aug 2026) for project MadVFold (arXiv:2609.23500).

cd $(dirname $0)/..

usage(){
  echo "Usage: $0 [-p prefix] [-o] [-nd3|-nd5|-linkDumps] [-nolhe|-vs363] [-xihat] [-dedupJacpdf] [Events/]run_A [Events/]run_B"
  exit 1
}

oOpt=
numdiff=0
lhe=1
xihat=0
dedupjac=0
prefix=
while [ "$3" != "" ]; do
  if [ "$1" == "-p" ]; then
    prefix="$2"; shift; shift
  elif [ "$1" == "-o" ] && [ "${oOpt}" == "" ]; then
    oOpt=-o; shift
  elif [ "$1" == "-nd5" ] && [ "${numdiff}" == "0" ]; then
    numdiff=5; shift
  elif [ "$1" == "-nd3" ] && [ "${numdiff}" == "0" ]; then
    numdiff=3; shift
  elif [ "$1" == "-linkDumps" ] && [ "${numdiff}" == "0" ]; then
    numdiff=-1; shift
  elif [ "$1" == "-nolhe" ] && [ "${lhe}" == "1" ]; then
    lhe=0; shift
  elif [ "$1" == "-vs363" ] && [ "${lhe}" == "1" ]; then
    lhe=-1; shift
  elif [ "$1" == "-xihat" ] && [ "${xihat}" == "0" ]; then
    xihat=1; shift
  elif [ "$1" == "-dedupJacpdf" ] && [ "${dedupjac}" == "0" ]; then
    dedupjac=1; shift
  else
    usage
  fi
done
if [ "$2" == "" ] || [ "$1" == "$2" ]; then usage; fi
if [ "${prefix}" == "" ]; then
  if [ "${1#Events/}" == "$1" ]; then runA=Events/$1; else runA=$1; fi
  if [ "${2#Events/}" == "$2" ]; then runB=Events/$2; else runB=$2; fi
else
  runA=Events/${prefix}/$1
  runB=Events/${prefix}/$2
fi
shift
shift
runA=${runA%/}
runB=${runB%/}
if [ ! -d ${runA} ]; then echo "Directory not found: ${runA}"; exit 1; fi
if [ ! -d ${runB} ]; then echo "Directory not found: ${runB}"; exit 1; fi

# Paths for linking dump files
up=
if [ "${numdiff}" == "-1" ]; then
  numdiff=0
  if [ "$(cd ${runB}; basename $(dirname $(pwd)))" == "Events" ]; then
    up=..
  elif [ "$(cd ${runB}; basename $(dirname $(cd ..; pwd)))" == "Events" ]; then
    up=../..
  elif [ "$(cd ${runB}; basename $(dirname $(cd ../..; pwd)))" == "Events" ]; then
    up=../../..
  else
    echo "WARNING! Cannot decode path ${runB} for linking dump files"
  fi
fi

# Files to diff (metadata files: plain diff)
txtfilesNoFail="VERSIONS.txt"

# Files to diff (result files: stricter tolerance)
if [ "$oOpt" != "-o" ]; then mints=$(seq 0 2); else mints="2"; fi
resfiles=""  
for mint in ${mints}; do resfiles="res_${mint}.txt $resfiles"; done
resfiles="summary.txt ${resfiles}"
resfiles="nevents_unweighted results.dat ${resfiles}"

# Files to diff (LHE result files: stricter tolerance - and fix indentation if required)
# (ignore the GIT line of the banner: the two runs may come from different branches)
if [ "${lhe}" == "1" ]; then
  grep -v '^#\* *GIT ' ${runA}/events.lhe > ${runA}/events.lhe.tmp
  grep -v '^#\* *GIT ' ${runB}/events.lhe > ${runB}/events.lhe.tmp
  resfiles="events.lhe.tmp ${resfiles}"
elif [ "${lhe}" == "-1" ]; then # cross-compare v3.8.1 and v3.6.3 LHE files
  cat ${runA}/events.lhe \
    | awk -v ok=0 '/<event>/{ok=1;}; {if(ok==1) print $0}' \
    | sed "s|  </event>|</event>|" \
    | sed "s|  <event>|<event>|" \
    > ${runA}/events.lhe.tmp
  cat ${runB}/events.lhe \
    | awk -v ok=0 '/<event>/{ok=1;}; {if(ok==1) print $0}' \
    | sed "s|  </event>|</event>|" \
    | sed "s|  <event>|<event>|" \
    > ${runB}/events.lhe.tmp
  resfiles="events.lhe.tmp ${resfiles}"
fi

# Files to diff (dump files: looser tolerance)
if [ "$oOpt" != "-o" ]; then
  dmpfilesA="$(cd ${runA}; ls dump*.dat)"
  dmpfilesB="$(cd ${runB}; ls dump*.dat)"
  if [ "${dmpfilesA}" != "${dmpfilesB}" ]; then
    echo "ERROR! Lists of dumpfiles differ"
    echo ${dmpfilesA}
    echo ${dmpfilesB}
    exit 1
  fi
  dmpfiles=${dmpfilesA}
else
  dmpfiles=
fi

# Diff files
status=0
for file in $txtfilesNoFail; do
  ###echo "=== diff ${runA}/${file} ${runB}/${file}"
  ######if ! diff ${runA}/${file} ${runB}/${file}; then status=1; fi
  ###diff ${runA}/${file} ${runB}/${file} # do not fail on VERSIONS.txt differences
  cat ${runA}/${file} | egrep -v '^(FC=|f90cache is not installed|nvcc is not installed)$' > ${runA}/${file}.tmp
  cat ${runB}/${file} | egrep -v '^(FC=|f90cache is not installed|nvcc is not installed)$' > ${runB}/${file}.tmp
  echo "=== diff ${runA}/${file}.tmp ${runB}/${file}.tmp"
  diff ${runA}/${file}.tmp ${runB}/${file}.tmp # do not fail on VERSIONS.txt differences
  rm ${runA}/${file}.tmp
  rm ${runB}/${file}.tmp
done
for file in $resfiles; do
  if [ "${lhe}" == "-1" ] && [ "${file}" == "results.dat" ]; then continue; fi # ignore diffs in MINT2-only 381/363 comparisons
  if [ "${lhe}" == "-1" ] && [ "${file}" == "res_0.txt" ]; then continue; fi # no res_0.txt in MINT2-only 381/363 comparisons
  if [ "${numdiff}" != "0" ]; then
    # Step 11s (MINT0 vectorization): increase tolerance from 1e-7 to 1.2e-7
    # Note that this only applies to -nd5/-nd3 comparisons (i.e. only to specific C/simd and C/cuda tests)
    # Step 14f: -nd3 uses the same loose tolerance as for its dumps (e.g. eemumubbg: MadLoop amplifies rounding)
    if [ "${numdiff}" == "3" ]; then rtol=1e-3; else rtol=1.2e-7; fi
    echo "=== python3 vfold/numdiff.py ${runA}/${file} ${runB}/${file} --rtol ${rtol} --atol 0"
    if ! python3 vfold/numdiff.py ${runA}/${file} ${runB}/${file} --rtol ${rtol} --atol 0; then status=1; fi
  else
    echo "=== diff ${runA}/${file} ${runB}/${file}"
    if ! diff ${runA}/${file} ${runB}/${file}; then status=1; fi
  fi
done
if [ "${lhe}" != "0" ]; then rm -f ${runA}/events.lhe.tmp ${runB}/events.lhe.tmp; fi
# Step 11l (PS memo): the n+1 generate_momenta call is skipped, so its JACPDF* dump lines disappear
# With -dedupJacpdf, drop any JACPDF* block that repeats the previous one within the same NEWPS point
# (bit-identical lines only: this is what a memo hit removes; dumps from step 11l onwards are unchanged)
dedupJacpdf(){
  awk '
    function flush(){ if (blk != "") { if (blk != prev) { printf "%s", blk; prev = blk }; blk = "" } }
    $1 ~ /^JACPDF/ { blk = blk $0 "\n"; next }
    { flush(); if ($1 == "NEWPS") prev = ""; print }
    END { flush() }'
}
for file0 in $dmpfiles; do
  if [ "${xihat}" == "0" ] || [ "${dedupjac}" == "1" ]; then
    for run in ${runA} ${runB}; do
      cat ${run}/${file0} > ${run}/${file0}.tmp
      if [ "${xihat}" == "0" ]; then grep -v XIHAT ${run}/${file0}.tmp > ${run}/${file0}.tmp2; mv ${run}/${file0}.tmp2 ${run}/${file0}.tmp; fi
      if [ "${dedupjac}" == "1" ]; then dedupJacpdf < ${run}/${file0}.tmp > ${run}/${file0}.tmp2; mv ${run}/${file0}.tmp2 ${run}/${file0}.tmp; fi
    done
    file=${file0}.tmp
  else
    file=${file0}
  fi
  # NB: rtol for dumps was 1e-5 until run_12* included
  # NB: rtol for dumps becomes 1e-2 as of run_13* included (UNWGT discrepancies)
  if [ "${numdiff}" == "3" ]; then
    echo "=== python3 vfold/numdiff.py ${runA}/${file} ${runB}/${file} --rtol 1e-3 --atol 0"
    if ! python3 vfold/numdiff.py ${runA}/${file} ${runB}/${file} --rtol 1e-3 --atol 0; then status=1; fi
  elif [ "${numdiff}" == "5" ]; then
    echo "=== python3 vfold/numdiff.py ${runA}/${file} ${runB}/${file} --rtol 1e-5 --atol 0"
    if ! python3 vfold/numdiff.py ${runA}/${file} ${runB}/${file} --rtol 1e-5 --atol 0; then status=1; fi
  elif [ "${up}" != "" ]; then
    echo "=== diff ${runA}/${file0} ${runB}/${file0}"
    if ! diff ${runA}/${file0} ${runB}/${file0}; then
      status=1
    elif [ ! -L ${runB}/${file0} ]; then # DO NOT CHECK [ ! -L ${runA}/${file0} ]
      echo "=== ln -sf ${up}/${runA#Events/}/${file0} ${runB}/${file0}"
      ln -sf ${up}/${runA#Events/}/${file0} ${runB}/${file0}
    fi
  else
    echo "=== diff ${runA}/${file} ${runB}/${file}"
    if ! diff ${runA}/${file} ${runB}/${file}; then status=1; fi
  fi
  if [ "${xihat}" == "0" ] || [ "${dedupjac}" == "1" ]; then
    rm ${runA}/${file0}.tmp
    rm ${runB}/${file0}.tmp
  fi
done
echo "STATUS=${status}"
echo
exit ${status}
