#!/bin/bash
# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Oct 2026) for project MadVFold (arXiv:2609.23500).

# Generate a process directory with the current mg5amcnlo code (output only,
# no launch), by default with 'output <gendir> --vfold-with-tests'.
# Compare it to another directory with compareProc.sh.

set -e
set -o pipefail

usage() {
  echo "Usage:   $0 [-opt '<output options>'] <proc> [<gendir>]"
  echo "         (default options: --vfold-with-tests; default gendir: madvfold/DEV/<proc>_vfgen)"
  echo "Example: $0 eebbbb"
  echo "Example: $0 -opt '' eebbbb /tmp/eebbbb_stock"
  exit 1
}
mg5dir=$(cd $(dirname $0)/../..; pwd -P)
opts="--vfold-with-tests"
while [ "$1" != "" ]; do
  if [ "$1" == "-opt" ] && [ "$#" -ge 2 ]; then opts="$2"; shift 2
  elif [ "${1:0:1}" == "-" ]; then usage
  else break; fi
done
if [ "$1" == "" ] || [ "$3" != "" ]; then usage; fi
proc=$1
gen=${2:-madvfold/DEV/${proc}_vfgen}

case ${proc} in
  eebb) gencmd="generate e+ e- > b b~ [QCD]";;
  eebbbb) gencmd="generate e+ e- > b b~ b b~ [QCD]";;
  eemumubb) gencmd="generate e+ e- > mu+ mu- b b~ [QCD]";;
  eemumubbg) gencmd="generate e+ e- > mu+ mu- b b~ g [QCD]";;
  ggbbg) gencmd="generate g g > b b~ g [QCD]";;
  ggttg) gencmd="generate g g > t t~ g [QCD]";;
  *) echo "ERROR! Unknown proc=${proc}"; exit 1;;
esac

cd ${mg5dir}
log=${gen}.log
echo "Generate ${gen} with 'output ${gen} ${opts}' (log in ${log})"
rm -rf ${gen}
start=$(date)
echo "START:    ${start}" > ${log}
echo "HOSTNAME: $(hostname)" >> ${log}
cat > ${gen}.mg <<EOF2
set automatic_html_opening False --no_save
set pythia8_path ./HEPTools/pythia8 --no_save
set mg5amc_py8_interface_path ./HEPTools/MG5aMC_PY8_interface --no_save
set hepmc_path ./HEPTools/hepmc --no_save
set lhapdf_py3 ./HEPTools/lhapdf6_py3/bin/lhapdf-config --no_save
set ninja ./HEPTools/lib --no_save
set collier ./HEPTools/collier --no_save
${gencmd}
output ${gen} ${opts}
EOF2
./bin/madgraph ${gen}.mg >> ${log} 2>&1 || { echo "ERROR! generation failed, see ${log}"; exit 1; }
rm -f ${gen}.mg
end=$(date)
echo "START:    ${start}" >> ${log}
echo "END:      ${end}" >> ${log}
echo "ELAPSED:  $(($(date +%s -d "${end}")-$(date +%s -d "${start}"))) seconds" >> ${log}
# Keep the generation log in the generated directory
mv ${log} ${gen}/GENLOG.txt
echo "Generation completed (log in ${gen}/GENLOG.txt)"
