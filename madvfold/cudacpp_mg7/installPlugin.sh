#!/bin/bash
# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Nov 2026) for project MadVFold (arXiv:2609.23500).

# Install the cudacpp plugin of madgraph4gpu (branch vfold7, tarball as in
# madvfold/README_CUDACPP.md, e.g. that of the MadVFold release
# valassi_cudacpp_for3.8.1_v1.02.00_vfold7on20261112) in PLUGIN/CUDACPP_OUTPUT
# for use in MadGraph7.
# The MadGraph7 modules are NOT modified, so madmatrix is not affected: the plugin
# adds the GPU classes removed by MadGraph7 (export_cpp_gpu.py) to its own copy of
# the MadGraph7 export_cpp module, and uses its own copy of the mg5amcnlo v3.8.1
# aloha_writers.py: both are loaded only by the plugin.
# Usage: madvfold/cudacpp_mg7/installPlugin.sh [<cudacpp.tar.gz>] (default PLUGIN/cudacpp.tar.gz)
set -e
cd $(dirname $0)/../..
tarball=${1:-PLUGIN/cudacpp.tar.gz}
P=PLUGIN/CUDACPP_OUTPUT
command rm -rf ${P}
tar xzf ${tarball} -C PLUGIN
command cp -f madvfold/cudacpp_mg7/export_cpp_gpu.py ${P}/
command cp -f madvfold/cudacpp_mg7/aloha_writers_mg5v381.py ${P}/
python3 madvfold/cudacpp_mg7/patchPlugin.py ${P}
echo "INFO: cudacpp plugin installed in ${P}"
