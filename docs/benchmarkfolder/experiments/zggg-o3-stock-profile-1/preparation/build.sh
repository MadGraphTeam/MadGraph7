#!/bin/bash
set -euo pipefail
make -C "/shared/external/madloop_tirust/madgraph7_madloop_tirust/.tirust-build/zggg-o3-stock-profile-1/build/SAVEDTMP_CHECK_uux_zggg/Source/MODEL" -j1 flavor_couplings.o
make -C "/shared/external/madloop_tirust/madgraph7_madloop_tirust/.tirust-build/zggg-o3-stock-profile-1/build/SAVEDTMP_CHECK_uux_zggg/Source" -j2 all
make -C "/shared/external/madloop_tirust/madgraph7_madloop_tirust/.tirust-build/zggg-o3-stock-profile-1/build/SAVEDTMP_CHECK_uux_zggg/SubProcesses/P0_uux_zggg" -j2 check
