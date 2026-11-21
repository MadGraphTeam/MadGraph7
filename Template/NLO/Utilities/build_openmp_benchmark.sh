#!/usr/bin/env bash
# Build the historical lane-parallel amplitude loop for benchmark studies.
# This is deliberately not a production thread-safety claim.

set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
jobs=${JOBS:-1}

if [[ ! -f "$root/Source/make_opts" ]]; then
    echo "error: $root is not a generated NLO output" >&2
    exit 1
fi

# DHELAS is in the OpenMP amplitude call graph.  Rebuild it with automatic
# local storage while leaving MODEL and the other legacy libraries untouched.
make -C "$root/Source/DHELAS" clean
make -C "$root/Source/DHELAS" -j "$jobs" openmp=true all

if (( $# )); then
    process_dirs=("$@")
else
    process_dirs=("$root"/SubProcesses/P*)
fi

for process_dir in "${process_dirs[@]}"; do
    if [[ ! -f "$process_dir/makefile" ]]; then
        echo "error: no subprocess makefile in $process_dir" >&2
        exit 1
    fi

    # The subprocess makefile applies -frecursive only to these amplitude
    # objects.  Force their rebuild, plus the OpenMP driver and final link.
    rm -f "$process_dir"/driver_vec.o \
          "$process_dir"/real_me_chooser.o \
          "$process_dir"/born.o \
          "$process_dir"/matrix_*.o \
          "$process_dir"/driver_mintMC.o \
          "$process_dir"/madevent_mintMC
    make -C "$process_dir" openmp=true madloop=true madevent_mintMC
done
