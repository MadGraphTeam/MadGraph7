#!/bin/bash -l
# Runs on the self-hosted runner (GPU node): job madspace_ops of gpu_runner_ci.yml.
# madspace GPU runtime operations against the CPU runtime: see madspace_checks.py.
# Environment: as pp_ttx_mg7.sh (BACKEND, MODULES, MADSPACE_PREFIX, WORKDIR).
# torch (from the PyTorch module) puts the inputs on the GPU. This runs the python3 of
# the modules, not the venv of build_madspace (same interpreter, which madspace was built
# with): EasyBuild finds some packages of the modules (e.g. typing_extensions of
# Python-bundle-PyPI, which torch imports) through EBPYTHONPREFIXES, which only the
# python3 of the modules reads (its sitecustomize), not a venv made from it.
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
section() { echo; echo "=================== $* ($(date +%T))"; }

section "Environment"
if [ -n "$MODULES" ]; then source "$HERE/load_modules.sh"; fi
python3 --version
case $BACKEND in
    cuda) GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p) ;;
    hip)  GPU_NAME=$(rocminfo 2> /dev/null | awk -F': *' '/Marketing Name/ {name=$2} /Device Type: *GPU/ {print name; exit}' || true) ;;
    *)    echo "::error::the madspace checks need a GPU backend, not '$BACKEND'"; exit 2 ;;
esac
export GPU_NAME
export PYTHONPATH=$MADSPACE_PREFIX${PYTHONPATH:+:$PYTHONPATH}

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
STATUS=0
python3 "$HERE/madspace_checks.py" 2>&1 | tee checks.log || STATUS=1

section "Summary"
cat summary.txt 2> /dev/null || echo "no summary.txt"
exit $STATUS
