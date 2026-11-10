#!/usr/bin/env bash
set -euo pipefail
source tools/tirust_env.sh
export TIRUST_CLEAN_WORK="$PWD/.tirust-build/qp-transport-install"
export TIRUST_NUMERICAL_LOCK="$PWD/.tirust-build/numerical-tests.lock"
export CARGO_HOME="$PWD/.tirust-build/cargo-home"
export CARGO_TARGET_DIR="$PWD/.tirust-build/target"
export CARGO_BUILD_BUILD_DIR="$PWD/.tirust-build/cargo-build"
export XDG_CONFIG_HOME="$TIRUST_CLEAN_WORK/xdg-config"
export XDG_CACHE_HOME="$TIRUST_CLEAN_WORK/xdg-cache"
export PYTHONPATH="$TIRUST_CLEAN_WORK/observer"
export PYTHONDONTWRITEBYTECODE=1
unset CARGO_NET_OFFLINE MADGRAPH_BASE RUSTC_WRAPPER RUSTC_WORKSPACE_WRAPPER
python3 "$TIRUST_CLEAN_WORK/mg/bin/madgraph" --plain "$TIRUST_CLEAN_WORK/install.mg5"
