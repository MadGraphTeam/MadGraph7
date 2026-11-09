#!/usr/bin/env bash
# Source from the MadGraph checkout. Prefer the pinned dev shell; this offline
# fallback discovers already-installed tools without modifying reference trees.
TIRUST_MG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export TIRUST_MG_ROOT
export MPLCONFIGDIR="${MPLCONFIGDIR:-$TIRUST_MG_ROOT/.tirust-build/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
_tirust_find_tool() {
    local name="$1" pattern candidate
    shift
    command -v "$name" >/dev/null 2>&1 && return 0
    for pattern in "$@"; do
        for candidate in /nix/store/$pattern/bin/"$name"; do
            if [[ -x "$candidate" ]]; then
                PATH="$PATH:${candidate%/*}"
                export PATH
                return 0
            fi
        done
    done
    printf 'Missing %s; activate the project Nix development shell.\n' "$name" >&2
    return 1
}
_tirust_find_tool rustc '*-rustc-1.97.1' '*-rustc-1.*' || return 1
_tirust_find_tool cargo '*-cargo-1.97.1' '*-cargo-1.*' || return 1
_tirust_find_tool cc '*-gcc-wrapper-15.2.0' '*-gcc-wrapper-*' || return 1
_tirust_find_tool gcc '*-gcc-wrapper-15.2.0' '*-gcc-wrapper-*' || return 1
_tirust_find_tool c++ '*-gcc-wrapper-15.2.0' '*-gcc-wrapper-*' || return 1
_tirust_find_tool gfortran '*-gfortran-wrapper-15.2.0' '*-gfortran-wrapper-*' || return 1
_tirust_find_tool ar '*-binutils-wrapper-*' || return 1
_tirust_find_tool make '*-gnumake-4.4.1' '*-gnumake-*' || return 1
_tirust_find_tool m4 '*-gnum4-*' '*-m4-*' || return 1
_tirust_find_tool cmake '*-cmake-*' || return 1
_tirust_find_tool pkg-config '*-pkg-config-wrapper-*' || return 1
if ! command -v python3 >/dev/null 2>&1; then
    for _tirust_python in /nix/store/*-python3-3.13.*-env/bin/python3; do
        if [[ -x "$_tirust_python" ]] && "$_tirust_python" -c 'import numpy,scipy,matplotlib,pytest,six' >/dev/null 2>&1; then
            PATH="$PATH:${_tirust_python%/*}"
            export PATH
            break
        fi
    done
fi
_tirust_find_tool python3 '*-python3-3.13.*-env' '*-python3-3.*' || return 1
_tirust_find_tool typst '*-typst-0.15.1' '*-typst-*' || return 1
_tirust_find_tool pdftoppm '*-poppler-utils-*' '*-poppler-*' || return 1
_tirust_find_tool curl '*-curl-*' || return 1
_tirust_find_tool wget '*-wget-*' || return 1
export CARGO_HOME="${CARGO_HOME:-$TIRUST_MG_ROOT/.tirust-build/cargo-home}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$TIRUST_MG_ROOT/.tirust-build/target}"
export CARGO_BUILD_BUILD_DIR="${CARGO_BUILD_BUILD_DIR:-$TIRUST_MG_ROOT/.tirust-build/cargo-build}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$TIRUST_MG_ROOT/.tirust-build/cache}"
export SYMBOLICA_LOCK_DIR="${SYMBOLICA_LOCK_DIR:-$TIRUST_MG_ROOT/.tirust-build/symbolica-locks}"
export SYMBOLICA_HIDE_BANNER=1
export RUST_MIN_STACK=134217728
export RUST_TEST_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CC="${CC:-$(command -v cc)}"
export CXX="${CXX:-$(command -v c++)}"
export FC="${FC:-$(command -v gfortran)}"
mkdir -p "$CARGO_HOME" "$SYMBOLICA_LOCK_DIR"
for _tirust_font in /nix/store/*-libertinus-*/share/fonts /nix/store/*-dejavu-fonts-*/share/fonts; do
    if [[ -d "$_tirust_font" ]]; then
        TYPST_FONT_PATHS="${TYPST_FONT_PATHS:+$TYPST_FONT_PATHS:}$_tirust_font"
    fi
done
export TYPST_FONT_PATHS
unset _tirust_font _tirust_python
unset -f _tirust_find_tool
