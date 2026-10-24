#!/usr/bin/env bash
# Build into ~/.cache/HEPtools every HEPTools component that check_heptools
# does not find complete there, with the same install commands as the
# warm_cache.yml jobs. Runs from a pristine copy of the checked-out tree, so
# the job's own tree is left as checked out.
set -u
HEP="$HOME/.cache/HEPtools"
CHECK="$GITHUB_WORKSPACE/.github/actions/check_heptools/check_heptools.sh"
export MAKEFLAGS="-j$(nproc)"
mkdir -p "$HEP"

# one build at a time per machine, should several runners share this home
exec 9> "$HEP.lock"
flock 9

complete() {
    HEPTOOLS_PREFIX="$HEP" CHECK_MODE=fatal GITHUB_OUTPUT= bash "$CHECK" "$1" > /dev/null 2>&1
}

SRC=""
prepare_src() {
    [ -n "$SRC" ] && return
    SRC=$(mktemp -d "${RUNNER_TEMP:-/tmp}/heptools-src.XXXXXX")
    git -C "$GITHUB_WORKSPACE" archive HEAD | tar -x -C "$SRC"
    cp "$SRC/input/.mg7_configuration_default.txt" "$SRC/input/mg7_configuration.txt"
    echo "heptools_install_dir=$HEP" >> "$SRC/input/mg7_configuration.txt"
    cp "$SRC/Template/LO/Source/.make_opts" "$SRC/Template/LO/Source/make_opts"
}

mg_install() {
    printf '%s\n' "$@" > "$SRC/cmd"
    (cd "$SRC" && ./bin/madgraph cmd)
}

build_lhapdf() {
    mg_install "install lhapdf6"
    cd "$HEP/lhapdf6_py3/share/LHAPDF" || return 1
    for SET in NNPDF23_nlo_as_0118_qed cteq6l1 NNPDF40_lo_as_01180 NNPDF40_nlo_as_01180 \
               NNPDF40_nlo_as_01180_nf_4 NNPDF40MC_lo_as_01180 NNPDF23_lo_as_0130_qed; do
        [ -d "$SET" ] && continue
        wget -q "https://lhapdfsets.web.cern.ch/current/$SET.tar.gz" \
            && tar -xzpf "$SET.tar.gz" && rm -f "$SET.tar.gz"
    done
}
build_boost()   { mg_install "install boost"; }
build_pythia8() { mg_install "install pythia8"; }
build_emela()   { mg_install "install eMELA"; }
build_contur()  { mg_install "install rivet" "install contur"; }
build_looptools() {
    # serial: the IREGI/CutTools makefiles do not declare their Fortran module
    # dependencies, so a parallel make fails on a missing .mod
    (cd "$SRC/vendor/IREGI/src" && MAKEFLAGS= make) || return 1
    (cd "$SRC/vendor/CutTools" && MAKEFLAGS= make) || return 1
    mkdir -p "$HEP/IREGI/src" "$HEP/CutTools/lib"
    cp "$SRC/vendor/IREGI/src/libiregi.a" "$HEP/IREGI/src/"
    cp "$SRC/vendor/CutTools/includects/libcts.a" "$SRC/vendor/CutTools/includects/mpmodule.mod" "$HEP/CutTools/lib"
    mg_install "install collier" "install ninja"
}

# dependency order: emela needs boost, contur needs pythia8 (hepmc)
for tool in lhapdf boost pythia8 emela contur looptools; do
    if complete "$tool"; then
        echo "$tool: already built on $RUNNER_NAME"
        continue
    fi
    echo "::group::building $tool on $RUNNER_NAME (first use on this machine)"
    prepare_src
    ( "build_$tool" )
    echo "::endgroup::"
    complete "$tool" || echo "::warning::HEPTools $tool could not be built on $RUNNER_NAME"
done

[ -n "$SRC" ] && rm -rf "$SRC"
exit 0
