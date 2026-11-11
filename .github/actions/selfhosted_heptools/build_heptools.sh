#!/usr/bin/env bash
# usage: build_heptools.sh [component ...]   (default: all of them)
# Build into ~/.cache/HEPtools every requested HEPTools component that check_heptools
# does not find complete there, with the same install commands as the
# warm_cache.yml jobs. Runs from a pristine copy of the checked-out tree, so
# the job's own tree is left as checked out.
set -u
HEP="$HOME/.cache/HEPtools"
CHECK="$GITHUB_WORKSPACE/.github/actions/check_heptools/check_heptools.sh"
# The self-hosted machines have no swap and little memory (6 GB on the alma10
# "large" ones; a -j$(nproc) rivet build got the runner service OOM-killed on
# the small ones): one compile per 2.5 GB available, at most one per core.
avail_mb=$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)
jobs=$(( avail_mb / 2500 ))
[ "$jobs" -gt "$(nproc)" ] && jobs=$(nproc)
[ "$jobs" -lt 1 ] && jobs=1
export MAKEFLAGS="-j$jobs"
echo "building with $MAKEFLAGS (${avail_mb} MB available, $(nproc) cores)"
mkdir -p "$HEP"

# one build at a time per machine, should several runners share this home
exec 9> "$HEP.lock"
flock 9

complete() {
    HEPTOOLS_PREFIX="$HEP" CHECK_MODE=fatal GITHUB_OUTPUT= bash "$CHECK" "$1" > /dev/null 2>&1 \
        || return 1
    # here also YODA's Python module, which rivet-mkhtml imports: a YODA build
    # whose Python extension failed still installs libYODA
    [ "$1" != contur ] || yoda_pyext
}
yoda_pyext() { compgen -G "$HEP/yoda/lib*/python3*/*-packages/yoda/core*.so" > /dev/null; }

check_prereqs() {
    # Build prerequisites the HEPTools installers expect from the system (the
    # runners have no sudo: they must be installed as root, e.g. on AlmaLinux
    #   dnf install -y cmake wget bzip2 patch zlib-devel autoconf automake libtool
    # ). Say which are missing up front instead of failing deep in a configure.
    missing=""
    for c in cmake wget curl bzip2 patch autoconf automake libtoolize make gfortran g++ flock; do
        command -v "$c" > /dev/null || missing="$missing $c"
    done
    [ -e /usr/include/zlib.h ] || missing="$missing zlib-devel(zlib.h)"
    if [ -n "$missing" ]; then
        echo "::warning::$RUNNER_NAME ($(hostname)) lacks build prerequisites:$missing; HEPTools components may fail to build"
    fi
}

SRC=""
prepare_src() {
    [ -n "$SRC" ] && return
    check_prereqs
    SRC=$(mktemp -d "${RUNNER_TEMP:-/tmp}/heptools-src.XXXXXX")
    git -C "$GITHUB_WORKSPACE" archive HEAD | tar -x -C "$SRC"
    cp "$SRC/input/.mg7_configuration_default.txt" "$SRC/input/mg7_configuration.txt"
    echo "heptools_install_dir=$HEP" >> "$SRC/input/mg7_configuration.txt"
    cp "$SRC/Template/LO/Source/.make_opts" "$SRC/Template/LO/Source/make_opts"
}

mg_install() {
    printf '%s\n' "$@" > "$SRC/cmd"
    # `install` into this external prefix records the tool paths in the
    # per-user config, and a later install (maybe in a later job) looks them up
    # there: keep that config with the build, on this machine, apart from the
    # per-job XDG_CONFIG_HOME the tests get from clean_tmp.
    (cd "$SRC" && XDG_CONFIG_HOME="$HEP.build-config" ./bin/madgraph cmd)
}

# the same sets as warm_cache.yml (lhapdf_cache)
PDFSETS="NNPDF23_nlo_as_0118_qed cteq6l1 NNPDF40_lo_as_01180 NNPDF40_nlo_as_01180
         NNPDF40_nlo_as_01180_nf_4 NNPDF40_nlo_as_01180_qed NNPDF40MC_lo_as_01180
         NNPDF23_lo_as_0130_qed"
fetch_pdfsets() {
    bash "$GITHUB_WORKSPACE/.github/actions/restore_heptools_lhapdf/fetch_pdfsets.sh" \
        "$HEP/lhapdf6_py3/share/LHAPDF" $PDFSETS
}
build_lhapdf() {
    mg_install "install lhapdf6"
    fetch_pdfsets
}
build_boost()   { mg_install "install boost"; }
build_pythia8() { mg_install "install pythia8"; }
build_emela()   { mg_install "install eMELA"; }
build_contur()  {
    # YODA's python extension needs Cython (see warm_cache.yml contur_cache);
    # no sudo here, so into the user site of the interpreter madgraph uses
    python3 -c 'import Cython' 2> /dev/null || python3 -m pip install --user Cython
    # ... and two commands from PATH, both there on the hosted runners: the
    # `cython` that regenerates core.cpp (pip --user puts it in ~/.local/bin,
    # not on this runner service's PATH), and an unversioned `python` for its
    # `#!/usr/bin/env python` template script (AlmaLinux only has python3).
    # Without them YODA's Python extension failed while libYODA still
    # installed (rivet-mkhtml: "No module named 'yoda'").
    mkdir -p "$HEP.bin"
    ln -sfn "$(command -v python3)" "$HEP.bin/python"
    export PATH="$HEP.bin:$(python3 -m site --user-base)/bin:$PATH"
    if [ -d "$HEP/yoda" ] && ! yoda_pyext; then
        # rebuild YODA alone (rivet does not, YODA being found): its installer
        # untars over an existing tree, so start from none
        rm -rf "$HEP/yoda"
        mg_install "install yoda"
    fi
    if ! yoda_pyext && [ -f "$HEP/yoda/yoda_install.log" ]; then
        # its installer reports success all the same: show why
        echo "::warning::YODA built without its Python module on $RUNNER_NAME"
        grep -i -E 'cython|python|error' "$HEP/yoda/yoda_install.log" | tail -30
    fi
    HEPTOOLS_PREFIX="$HEP" CHECK_MODE=fatal GITHUB_OUTPUT= bash "$CHECK" contur > /dev/null 2>&1 \
        || mg_install "install rivet" "install contur"
}
build_looptools() {
    # serial: the IREGI/CutTools makefiles do not declare their Fortran module
    # dependencies, so a parallel make fails on a missing .mod
    (cd "$SRC/vendor/IREGI/src" && MAKEFLAGS= make) || return 1
    (cd "$SRC/vendor/CutTools" && MAKEFLAGS= make) || return 1
    mkdir -p "$HEP/IREGI/src" "$HEP/CutTools/lib"
    cp "$SRC/vendor/IREGI/src/libiregi.a" "$HEP/IREGI/src/"
    cp "$SRC/vendor/CutTools/includects/libcts.a" "$SRC/vendor/CutTools/includects/mpmodule.mod" "$HEP/CutTools/lib"
    compgen -G "$HEP/collier/lib*/libcollier.*" > /dev/null \
        || compgen -G "$HEP/collier/libcollier.*" > /dev/null \
        || mg_install "install collier"
    # ninja serially too: with -j, src/Makefile asks for quadninjago_module.mod
    # before the compile that writes it ("No rule to make target"). Drop what a
    # failed attempt left: its installer untars over an existing tree.
    if ! compgen -G "$HEP/ninja/lib*/libninja.*" > /dev/null; then
        rm -rf "$HEP/ninja"
        (export MAKEFLAGS=; mg_install "install ninja")
    fi
}

recipe=$(sha1sum "${BASH_SOURCE[0]}" | cut -c1-12)

# dependency order: emela needs boost, contur needs pythia8 (hepmc)
requested=" ${*:-lhapdf boost pythia8 emela contur looptools} "
for tool in lhapdf boost pythia8 emela contur looptools; do
    [[ "$requested" == *" $tool "* ]] || continue
    if complete "$tool"; then
        echo "$tool: already built on $RUNNER_NAME"
        # the PDF sets are checked every time: a build cancelled while
        # fetching them still passes the check above
        [ "$tool" = lhapdf ] && fetch_pdfsets
        continue
    fi
    # A component that failed to build is not retried for 12 h: the build of
    # rivet/contur alone takes the better part of an hour, and every job of
    # this machine would pay it again until the cause (typically a missing
    # system package) is fixed. Delete ~/.cache/HEPtools.failed to retry now.
    # A change to this script (a fixed recipe) retries at once: the marker
    # records the script's hash.
    marker="$HEP.failed/$tool"
    if [ -n "$(find "$marker" -mmin -720 2>/dev/null)" ] && grep -qs "recipe $recipe" "$marker"; then
        echo "::warning::HEPTools $tool failed to build on $RUNNER_NAME less than 12 h ago, not retrying (see $marker)"
        continue
    fi
    echo "::group::building $tool on $RUNNER_NAME (first use on this machine)"
    prepare_src
    ( "build_$tool" )
    echo "::endgroup::"
    if complete "$tool"; then
        rm -f "$marker"
    else
        mkdir -p "$HEP.failed"
        echo "$(date -u) $GITHUB_REPOSITORY run $GITHUB_RUN_ID recipe $recipe" > "$marker"
        echo "::warning::HEPTools $tool could not be built on $RUNNER_NAME"
    fi
done

[ -n "$SRC" ] && rm -rf "$SRC"
exit 0
