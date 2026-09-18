#!/bin/bash -l
# Runs on the self-hosted runner (GPU node): job pp_ttx of gpu_runner_ci.yml.
# generate p p > t t~, output mg7, run it on the GPU (run_card device = cuda|hip)
# and check that a cross section comes out.
# Environment:
#   BACKEND          cuda | hip   (cpu also works, to try the script on a laptop)
#   MODULES          modules to load (see load_modules.sh)
#   GPU_ARCH         optional, forwarded to the matrix-element build
#   VENV             python environment, and
#   MADSPACE_PREFIX  madspace install, both made by build_madspace_gpu.sh
#   WORKDIR          run directory: process, logs and summary.txt
#   NEVENTS          number of unweighted events
#   PDF_SET          LHAPDF set of the run, downloaded into $CACHE_DIR/lhapdf if needed
#   CACHE_DIR        cache root on the cluster
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
section() { echo; echo "=================== $* ($(date +%T))"; }

section "Environment"
if [ -n "$MODULES" ]; then source "$HERE/load_modules.sh"; fi
source "$VENV/bin/activate"
python3 --version
case $BACKEND in
    cuda) nvidia-smi
          GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p) ;;
    hip)  rocm-smi || true
          GPU_NAME=$(rocminfo 2> /dev/null | awk -F': *' '/Marketing Name/ {name=$2} /Device Type: *GPU/ {print name; exit}' || true) ;;
    cpu)  GPU_NAME=none ;;
    *)    echo "::error::unknown BACKEND '$BACKEND'"; exit 2 ;;
esac
if [ -n "$GPU_ARCH" ]; then
    export MADGRAPH_CUDA_ARCHITECTURE=$GPU_ARCH MADGRAPH_HIP_ARCHITECTURE=$GPU_ARCH
fi

# madspace goes where the mg7 runtime looks for it: <MadGraph7>/madspace/install
rm -rf "$REPO/madspace/install"
ln -s "$MADSPACE_PREFIX" "$REPO/madspace/install"
# default configuration, as .github/actions/checkout_mg5 does for the other CI jobs
cp "$REPO/input/.mg7_configuration_default.txt" "$REPO/input/mg7_configuration.txt"
cp "$REPO/Template/LO/Source/.make_opts" "$REPO/Template/LO/Source/make_opts"
cat >> "$REPO/input/mg7_configuration.txt" << EOF
auto_update = 0
automatic_html_opening = False
notification_center = False
nb_core = ${SLURM_CPUS_PER_TASK:-4}
EOF

export LHAPDF_DATA_PATH=$CACHE_DIR/lhapdf
if [ ! -d "$LHAPDF_DATA_PATH/$PDF_SET" ]; then
    echo "Downloading the PDF set $PDF_SET"
    mkdir -p "$LHAPDF_DATA_PATH"
    curl -fsSL --retry 3 "https://lhapdfsets.web.cern.ch/current/$PDF_SET.tar.gz" | tar -xz -C "$LHAPDF_DATA_PATH"
fi

section "Generating p p > t t~"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
cat > pp_ttx.mg7 << EOF
generate p p > t t~
output mg7 PROC_pp_ttx
EOF
cat pp_ttx.mg7
python3 "$REPO/bin/madgraph" pp_ttx.mg7 2>&1 | tee output.log

CARD=PROC_pp_ttx/Cards/run_card.toml
sed -i.orig -E -e "s/^device = .*/device = [\"$BACKEND\"]/" \
               -e "s/^events = [0-9]+/events = $NEVENTS/" \
               -e "s/^pdf = \".*\"$/pdf = \"$PDF_SET\"/" "$CARD"
grep -n -E '^(device|events|pdf) =' "$CARD"

section "Running on $BACKEND"
START=$SECONDS
STATUS=0
(cd PROC_pp_ttx && python3 bin/generate_events -f) 2>&1 | tee generate_events.log || STATUS=1
WALLTIME=$((SECONDS - START))

section "Checks"
INFO=$(ls -t PROC_pp_ttx/Events/*/info.json 2> /dev/null | sed -n 1p || true)
XSEC=
if [ -n "$INFO" ]; then
    XSEC=$(python3 - "$INFO" << 'EOF' || true
import json, math, sys
proc = json.load(open(sys.argv[1]))["process"]
mean, err = float(proc["mean"]), float(proc.get("error") or 0.0)
if mean > 0 and math.isfinite(mean):
    print(f"{mean:.6g} +- {err:.2g} pb")
EOF
)
fi
echo "info.json: ${INFO:-<missing>}"
echo "cross section: ${XSEC:-<none>}"
if [ -z "$XSEC" ]; then
    echo "::error::no valid cross section produced on $BACKEND"
    STATUS=1
fi

cat > summary.txt << EOF
node=$(hostname)
gpu=${GPU_NAME:-unknown}
backend=$BACKEND
modules=${MODULES:-<none>}
madspace=$(basename "$MADSPACE_PREFIX")
xsec=${XSEC:-n/a}
events=$NEVENTS
walltime=${WALLTIME}s
EOF
section "Summary"
cat summary.txt
exit $STATUS
