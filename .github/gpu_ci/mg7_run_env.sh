# Sourced by the mg7 run jobs of gpu_runner_ci.yml (pp_ttx_mg7.sh, crossing_folding_mg7.sh),
# with HERE (this directory) and REPO (the checkout) set: loads the modules and the python
# environment, names the GPU, links madspace where the mg7 runtime looks for it, writes the
# default configuration and makes the PDF set available.
# Environment: BACKEND, MODULES, GPU_ARCH, VENV, MADSPACE_PREFIX, PDF_SET, CACHE_DIR
# (see pp_ttx_mg7.sh). Sets GPU_NAME, CACHE_DIR and LHAPDF_DATA_PATH.
CACHE_DIR=${CACHE_DIR:-${GLOBALSCRATCH:-$HOME}/mg7-gpu-ci/cache}
mkdir -p "$CACHE_DIR"

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

# madspace goes where the mg7 runtime looks for it: <MadGraph7>/madspace/install.
# In the CI the checkout is the runner's own, set up here from scratch. Anywhere else (the
# BACKEND=cpu try-out in a developer checkout) an in-tree madspace build, or a link to
# another install, and the developer's configuration files are left alone.
INSTALL=$REPO/madspace/install
if [ "$GITHUB_ACTIONS" = true ]; then
    rm -rf "$INSTALL"
    ln -s "$MADSPACE_PREFIX" "$INSTALL"
elif [ ! -e "$INSTALL" ] && [ ! -L "$INSTALL" ]; then
    ln -s "$MADSPACE_PREFIX" "$INSTALL"
elif [ "$(cd "$INSTALL" 2> /dev/null && pwd -P)" != "$(cd "$MADSPACE_PREFIX" 2> /dev/null && pwd -P)" ]; then
    echo "::error::$INSTALL exists and is not MADSPACE_PREFIX=$MADSPACE_PREFIX:" \
         "set MADSPACE_PREFIX=$INSTALL to use it, or remove it yourself"
    exit 2
fi
# default configuration, as .github/actions/checkout_mg5 does for the other CI jobs
if [ "$GITHUB_ACTIONS" = true ] || [ ! -e "$REPO/input/mg7_configuration.txt" ]; then
    cp "$REPO/input/.mg7_configuration_default.txt" "$REPO/input/mg7_configuration.txt"
    cat >> "$REPO/input/mg7_configuration.txt" << EOF
auto_update = 0
automatic_html_opening = False
notification_center = False
nb_core = ${SLURM_CPUS_PER_TASK:-4}
EOF
else
    echo "Keeping the existing input/mg7_configuration.txt (not a CI run)"
fi
if [ "$GITHUB_ACTIONS" = true ] || [ ! -e "$REPO/Template/LO/Source/make_opts" ]; then
    cp "$REPO/Template/LO/Source/.make_opts" "$REPO/Template/LO/Source/make_opts"
fi

export LHAPDF_DATA_PATH=$CACHE_DIR/lhapdf
if [ ! -d "$LHAPDF_DATA_PATH/$PDF_SET" ]; then
    echo "Downloading the PDF set $PDF_SET"
    mkdir -p "$LHAPDF_DATA_PATH"
    curl -fsSL --retry 3 "https://lhapdfsets.web.cern.ch/current/$PDF_SET.tar.gz" | tar -xz -C "$LHAPDF_DATA_PATH"
fi
