# environment for the folding Z j j scan on nic6
module load GCC/14.3.0 SciPy-bundle/2025.07-gfbf-2025b CMake/3.31.8-GCCcore-14.3.0 >/dev/null 2>&1
export FZ=/scratch/ucl/cp3/omatt/fold-zjj
export MG7=/scratch/ucl/cp3/omatt/fold-zjj/MG7
export LD_LIBRARY_PATH=/scratch/ucl/cp3/omatt/fold-zjj/MG7/HEPTools/lhapdf6_py3/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/scratch/ucl/cp3/omatt/fold-zjj/MG7/HEPTools/lhapdf6_py3/lib/python3.13/site-packages:${PYTHONPATH:-}
