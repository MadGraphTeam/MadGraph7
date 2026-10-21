# HELD: do not execute without root's later exclusive CPU207 timing handoff.
source .ninjars-build/dev-env.sh
env -u FFLAGS -u CFLAGS -u CXXFLAGS -u LDFLAGS -u GLOBAL_FLAG -u FC -u F77 -u F90 -u CC -u CXX \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 207 python -B tools/ninjars_benchmark_pair_v3.py \
    --spec docs/benchmarkfolder/policy-v3/timing-pair-plan-v1/specification.json \
    --handoff docs/benchmarkfolder/policy-v3/timing-pair-plan-v1/ROOT-HANDOFF-NOT-CREATED.json \
    --execute-numerical
