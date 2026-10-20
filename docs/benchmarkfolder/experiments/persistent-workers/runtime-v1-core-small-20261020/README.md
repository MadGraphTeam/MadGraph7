# Five-process core series with the corrected GNU runtime

The eight generated candidate processes are recorded in
`../../final-regeneration`. This series measures five cases with all seven
targets: original CutTools/Ninja/Collier and candidate
CutTools/Ninja/Collier/NinjaRS.

```sh
source .ninjars-build/dev-env.sh
python tools/ninjars_benchmark_workers.py measure \
  --source both --backends 1,6,7,8 \
  --cases qcd-1,ew-1,qcd-2,ew-2,ew-3 \
  --cpu 207 --repetitions 7 --timeout 21600 --clean-build \
  --run-id runtime-v1-core-small-20261020
```

All task-owned profiling, numerical tests, and independent compilation were
stopped before the series started. The host remains shared with unrelated
workloads. `cpu-selection.json` records an earlier ten-second observation
used to choose CPU 207; it is not a claim of exclusive CPU ownership.
The harness compiles the seven workers before evaluating each case, runs only
one worker at a time, warms the complete stream, and alternates backend order
for the paired repetitions.

The evaluated numerical library and generated-source identities are frozen.
Isolated cache and additional Fortran sparsity prototypes are not part of
these executables. The subsequent interface-source packaging change preserves
the archive and Fortran module byte for byte.

`run-status.json` records whether the series has finished. Completion of
measurement alone does not establish acceptance: promotion must validate the
retained protocol proof, all numerical rows, original-versus-candidate legacy
results, input/executable identities, and paired confidence intervals.
The two 2→4 cases and strict QCD three-body performance remain separate
outstanding work.
