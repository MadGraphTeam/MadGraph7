# Full-amplitude diagnostic evidence

These experiments do not replace the qualified n1 runtime gate. Production
source and installed libraries are unchanged.

| Files | Evidence |
|---|---|
| `run.py`, `analyse.py` | Isolated controller and summary reproduction |
| `split-workspace-1/runner-used.py` | Exact historical controller matching `results.json` |
| `split-workspace-1/candidate-build-metadata.json` | Selected isolated DSO's build/source identity; library hash is checked before use |
| `split-workspace-1/*-MadLoopParams.dat` | Actual cloned cards with frozen settings |
| `split-workspace-1/results.json`, `summary.json`, `controller.log` | Complete observations, compact summary, and controller output |
| `split-workspace-1/{0..6}/{baseline,original,split}/` | All seven triads, each with retained calibration, values, return codes, timing, runtime-library identity and worker log |
| `split-workspace-1/{0..2}/profile-split/` | Separate stage runs including Fortran clocks and inclusive bridge/upstream counters |
| `../bridge/split-validation.json`, `../bridge/split-validation-workers/` | Separate original/split ABI output, cache and failure-equivalence validation |
| `prepare_o3.py`, `tirust-adapter-o3-build/` | Isolated adapter-only compiler experiment, exact commands/flags, static archive pins, untouched executable reconstruction and before/after object identities |
| `split-adapter-o3-1/` | Seven triads comparing original/O3 Fortran adapters with the same split DSO, plus baseline; all calibration, numerical, timing and dependency evidence retained |
| `SHA256SUMS` | Every retained file except this checksum manifest and Python bytecode caches |

The main split comparison used the original uninstrumented worker. Only its
selected TIRust DSO differed. The paired coefficients were compared both with
the numerical criterion and as packed binary64 bytes. All runtime dependency
hashes were verified against the qualified attempt-3 inputs, with only that
explicit DSO replacement allowed. Host interference and the CPU change from
the earlier attribution remain explicit limitations.

The O3 experiment's variant names follow the generic controller: `original`
means the original adapter with the split DSO; `split` means the O3 adapter
with that **same** DSO. Its confidence interval includes unity, so no benefit
is established and the compiler-policy experiment remains unpromoted.
