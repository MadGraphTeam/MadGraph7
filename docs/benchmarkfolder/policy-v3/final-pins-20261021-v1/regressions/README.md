# Final-pin MG7 software regressions

The final software suites ran 66 tests: 65 passed and one existing Python 3
GoSam/low-memory-generation bypass was explicitly skipped. No actual reduction
engine was evaluated. Physics agreement, stability, convergence and performance
require their separate final-pin evidence.

The focused suite covers card ID 8 and defaults, mocked installation and
rollback, copied-source provenance, pinned dependencies, old/new statistics,
and compiled Fortran recording backends for conventions, precision transport,
failure zeros, retry state and cleanup. The exporter suite checks standalone,
loop-induced and split-order output. The FKS suite checks generated output and
all five supported serial-versus-two-worker generation comparisons.

Selecting only unavailable NinjaRS in an already generated process now explains
how to install it, regenerate and rebuild. The compiled regression also checks
ordered fallback, the legacy default, and available ID 8. Fourteen current IO
fixtures received exactly the same diagnostic insertion; their other bytes and
legacy fixtures were preserved. See `fixture-review.json`.

`summary.json`, the individual results and logs retain the exact observed
commands, toolchain, counts and limitations. The first exporter observation
failed before the expected diagnostic-only fixture updates; it remains here as
history. Its successful rerun observed an unrelated concurrent policy-reader
test edit. Every relevant production, exporter, dependency and fixture hash
remained unchanged; both the broad observation and the explicit source scope
are retained. Missing optional Ghostscript/card-image tools and normal process
warnings remain in the FKS log. Earlier polynomial/HELAS runs are separate
historical evidence and are not counted again here.

Verify payloads, current source hashes and the actual unittest log footers from
the repository root with standard-library Python:

```sh
python -B docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/regressions/verify.py
```

To repeat the software suites, enter the repository's pinned Nix development
environment (`nix develop`), then run the following from the repository root.
Choose a previously unused attempt number if a result already exists. The
recorded runner is byte-identical to the original helper and writes exclusive
results under `.benchmark-work/final-dependency-regression-20261021-v1`.

```sh
mkdir -p .ninjars-build/tmp .benchmark-work/final-dependency-regression-20261021-v1
export TMPDIR="$PWD/.ninjars-build/tmp"
export MAKEFLAGS="SHELL=$(command -v bash)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
runner=docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/regressions/recorded-runner.py
taskset -c 12 python -B -W ignore::ResourceWarning -W ignore::DeprecationWarning "$runner" focused --attempt 20261021
taskset -c 12 python -B -W ignore::ResourceWarning -W ignore::DeprecationWarning "$runner" exporters --attempt 20261021
taskset -c 12,13 python -B -W ignore::ResourceWarning -W ignore::DeprecationWarning "$runner" fks --attempt 20261021
```

The FKS harness bounds `nb_core` to two and respects the existing Python 3
GoSam bypass. It changes neither production source nor numerical tolerances.
CPU IDs can be changed to available quiet CPUs; these are software checks,
not performance measurements.
