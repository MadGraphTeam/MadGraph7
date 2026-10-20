# Bounded MG7 regression campaign

This campaign completed 90 unittest cases: 89 passed and one existing Python 3
GoSam bypass was honored. It covers polynomial generation, compiled Fortran
updates and IEEE guards, HELAS calls, loop exporter IO (including loop-induced
output and eight split-order variants), FKS IO, and all five supported comparisons
between serial and multicore NLO generation. It does not execute the Rust
reduction engine or scalar-integral providers and does not establish an amplitude
correctness or performance gate.

The source snapshot is checkpoint `ecdf397f0823a571b4b2cfa213a8ca133d270776`,
with the reviewed fixture updates and the one-line FKS test helper described
below. `validation.json` binds 102 relevant production source paths and the
individual test, fixture and log files. Its report adapter checks those production
hashes against the final timing candidate instead of assuming that a later commit
has the same tested code. Compiler and Python identities are recorded in that
file. The reported suite durations are test-run wall times, not MadLoop timings.

| Suite | Cases | Passed | Skipped | Result |
|---|---:|---:|---:|---|
| Polynomial, compiled Fortran and HELAS | 69 | 69 | 0 | pass |
| Loop exporter IO | 2 | 2 | 0 | pass |
| Supported FKS regressions | 19 | 18 | 1 | pass |

The skipped case is `TestFKSOutput.test_w_nlo_gen_gosam`. It already carries
`@test_manager.bypass_for_py3`; direct unittest execution ignores that registry and
hits the deliberate CLI rejection of GoSam with low-memory multicore generation
on Python 3. The retained supported-suite harness applies that existing bypass as
an explicit unittest skip and limits each interface to `nb_core=2`. No other
case is excluded. Missing Ghostscript produces diagram-conversion warnings during
FKS export; these do not bypass any assertions.

## Review of fixture failures

The interrupted initial broad run has no completed result and remains labeled
incomplete. Fresh unmodified runs reproduced two loop IO failures, and the first
plain FKS run produced two stale IO failures plus the known GoSam restriction.
All these failed/incomplete logs are retained. Diagnostic regeneration wrote only
to private copies in `.benchmark-work`; its successful update-mode result is not
a test pass.

Exactly 22 expected-output files were updated after review:

- Eight split-order `loop_matrix` files receive the same existing NinjaRS retry
  wrapper, state initialization and compensated-DP retry. Their old lines are
  preserved; all eight inserted diff payloads are identical.
- The optimized `gg > t t~` coefficient-construction fixture redirects seven calls
  to the already committed private QP factor helpers and adds one entry
  `IEEE_INEXACT` capture. Removing just those substitutions restores its complete
  former Fortran statement stream. Its polynomial fixture adds four private
  helpers and preserves every legacy line. Existing compiled arithmetic/IEEE tests
  passed before these expectations were updated.
- Twelve FKS fixtures cover four subprocesses: `NLOOPLIB=8` and corresponding
  unavailable backend-array entries in `loop_matrix`/`TIR_interface`, and the
  backend-8 `UseQPIntegrandForNinjaRS` predicate in `mp_compute_loop_coefs`.
  Each of the three four-file diff families is identical. The old backend values,
  QP count of two, arithmetic and thresholds are unchanged. Four CT interface
  files differed only by a blank line and were deliberately left untouched.

The FKS IO helper now disables NinjaRS alongside its existing Ninja and Collier
disables. The initial private capture revealed local-installation-dependent
availability flags; that capture and its summary are retained, but none of those
installed-backend expectations was adopted. The accepted FKS fixtures set
NinjaRS DP and QP availability to false and emit no retry or finalization code.
Installed/copied NinjaRS exports are validated by separate campaigns.

`fixture-review.json` records every old/new hash, classification and diff shape.
`verify_fixture_review.py` rechecks this narrow scope against Git's checkpoint.
No production file was changed by this regression work.

## Verification and reproduction

Run from the repository root with the retained toolchain environment (or compatible
Python 3 and a GNU Fortran compiler with binary128 support):

```sh
source .ninjars-build/dev-env.sh
python -B docs/benchmarkfolder/policy-v2/final-regressions-v1/verify_validation.py
python -B docs/benchmarkfolder/policy-v2/final-regressions-v1/verify_seal.py
```

`validation.json` contains each exact command. For the FKS replay harness, create
its disposable output directory first:

```sh
mkdir -p .benchmark-work/final-regression-review-v1
export TMPDIR="$PWD/.ninjars-build/tmp"
export MAKEFLAGS="SHELL=$(command -v bash)"
taskset -c 2,3 python -B -m unittest -v \
  tests.unit_tests.various.test_q_polynomial \
  tests.unit_tests.various.test_q_polynomial_compiled \
  tests.unit_tests.various.test_q_polynomial_factorized \
  tests.unit_tests.iolibs.test_helas_call_writers \
  tests.unit_tests.loop.test_loop_exporters
taskset -c 2,3 python -B \
  docs/benchmarkfolder/policy-v2/final-regressions-v1/diagnostics/run_supported_fks.py
```

The original three runs were separately bounded and pinned, as recorded in the
machine-readable commands. The combined command above reruns the same two
supported modules/groups without altering their assertions. Diagnostic collector
scripts are retained as provenance; they refuse to overwrite their private
comparison copy. Use a fresh scratch namespace if repeating a diagnostic capture.
The bulky private output copies and generated executables are not part of this
package and need not be pushed.

The report adapter is
`verify_validation.report_snapshot(path, candidate_generator_files, inputs=None)`.
It verifies raw logs and files, requires every tested production digest to match
the supplied candidate generation map, reports the explicit skip, and records all
checked local inputs when a dictionary is supplied. Optimized Python is rejected
so assertions cannot silently disable verification. `verifier-controls.json`
retains successful checks that a wrong/missing candidate source and both `-O`
entry points are rejected.
