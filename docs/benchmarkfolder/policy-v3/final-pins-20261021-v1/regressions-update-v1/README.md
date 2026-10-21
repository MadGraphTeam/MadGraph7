# Installer update regression refresh

The new `install ninjars --update` option passed **47 fresh focused tests**,
including replacement, rollback after failed build/smoke checks, exact current
lockfile publication, CLI dispatch and the retained `--force` alias. The update
uses this MadGraph checkout's committed pins and lockfile after `git pull`.
An additional read-only call to the real MG7 command validator confirms that
`ninjars --update --jobs=3` reaches the installer with both options intact; this
control is not counted as another unittest case.

The earlier `regressions` capsule remains immutable. Its exporter and FKS
observations are reused: **20 passed and one existing Python 3 GoSam bypass**.
Those 21 tests were not rerun. Exact source hashes and an AST comparison show
that only the installer alias/diagnostic text, installation help, documentation
and focused tests changed. Fortran templates, generation logic and fixtures
remain unchanged. The reuse check accepts only the recorded five-file change
set and the exact parser/help transformations.

This capsule therefore covers 68 software cases through 47 fresh observations
and 21 retained observations. No real reduction engine was called. Physics,
stability, convergence and performance retain their independent final-pin gates.

A later correction changed only the ignored Rust `cfg(test)` master-diagnostic
collector. The 47 software tests do not compile or execute that module; the
recording harness had also inventoried its bytes. Its original hash, the actual
47-test result and log remain unchanged. `cfg-test-equivalence.json` explicitly
binds the one old-to-current test-module change and the unchanged production
declaration. Separate current-source Rust fenv tests pass; the planar diagnostic
still fails and remains an outstanding numerical finding. These Rust observations
are not added to the 47 software tests or treated as a numerical gate pass.

From the repository root, verify payloads, current source hashes, the actual
test footers and the retained source comparison without rebuilding anything:

```sh
python -B docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/regressions-update-v1/verify.py
```

The verifier reads the earlier capsule and the tested Git revision's unchanged
installation-help source. Its one small pre-update installer source is retained
with its exact SHA256 to make the AST comparison reproducible.

To repeat the affected suite, enter the repository's pinned Nix environment
(`nix develop`) and run from the repository root:

```sh
mkdir -p .ninjars-build/tmp
export TMPDIR="$PWD/.ninjars-build/tmp"
export MAKEFLAGS="SHELL=$(command -v bash)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
taskset -c 12 python -B -W ignore::ResourceWarning -W ignore::DeprecationWarning -m unittest -v \
  tests.unit_tests.loop.test_ninjars_fortran \
  tests.unit_tests.various.test_ninjars_install \
  tests.unit_tests.various.test_ninjars_statistics \
  tests.unit_tests.various.test_ninjars_integration_boundaries \
  tests.unit_tests.various.test_ninjars_upstream
```

Choose an available CPU if CPU 12 is unavailable. The exact original command,
Python/GFortran versions, CPU affinity and development-environment hash are
retained in `focused-result.json`; `recorded-runner.py` contains the actual
exclusive-output recording harness. No exporter/FKS rerun is required for this
installer-only change.
