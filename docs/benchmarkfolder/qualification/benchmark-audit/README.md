# Independent benchmark evidence audit

This audit inspected the campaign controller, persistent Fortran/Python worker,
standalone pole diagnostics, and their unit/compiled-fixture tests. It performed
no physical amplitude timing while builds were active. Parser/driver tests use
synthetic numerical stubs and cannot establish physical accuracy or performance.

## Confirmed behavior

- The frozen `qcd_n1_uux_z` baseline generation record materializes
  `8|6|7|1`. The reserved `tirust-first` label requires exactly that chain for
  the baseline and `9|8|6|7|1` for the candidate. Explicit backend labels require
  the corresponding single reducer; TIRust compares against baseline CutTools.
- The driver allocates every loop and Born-only split-order column. It retains
  Born, finite, simple-pole and double-pole coefficients, with the total loop
  accuracy used only for Born-only columns lacking their own loop accuracy.
- Every timed amplitude call contributes to a status histogram, not just the
  final traversal. The parser requires the histogram total to equal
  `loops * points` and rejects unstable or unexpected intermediate statuses.
  It decodes both H and U, distinguishing TIRust DP/QP from legacy aliases.
- The finite numerical rows come from the final timed traversal. Seven
  alternating paired repetitions use the worker's wall clock. The paired
  confidence interval uses Student-t on log ratios, with a conservative
  two-sided 95% upper endpoint for acceptance. Absolute standalone medians are
  descriptive and do not receive the paired-ratio interval.
- Failed worker shutdown cannot publish a passing summary. Retained failure
  markers override passing summaries; missing runs remain pending. Reparse
  recomputes acceptance from raw rows, histograms and timings.
- Pole diagnostics retain full and partial results independently, require NaN
  finite output for partial evaluations, and compare evaluated poles using
  decimal arithmetic. Omitted high-rank fixtures remain explicitly unqualified.
  Their fixed full-then-partial timing is excluded from amplitude acceptance.

## Findings and remediation

1. **Effective compiler flags were not compared.** A synthetic paired run with
   candidate `FFLAGS=-O3 -ffast-math` and unchanged baseline flags was accepted
   before the correction. The new `tools/tirust_benchmark_compiler.py` resolves
   actual GNU make recipes, including environment overrides, target-specific
   flags, special polynomial rules, MODEL and DHELAS. It retains ordered
   arithmetic/optimization/code-generation options, compiler/wrapper hashes,
   and injected compiler environment. Include/output paths are normalized for
   comparison. Eight focused tests cover these cases. The harness now calls
   this helper at build/startup and when comparing/revalidating paired evidence.

2. **Static archive hashes were sampled only at launch.** A static library
   rebuilt after the worker executable could otherwise be recorded as the
   dependency of an executable containing older embedded code. The harness now
   captures archives and ABI/provenance metadata before and after linking,
   checks those records at worker startup, and checks runtime/executable bytes
   after measurements. The pole runner also retains executable, dependency,
   compiler, source and resolved parameter-card identities.

3. **Legacy objects need a real rebuild before assigning compiler evidence.**
   Relinking only the executable does not prove that existing process, MODEL,
   and DHELAS objects used the newly captured flags. The harness now requires
   an explicit numerical-build record and rebuilds these generated objects
   and archives when prior compiler evidence is missing. Source and flag values
   themselves must not be changed to make a comparison pass.

4. **The compiled worker must match its retained protocol template.** An old
   executable and old generated driver can be mutually consistent while a run
   archives a newer `tirust_benchmark_worker.f`. The harness now binds
   the template hash and expansion arguments to the compiled driver, and
   verifies the expanded retained template and its exact generated-source
   inventory entry during reparse. The inspected
   frozen baseline `qcd_n1_uux_z` driver already matched the current template;
   this finding prevents future reuse of a stale driver.

All eleven revised frozen-baseline compilation records were checked: each
records a complete numerical source rebuild, MODEL/DHELAS/process build logs
and compiler policies, identical pre/post link-input hashes, and a valid exact
worker-template/source binding. No acceptance result should be
promoted using a legacy object record lacking these bindings. The pole runner
likewise requires a qualified original source build; copied objects must keep
that build's effective compiler policy when linking a diagnostic; that
comparison is now enforced before the pole diagnostic link.

## Measurement scope that the report must retain

`cold_seconds` includes the first MadLoop call and warmup through a complete
steady traversal. It excludes process startup and `SETPARA`, so it is an
initialization-plus-warmup measurement, not isolated initialization overhead.

`FORCE_STABILITY_CHECK(.TRUE.)` remains active throughout timing for every
backend. This matching policy must be disclosed. Backend/precision histograms
describe the final accepted result of each amplitude call; they do not count
every internal reducer attempt or tensor reduction.

The main campaign gate concerns the eight main processes. It must not be used
alone to claim completion of control-process, stress, cache, pole-mode, or
installation qualification. Generated stress fixtures are not measured stress
results. NaN partial-finite output is not proof that finite kernels were
skipped; that requires the separate kernel instrumentation.

After a failed CLI pair/campaign, the retained `failure.json` is authoritative.
The top-level summary and PDF must be refreshed before presentation, so an older
pending report cannot conceal an observed failure.

## Calibration retries and the first qualified runtime failure

The first two `qcd_n1_uux_z` attempts had seven pairs and passing numerical
comparisons, but minimum timed batches of 0.82844 and 0.75077 seconds. They
therefore failed the duration requirement. The third attempt uses sustained
calibration with a conservative two-second target; its minimum batch is
1.87734 seconds. Independent reparse confirms an eligible candidate/baseline
ratio of `1.1210500947672746`, with paired 95% interval
`[1.1034925280600616, 1.138887018281026]`. This is an observed runtime failure
for that candidate, not a missing measurement.

The first implementation allowed selecting arbitrary later attempts, which
could permit choosing a favorable rerun after a qualified failure. Following
this audit, retries require an exclusive pre-registration before measurement,
an objective short-duration numerical pass or classified protocol failure,
and unchanged prior-evidence hashes. An already eligible result cannot be
replaced. Summaries also retain qualified failures from every chain or
existing-backend attempt even if a selected-directory record changes. A new
source/library optimization requires a separately identified campaign.
Historical retry reasons and all raw attempts remain retained. Fourteen
focused integrity tests passed after these corrections.

## Fixed-QP warmup protocol finding

The original persistent driver unconditionally enabled
`FORCE_STABILITY_CHECK(.TRUE.)`. In fixed QP (`CTModeRun=4`), an unchecked
answer has H=1 and accuracy=-1, which causes MadLoop to repeat the call and
return initialization-like T=4. Repeated warmup calls therefore cannot reach
the original driver's steady-state criterion. The interrupted QCD n3 and
other fixed-QP attempts using that driver are protocol-confounded; they cannot
establish numerical failure or slow QP execution.

The main worker now explicitly requires automatic `CTModeRun=-1` with
`CTModeInit=1`. It records every warmup call and every output column, rejects
invalid later statuses including T=4/U=0 immediately, and reports these as
numerical `MeasurementError` rather than retry-eligible protocol failures.
Constructor failures retain startup evidence; a valid continuing DP
initialization state remains permitted. Independent source inspection and
the worker/integrity suite passed 29 tests after this correction. The existing
qualified QCD n1 runtime failure remains authoritative and was not rerun.

Separate fixed-QP diagnostics use an explicitly derived isolated driver with
forced checks disabled. They retain unchanged numerical objects, archives,
compiler policy, physical inputs and genuine QP final evaluation; status 120
is labeled unchecked. Their coefficient comparisons do not qualify ordinary
stability behavior or runtime acceptance.

## Verification performed

The final combined suite below passed 57 tests after the audit fixes, including compiled
Fortran stub drivers. The compiler helper was also exercised read-only against
the real frozen baseline export: it resolved 20 process, 14 MODEL and 17 DHELAS
compilation recipes, with ordinary process flags unoptimized and the polynomial
rule retaining `-O3`.

```sh
source tools/tirust_env.sh
python3 -m unittest \
  tests.unit_tests.loop.test_tirust_benchmark \
  tests.unit_tests.loop.test_tirust_benchmark_integrity \
  tests.unit_tests.various.test_tirust_benchmark_worker \
  tests.unit_tests.various.test_tirust_benchmark_compiler \
  tests.unit_tests.loop.test_tirust_benchmark_poles \
  tests.unit_tests.loop.test_tirust_benchmark_points
```
