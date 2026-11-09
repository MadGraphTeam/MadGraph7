# TIRust / MadLoop benchmark

This campaign compares the exact starting revision
`9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f` with the integration candidate.
The prior NinjaRS report remains in `doc/benchmarkfolder`.

Run all commands from the repository root, using the recorded development
environment. Install CutTools, Ninja, COLLIER, NinjaRS and TIRust before exporting
the acceptance processes. The installer supplies the paths under `HEPTools`.

```sh
source tools/tirust_env.sh
python3 tools/tirust_benchmark.py init
python3 tools/tirust_benchmark.py model
python3 tools/tirust_benchmark.py build
python3 tools/tirust_benchmark.py campaign
python3 tools/tirust_benchmark.py summarize
```

`--cases qcd_n1_uux_z` restricts preparation or measurement to named cases.
`--source baseline` prepares only the frozen baseline. `--jobs` controls build
parallelism; `--cpu` selects one CPU from the current allowed affinity. The
default chooses its lowest permitted CPU. Measurement uses one CPU and one
thread. Preparation and measurement must not overlap on that CPU.

`cases.json` fixes the eight main processes, three controls, physical settings,
sample count and acceptance thresholds. The EW model pin records its archive
SHA-256 and complete converted UFO source inventory. The baseline manifest
records archive and individual source hashes. Disposable baseline sources,
model download archives, builds and logs are under `.benchmark-work/tirust`.

Preparation runs the real MG CLI. The generated Fortran executable reports its
actual external masses, and these determine the saved phase-space streams.
Each main process has 100 deterministic points. On-shell 2-to-1 samples use
longitudinal boosts at their physical invariant mass; other main samples have
1 TeV centre-of-mass energy. Separate stress streams cover applicable threshold,
soft/collinear and planar configurations. Complex internal masses require a
separate explicitly configured complex-mass process; ordinary real-mass runs
do not claim that qualification.

The worker computes full helicity/color sums with independent MadLoop stability
tests. Each process initializes and warms once, then receives serial commands.
Every timed call contributes to the returned backend/precision histogram; the
final traversal retains all Born and Laurent coefficients and split orders.
Raw files survive failed parses and commands. `cold_seconds` includes initial
evaluation and warmup; it excludes loading the parameter card.

There are at least seven adjacent timing pairs, alternating which executable
runs first. Calibration targets one second per batch; any retained batch below
95% of that target makes the series ineligible. The performance statistic is the
geometric mean of paired candidate/baseline ratios. Its conservative upper
two-sided 95% Student-t endpoint must not exceed 1.05. This is at least as strict
as the requested one-sided 95% upper bound. Absolute per-backend timing tables
use medians and median absolute deviations; their confidence columns stay null
unless separately estimated.

The default performance pair is the materialized baseline reduction chain and
the same chain prefixed with backend 9. Explicit-backend series cover 1, 6, 7, 8
and 9. Correctness compares every explicit candidate backend with candidate
CutTools on the same saved points, in addition to baseline/candidate agreement.
The acceptance comparison uses MadLoop's existing `LoopMEComparator` criterion:
`abs(a-b)/abs(a+b) < 3e-6`; poles may also pass when their magnitude divided by
the maximum finite coefficient is below that tolerance. Born coefficients do
not enlarge the finite scale. Zero/zero passes; opposite nonzero values have an
undefined symmetric ratio and fail unless the small-pole exception applies.
Raw ordinary relative residuals and the separate `1e-8` target are retained for
every coefficient. Native DP usage and rescued usage remain distinguishable.

`summary.json` contains only measured results or explicit pending/failed states.
A missing observation has null timing, never zero. A complete gate requires all
five numerical comparisons, all eight main process performance gates, and
separate regression gates for the four existing backends.
Each run directory is append-only: an existing label is never overwritten.
Preserve failed attempts under a different label before rerunning; final report
selection must explicitly identify the successful run and retain the failures.

## Build the report

After regenerating `summary.json` from retained observations:

```sh
source tools/tirust_env.sh
python3 docs/benchmarkfolder/build_report.py
python3 docs/benchmarkfolder/build_report.py --verify-only
```

The report build does not run numerical measurements. It writes `benchmark.pdf`
and `report-build.json`, which records input and PDF hashes, the exact Typst
version, creation timestamp, and font search paths. Typst 0.15.1, Libertinus
7.051 and DejaVu 2.37 reproduce the recorded presentation. The renderer refuses
a passing headline with missing main cases, absent backend comparisons,
insufficient samples/repetitions, or a runtime bound above the agreed limit.

`style-provenance.json` pins the requested pyAmpliCol report to commit
`7266c1600f6ec4e39be39986e6503a515c380c52`, including source/PDF hashes. Its
original `loop` branch URL is no longer available. Page geometry, typefaces,
heading/table colors and the logarithmic timing palette follow that source;
no external benchmark values are used as measurements for this integration.
An intermediate PDF explicitly says incomplete until the complete campaign
has passed the numerical and performance gates.
