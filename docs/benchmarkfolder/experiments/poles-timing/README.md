# Paired full versus poles-only tensor diagnostic

Ten of twelve predeclared settings completed seven warmed timing pairs with
matching requested pole coefficients. Both DP pentagon and hexagon evaluations
were refused in full and partial mode; their QP counterparts succeeded. This
diagnostic uses the historical release and is not a full-amplitude performance
acceptance result. The rank-six DP tadpole is slower in partial mode, so these
measurements do not establish universal acceleration.

The ratio below is poles-only time divided by full time, using paired per-call
wall times. Values below one favor poles-only evaluation. The interval is a
two-sided 95% Student-t interval on seven paired log ratios (six degrees of
freedom). Full and partial columns are median wall microseconds per call.

| Tensor | Route | Precision | Full (µs) | Poles (µs) | Ratio [95% interval] |
|:--|:--|:--|--:|--:|:--|
| Massive tadpole, N=1/rank 6 | Public Fortran bridge | DP | 9.011 | 9.418 | 1.0407 [1.0325, 1.0490] |
| Massive tadpole, N=1/rank 6 | Public Fortran bridge | QP | 1827.579 | 1838.398 | 1.0050 [0.9992, 1.0107] |
| Massive bubble, N=2/rank 4 | Generated adapter | DP | 11.833 | 11.312 | 0.9512 [0.9387, 0.9639] |
| Massive bubble, N=2/rank 4 | Generated adapter | QP | 1737.839 | 1121.059 | 0.6464 [0.6440, 0.6489] |
| Infrared triangle, N=3/rank 3 | Generated adapter | DP | 8.952 | 8.783 | 0.9766 [0.9542, 0.9996] |
| Infrared triangle, N=3/rank 3 | Generated adapter | QP | 1038.307 | 858.082 | 0.8235 [0.8154, 0.8316] |
| Massive box, N=4/rank 4 | Generated adapter | DP | 32.225 | 18.004 | 0.5575 [0.5510, 0.5642] |
| Massive box, N=4/rank 4 | Generated adapter | QP | 15384.019 | 2219.937 | 0.1438 [0.1414, 0.1463] |
| Massive pentagon, N=5/rank 2 | Generated adapter | DP | — | — | Both modes refused |
| Massive pentagon, N=5/rank 2 | Generated adapter | QP | 39077.989 | 2176.705 | 0.0556 [0.0547, 0.0566] |
| Massive hexagon, N=6/rank 2 | Generated adapter | DP | — | — | Both modes refused |
| Massive hexagon, N=6/rank 2 | Generated adapter | QP | 84851.836 | 5144.454 | 0.0606 [0.0601, 0.0611] |

The massive pentagon and hexagon fixtures have vanishing poles. Their large QP
gains therefore do not describe infrared-singular five- or six-point inputs.
Refusals record the public adapter's `STABLE=false`; this experiment does not
identify the internal refusal reason or assign timings to failed results.

## Protocol and scope

The [plan](PLAN.md) fixes twelve settings before execution. Ten use the actual
generated `qcd_n4_uux_zzgg` diagnostic entry points, which have rank-five
capacity. The two rank-six tadpole settings call the public `MTIRUST` Fortran
bridge directly. No generated rank limit or production source was changed.

The executable resolves a private copy of the historical DSO with SHA256
`49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12`
and Fortran archive
`04c2b1ac3b8275dd2d0060416c4147bebd5e9bd62b71509e4cb29ce3d5c47d08`.
The specification records the full copied installation, original generated
source/build provenance, diagnostic compiler flags, resolved runtime libraries,
and executable identity. These are historical identities; a subsequent library
build is not covered by the measurements.

Each setting runs on CPU 9 under the shared numerical lock. Full and partial
modes receive independent preflights and calibration, followed by seven pairs
that alternate which mode runs first. Separate fixed call counts target 1.1
seconds per mode, with at most four calibration batches and ten million calls
per batch. Every timed paired batch passed the declared 0.825–1.925 second duration
interval. Wall and process CPU times are retained. Across the measured batches,
the largest wall/CPU ratio was 1.01432. An independent four-job dependency build
ran on the shared host during this window; affinity, before/after host-load
snapshots, CPU times, and uncertainty intervals document the measurement context.
They do not turn this diagnostic into an isolated-host acceptance campaign.

Numerical caches are disabled, with a cache clear before each batch outside the
timer. Immutable reduction plans, prepared variants, and allocated scratch stay
warm. The pinned native executor recomputes master integrals and tensor moments
when numeric stores are disabled; these are warm reductions without numeric
cache hits. Input-array restoration is inside the timer. Driver-side output
packing, printing, and coefficient comparisons are outside it; adapter mapping
and public-bridge serialization remain inside the measured calls.

Every returned batch is checked against its same-mode preflight coefficients,
and requested poles are compared between the full and partial preflights, using
`abs(a-b)/max(1,abs(a),abs(b))`, with bounds 1e-10 in DP and 1e-24 in QP.
The largest observed pole residual is 3.4694469519536141888e-18 (DP triangle);
the other settings are exact at the retained precision. Partial generated
adapter finite slots must be `NaN+0i`. The direct bridge instead returns its
documented zero finite placeholder, which is checked separately. The timing
driver does not expose finite-operation counters; proof that finite work is
omitted remains in the native poles-only qualification evidence.

## Retained evidence and reproduction

The machine-readable table is
[analysis-summary.json](historical-release-paired/analysis-summary.json).
Each fixture folder retains its exact input, raw coefficients and timings,
host-load snapshots, and parsed result. The run-level specification, compilation
record, frozen execution sources, and manifests bind these files together.
Binary artifacts are identified by hash rather than included in this report.

After execution, the parser gained stricter checks for bounded calibration,
fixed calibrated call counts, ordering, and the completion footer. Its frozen
copy is in `analysis-protocol/`; the original execution protocol remains in
`protocol-sources/`. Replaying the stricter parser reproduces all twelve
original result JSON documents exactly. No native run was repeated or replaced.

Run from the repository root to repeat this analysis without compiling or
executing native libraries:

```sh
source tools/tirust_env.sh
python3 docs/benchmarkfolder/experiments/poles-timing/historical-release-paired/reanalyze.py
```

The original native invocation was:

```sh
source tools/tirust_env.sh
python3 tools/tirust_benchmark_pole_timing.py \
  .benchmark-work/tirust/outputs/candidate/qcd_n4_uux_zzgg \
  --label historical-release-paired --cpu 9
```

A new native run needs a new label, matching generated-source provenance, the
exact historical installation, and a coordinated CPU window. The tool rejects
a different installed DSO/archive instead of silently testing a newer build.
Its current parser has six focused tests; the Fortran driver has three compiled
stub tests with bounds checking, covering DP/QP routes, array strides, refusal
handling, and partial finite placeholders. Both suites passed during development.
