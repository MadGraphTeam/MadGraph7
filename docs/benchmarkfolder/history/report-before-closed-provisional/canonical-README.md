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
default samples per-core load and chooses the quietest allowed CPU. Measurement
uses one CPU and one thread. Finish this campaign's builds before acceptance
timing. Other jobs on the shared host remain uncontrolled; retained load samples
describe those conditions.

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
tests. The shared benchmark card freezes `MLStabThres=1e-6` for both revisions
and every reducer; production retains its default `1e-3`. The stricter common
policy follows the retained synthetic stress audit and was selected before
acceptance timing. It does not replace independent numerical comparisons.
Each process initializes and warms once, then receives serial commands.
Every timed call contributes its final accepted reducer and precision to the
returned histogram; this does not count every internal rescue attempt. The
final traversal retains all Born and Laurent coefficients and split orders.
Raw files survive failed parses and commands. `cold_seconds` includes initial
evaluation and warmup; it excludes loading the parameter card.

There are at least seven adjacent timing pairs, alternating which executable
runs first. The minimum target is one second per batch; calibration aims for
two seconds to allow for shared-host speed variation. Any retained batch below
95% of the minimum makes the series ineligible. Actual durations remain in the
raw observations. The performance statistic is the
geometric mean of paired candidate/baseline ratios. Its conservative upper
two-sided 95% Student-t endpoint must not exceed 1.05. This is at least as strict
as the requested one-sided 95% upper bound. Absolute per-backend timing tables
use medians and median absolute deviations; their confidence columns stay null
unless separately estimated.

The default performance pair is the materialized baseline reduction chain
`8|6|7|1` and the TIRust-first chain `9|8|6|7|1`. Explicit-backend series cover 1, 6, 7, 8
and 9. Correctness compares every explicit candidate backend with candidate
CutTools on the same saved points, in addition to baseline/candidate agreement.
The acceptance comparison uses MadLoop's existing `LoopMEComparator` criterion:
`abs(a-b)/abs(a+b) < 3e-6`; poles may also pass when their magnitude divided by
the maximum finite coefficient is below that tolerance. Born coefficients do
not enlarge the finite scale. Zero/zero passes; opposite nonzero values have an
undefined symmetric ratio and fail unless the small-pole exception applies.
Raw ordinary relative residuals and the separate `1e-8` target are retained for
every coefficient. Final accepted DP/QP outcomes and failed evaluations remain
distinguishable; intermediate failed attempts are not inferred from the final
return code.

`summary.json` contains only measured results or explicit pending/failed states.
A missing observation has null timing, never zero. A complete gate requires all
five numerical comparisons, all eight main process performance gates, and
separate regression gates for the four existing backends.
Each run directory is append-only: an existing label is never overwritten.
An objectively short batch series with passing numerical checks may be retried
with a reason recorded before the retry. All attempts remain visible. A qualified
runtime failure stays authoritative for that candidate; selecting a later,
faster attempt cannot erase it. A source optimization requires a new candidate
identity and campaign.

`qualification.json` indexes the separate installer, Fortran adapter, bridge,
pole-only, relocation and optimization checks. The detailed evidence records
test scope and any debug-build limitations. `experiments/numerical/` retains
early numerical checks performed during compilation; their times are explicitly
ineligible for acceptance. The original default-policy pilot is preserved.
The independently audited `experiments/numerical/completed-pilots-index.json`
summarizes eight completed 100-point pilots and binds their retained evidence.
Run `python3 docs/benchmarkfolder/experiments/numerical/audit_completed_pilots.py`
to reproduce that audit. These physics results have a separate report page and
do not supply the pending runtime-campaign measurements.
The standalone pole diagnostic is described in `POLES.md`.

The current PDF is a closed-evidence intermediate report for
`split-dp-terminal-1`: all eleven processes, all six lanes per process,
and 100 saved points per lane pass the `3e-6` criterion. The complete matrix
contains 66 lanes. The `1e-8` misses in QCD three- and four-particle cases
and the EW four-particle case remain visible; accepted precision
is reported separately from internal retries, which this dataset does not observe.

Final precise timings are **deferred to Hua-Sheng on his local machine**, as
recorded in [the user amendment](PLAN-AMENDMENT-2026-11-10.md). Shared-host timing
is provisional and uses no CPU reservation. No new paired timings are included
in this snapshot and no `≤1.05` performance claim is made. The runnable [local timing handoff](campaigns/split-dp-terminal-1/timing-handoff-2/LOCAL_TIMING_TODO.md)
covers fresh installation, local registration, numerical replay, timing and a
separate local PDF. Its source/test qualification is not a local measurement.
The historical quiet-window controller remains unrun.

The experimental adaptive-precision bridge is an isolated development build,
not a private repository or the installed backend used in this campaign. It
tries 256-bit MPFR first and retries an eligible conditioning refusal once at
512 bits. Its 20 Rust tests pass, but dynamic Fortran/physical qualification
remains open. The report also distinguishes the completed legacy Born/effective-
point checks and current synthetic stability replay from physical qualification.
The installer version-type hardening fix remains unpromoted; earlier passed
ABI/interface rows describe only their identified test groups.

The feature branch is `madloop_tirust`, based on `madloop_ninjars` at the frozen
baseline. The intended future PR target is `madloop_ninjars`.

`collect_intermediate.py` has an explicit allowlist of completed evidence; it
hashes retained files and extracts claim anchors without executing numerical
code. `intermediate_report.py` checks those hashes, actual installer identity,
raw-source assertions and diagnostic ratios, and rejects missing failures or
promotion to final acceptance. Active experiments outside the allowlist are
excluded. The previous PDF, ledger, context and every declared document input
are preserved under `history/report-before-status-refresh/` and still support
`build_report.py --verify-only` independently.

## Build the report

The canonical report uses the explicit immutable closed-evidence selection.
Its selected dossier binds the checkpoint inventories and excludes active
numerical directories before opening them. To reproduce or verify this snapshot:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/build_report.py --campaign split-dp-terminal-1 \
  --campaign-report campaigns/split-dp-terminal-1/report-snapshots/closed11-1.json
python3 -B docs/benchmarkfolder/build_report.py --verify-only --campaign split-dp-terminal-1 \
  --campaign-report campaigns/split-dp-terminal-1/report-snapshots/closed11-1.json
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_tests/various/test_tirust_report.py \
  docs/benchmarkfolder/test_intermediate_report.py \
  docs/benchmarkfolder/test_campaign_report*.py
```

For a future refresh, close the selected runs, create a new selection and dossier
path, and preserve this report's ledger/PDF before rebuilding. Do not overwrite
a previous dossier or use partial numerical rows as completed evidence. The
closed10 PDF, ledger and Typst sources are preserved under
`history/report-before-closed11/`; its immutable selections and dossier remain
in the campaign report-snapshots directory.

The report build does not run numerical measurements. It writes `benchmark.pdf`
and `report-build.json`, which records input and PDF hashes, the exact Typst
version, creation timestamp, and font search paths. Typst 0.15.1, Libertinus
7.051 and DejaVu 2.37 reproduce the recorded presentation. The renderer refuses
a passing headline with missing main cases, absent backend comparisons,
insufficient samples/repetitions, or a runtime bound above the agreed limit.
The build ledger now also binds all frozen dependencies of the intermediate
supplement. The historical intermediate supplement is retained separately. New campaign
refreshes use `campaign_report.py --selection`; collecting the dossier never
reruns or overwrites measurements.

`style-provenance.json` pins the requested pyAmpliCol report to commit
`7266c1600f6ec4e39be39986e6503a515c380c52`, including source/PDF hashes. Its
original `loop` branch URL is no longer available. Page geometry, typefaces,
heading/table colors and the logarithmic timing palette follow that source;
no external benchmark values are used as measurements for this integration.
An intermediate PDF explicitly says incomplete until the complete campaign
has passed the numerical and performance gates.

## Registered campaign report dossier

The twelve-page intermediate report is preserved exactly under
`history/report-before-split-dp-campaign/`. Its `reconstruct.py` verifies all
4,693 original report inputs against the recorded Git revision and can restore
them into a new ignored directory. The original installer outer inventory is
included: its later additive update is not substituted into that historical
ledger. The preceding eight-page archive and `doc/benchmarkfolder` NinjaRS
report remain unchanged. See the archive README for verification and exact PDF
reproduction commands.

The docs-only `campaign_report.py` collector prepares a separate immutable
dossier for a named campaign. It does not run an amplitude, compiler, live
campaign guard or performance measurement. It checks retained source/native
identities and reparses saved raw observations using the hash-bound parser
sources. Candidate export Git revisions may differ after documentary commits;
the registered exporter anchor and each observed revision are displayed
separately, with one exact exporter file inventory required throughout.

After the included run directories are closed and their outputs no longer
change, retain a **new** dossier and render the selected report explicitly:

```sh
source tools/tirust_env.sh
python3 tools/tirust_benchmark.py summarize --campaign split-dp-terminal-1
PYTHONDONTWRITEBYTECODE=1 python3 docs/benchmarkfolder/campaign_report.py \
  --campaign split-dp-terminal-1 \
  --output campaigns/split-dp-terminal-1/report-snapshots/closed-1.json
python3 docs/benchmarkfolder/build_report.py --campaign split-dp-terminal-1 \
  --campaign-report campaigns/split-dp-terminal-1/report-snapshots/closed-1.json
python3 docs/benchmarkfolder/build_report.py --verify-only \
  --campaign split-dp-terminal-1 \
  --campaign-report campaigns/split-dp-terminal-1/report-snapshots/closed-1.json
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_tests/various/test_tirust_report.py \
  docs/benchmarkfolder/test_intermediate_report.py \
  docs/benchmarkfolder/test_campaign_report.py \
  docs/benchmarkfolder/test_campaign_report_numerical.py
```

Do not freeze a dossier while its included run directories are active:
`results.json` and worker logs may still grow. The collector ignores later new
paths when validating a frozen snapshot, but changes to already-bound files
correctly invalidate that snapshot. An unlaunched row can remain pending in a
closed snapshot; it never becomes a successful or zero-time observation.
The current canonical PDF is not changed by collecting a dossier.

Three datasets have distinct roles. The final complete warmup traversal checks
all eleven physical cases, baseline CutTools plus five candidate backends, on
the exact 100-point streams; these rows have no timing, FPE-flag observation or
internal retry counters. Paired full-amplitude measurements determine the eight
main runtime gates and existing-backend regressions, including every timed
call's final accepted status. Traced cohort diagnostics separately time frozen
native-DP, compensated-DP and naturally rescued QP subsets, retaining flags,
retry counters, migration and failure evidence. Their tracing overhead and
different point subsets prohibit treating cohort medians as formal acceptance
measurements or direct backend speed ratios.

The dossier displays the exact 0.95-second eligibility floor, two-second
calibration target and observed batch minima. Qualified main-process failures
remain authoritative across all retained attempts. Control-process timings are
descriptive: generic raw-summary slowdown labels are preserved but do not
create main runtime failures. Numerical control checks remain required.
Qualification uses the registered shared `MLStabThres=1e-6` policy. The
historical synthetic case accepted inaccurately at the default `1e-3` threshold
remains an open disclosure; proving general accuracy at that looser threshold
is not an additional delivery gate. The required EW threshold, QCD
soft/collinear and complex-mass stress cases remain separate unresolved gates.
Complete delivery additionally requires the installed ABI and Fortran checks,
separate subprocess/event cache lifecycle qualification, precision cohorts and
resolution of retained stress failures. The cache lifecycle criterion stays
pending until a separate frozen native proof is bound; public-call parity alone
does not establish it. A passing eight-process runtime gate alone cannot
complete the integration goal.
