# Separate forced-DP stability producer

`tools/ninjars_dp_stability.py` and `tools/ninjars_dp_stability_driver.f`
provide source preparation and a strict raw reader. They do not start a MadLoop
campaign, change the existing benchmark worker, or qualify a performance gate.
Actual process compilation and evaluations require the orchestrator's resource
handoff. Initial observations use the same 100 fixed core points for all four
tools. Loop-induced and difficult 2→4 samples have their own declared inputs.

## Build preparation in a private generated copy

1. Copy an already qualified generated output into a new private directory.
   Preserve original source, executable, library, card, point, selector, and
   cohort identities. Do not rebuild or modify the source output in place.
2. Save its original `CT_interface.f`. Run the `instrument` subcommand into a
   **new** private file, with a receipt. Replace only the copied interface with
   that file. The operation finds generic `LOOP_n` dispatches and adds an
   observer immediately after every CTLOOP, Ninja, NinjaRS, or TIR call,
   including their QP branches. Unknown calls/signatures/indices fail closed.
   Removing the marked observer calls must recover every original source byte.
3. Render the dedicated driver with the actual process prefix, external-leg
   count, and loop split-order count. It replaces only the private driver's
   source; the original driver remains available for provenance.
4. Set a singleton backend, `CTModeInit=CTModeRun=1`, full UV/IR poles, and
   `COLLIERGlobalCache=0`. Keep the physical cards and other numerical settings
   fixed. The global cache is deliberately disabled for this standalone
   forward/reverse protocol, separately from normal-controller timing.
5. Compile the copied interface and driver with the retained compiler flags.
   Record exact source and executable hashes and compilation diagnostics.
   Run serially on the assigned CPU, with OMP, OpenBLAS, Rayon, and MKL thread
   environment variables all set to 1. The driver arguments are:

   ```text
   executable POINTS_FILE TOOL_ID HELICITY_INDEX NEW_RAW_OUTPUT
   ```

   `HELICITY_INDEX=-1` uses the complete helicity sum. A positive index must
   bind the same physical helicity tuple for all four tools. The main
   production timing and the single-helicity timing table are different
   measurement streams and use their own executables.

The dedicated driver warms the process in forced DP, then evaluates each
point in modes 1, 2, and mode 1 after the exact z90 component permutation.
It calls `SLOOPMATRIXHEL` directly, with the normal threshold wrapper disabled.
It explicitly clears the NinjaRS native-retry flag before every observation;
the per-dispatch recorder also records the flag actually seen. Any compensated
DP or QP observation is rejected by this native-DP stream's reader.

## Failure capture and floating-point state

The generated scalar `S(...)` is consumed at each dispatch before another
helicity or topology can overwrite it. The observer never writes the stable
flag, coefficient array, library selector, or precision mode. Its first fifteen
fields record calls; the final three are observation-level flags read by the
caller after evaluation:

```text
1  calls                       10 backend 6 calls
2  stable=false                11 backend 7 calls
3  any nonfinite Laurent part  12 backend 8 calls
4  actual CTMODE >= 4          13 algorithmic FPE_DP flag seen
5  backend 1 calls             14 algorithmic FPE_QP flag seen
6  backend 2 calls             15 NinjaRS compensated-DP flag seen
7  backend 3 calls             16 hardware IEEE_INVALID (0/1)
8  backend 4 calls             17 hardware IEEE_DIVIDE_BY_ZERO (0/1)
9  backend 5 calls             18 hardware IEEE_OVERFLOW (0/1)
```

The three complex Laurent outputs include both real and imaginary parts in
the nonfinite check. Rational terms have already been combined by each
interface before returning. Algorithmic FPE flags are reset for each
observation, recorded immediately at dispatch, and retained again at its end.
The three hardware exception flags are cleared at the start of each measured
observation and read in the driver after the complete evaluation. They are
read in the same caller scope: the compiler's IEEE procedure environment can
save and clear incoming flags on entry to a helper. These are sticky Boolean
observations, not independent exception counts. Hardware flags are retained
separately and do not alone establish numerical failure.
Underflow/inexact flags are not rejection criteria. Rounding-to-nearest is
checked initially and after every observation.

Forced explicit modes legitimately return `H=1` and internal accuracy `-1`.
Those values are not a successful stability certificate and are not a failure
by themselves. In particular, three zero outputs with any rejected reduction
are failures; three complete finite zero outputs with no rejected reduction
remain explicitly classified as `all_zero`.

## Retained authority and report interface

The raw format retains a header, warm-up marker, `BEGIN`, `OBS`, all loop
split-order `ROW` records with unnormalized Born/finite/poles/internal accuracy,
`ENDOBS`, and a final `DONE`. A crash leaves an incomplete observation. The
reader preserves missing slots in the fixed denominator and marks the
campaign incomplete. Duplicate, malformed, or out-of-range records raise an
error instead of silently selecting a subset.

The manifest schema is `ninjars-forced-dp-raw-v1`, with:

- `stream="forced_dp_self_consistency"`, `rescue_enabled=false`,
  `instrumentation="per-dispatch-observer-v1"`;
- the full numerical revision, generation revision, generator identity,
  runtime identity, and NinjaRS archive hash in `cohort`;
- a checksum-bound `cohort_record` for an actual consumer, revalidated with the
  existing `ninjars_benchmark_policy_v3.validate_cohort` rather than a global
  latest-dependency label;
- checksum-bound provenance records in `bindings`, an increasing `thresholds`
  array, and a `runs` array;
- each run's `case`, `process`, `group`, `sample_kind`, `tool`, `N`, `nexternal`,
  `nso`, `helicity`, checksum-bound `raw`, and `artifacts`;
- artifacts named `original_interface`, `instrumented_interface`, `driver`,
  `points`, `parameter_card`, `madloop_card`, `executable`, `build`, `execution`,
  `stdout`, and `stderr`, each with a `path` and SHA-256;
- the build record's `driver_sha256`, `instrumented_interface_sha256`, and
  `executable_sha256`; the execution record's `requested_tool`, `helicity`,
  actual `cpu`, `exit_code`, and `thread_environment`.

The manifest is diagnostic evidence, not an arbitrary `passed=true`
qualification. `report_snapshot(manifest_or_summary, expected=None,
inputs=None)` rereads raw artifacts, recomputes exact source instrumentation,
verifies driver and build bindings, and reconstructs all estimators. Supplying
a derived summary still reopens its original manifest and raw data.

These checks do not by themselves establish that a supplied cohort label is
the semantic source/runtime identity of the executable. Before physical
figures are published, the orchestrator and independent reviewer must match
each private original interface, generated source set, dependency archive,
cards, points, selector, build, and executable to the already qualified
cohort target. That actual preparation and runtime binding remains pending;
this diagnostic reader must not be treated as the final publication authority.

The summary schema is `ninjars-dp-stability-summary-v1`. Its `cases[id]`
contains process/group/sample/helicity/N metadata and `tools["1"|"6"|"7"|"8"]`.
Each observed tool has `status`, `scores` for `combined`, `direction`, and
`lorentz` contrasts, raw-derived point values/classifications, exact threshold
counts, Wilson intervals, zero-event upper bounds, failure categories,
hardware-flag observations, and raw/source bindings. Missing tools have no
invented zero curve and make the overall summary incomplete.
`observations_complete` describes raw coverage. In this preparation version,
`complete` and `physical_qualified` remain false until the actual semantic
target binding and instrumented-process parity controls have been supplied
and independently validated. A complete self-consistency data file alone
cannot promote physical results.

The compiled standalone `recorder-control` fixture contains the **exact same
observer routine** and tests false-stable zero results, valid zero results,
nonfinite parts, QP mode, algorithmic FPE, compensated-DP selection, hardware
flags, and argument immutability for tool IDs 1, 6, 7, and 8. This verifies
the recorder without loading a numerical reduction engine. It does not
replace the later actual instrumented-process parity and failure controls.
## Actual target qualification

The optional per-run `qualification` binds a controller start record, original
and instrumented result records, and a parity receipt. The reader validates the
explicit release through the existing v3 cohort reader. It then checks the
actual CLI origin through `validate_origin_records`, recomputes the private
generated source manifest and complete runtime through `actual_runtime`, and
requires every source change to be the exact rendered driver or declared
observer insertion. Model/DHELAS archives, scalar/reducer libraries, Fortran
modules, native link metadata, physical cards, fixed point bytes, executable,
selector, command exits, and thread settings are retained separately.

Publication observations use the agreed `MLStabThres=1e-9` and
`COLLIERRequiredAccuracy=1e-12`. Other numerical settings remain those of the
qualified origin, apart from explicit DP modes, singleton backend selection,
full poles, disabled COLLIER internal acceptance, and declared cache zero.
The first cheap legacy recorder control used the export default
`MLStabThres=0.001`; it remains narrower control evidence and cannot qualify a
publication stream. No card is silently relabelled.

The reader recomputes recorder parity from all raw Born/Laurent/split-order
coefficients and internal accuracy records. Finite numbers compare by float64
bits, including signed zero. Matching nonfinite spellings are counted separately
and never described as finite bit accuracy. Rejected reductions and FPEs remain
numerical failures in the stability estimator even when mechanical recorder
parity succeeds. Thus recorder qualification does not select successful points.

`physical_qualified` denotes an actual source/runtime-bound observation stream;
it is not a statement that all its numerical results are accurate or accepted.
Individual legacy target controls may qualify while the summary remains
incomplete. Summary completion requires every declared process to contain all
four qualified, complete backend streams. It does not replace the independent
core correctness, performance, or whole-task completion gates. Missing NinjaRS8
parity is reported as pending. All qualification is read-only: the reader never
initializes MadLoop, creates a numerical context, or executes a benchmark.

Current qualification requires a complete matching plain/instrumented raw
scope. The raw reader retains truncated streams and assigns missing slots to
the failure-inclusive diagnostic denominator, but abnormal-exit or truncated
streams cannot yet receive mechanical parity qualification. If an actual
campaign process fails this way, its attempted backend/process cell must remain
visible as failed and qualification pending; it must not disappear from the
case list or become a successful zero curve. Matched-prefix failure qualification
must then be added and audited against that concrete failure before a qualified
CDF is published. An unattempted backend is pending, not an observed failure.

