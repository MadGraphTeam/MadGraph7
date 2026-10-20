# Integrate NinjaRS into MadLoop on `madloop_ninjars`

## Goal and working rules

Deliver an installable NinjaRS reduction backend for MadGraph7, selected by
`MLReductionLib=8`, with a Fortran interface, oneloopmaster scalar integrals, and
its own higher-precision rescue. Validate agreement with CutTools, Ninja, and
Collier across eight QCD/EW processes, achieve per-process runtime within 5% of
original MadLoop with Ninja, and produce the requested reproducible Typst report.

- Work exclusively from `/shared/external/madloop_ninjars/madgraph7_madloop_ninjars`
  on branch `madloop_ninjars`. The goal has been assigned through `create_goal`.
- Commit as `huashengshao <erdissshaw@gmail.com>`, using that GitHub account over
  HTTPS. Commit and push tested milestones to `origin/madloop_ninjars`, without
  force-pushing.
- Preserve original revision `27147b3300f556a75661a886e92bdbe00558ee25` as an
  archived baseline source copy inside this repository.
- Treat the sibling NinjaRS and oneloopmaster clones as read-only references.
  Keep implementation, build products, and benchmark work inside MadGraph7.
- Linux x86_64 is the first supported and validated platform.

## Implementation

### Fortran interface and Rust bridge

- Add process-prefixed DP and QP NinjaRS wrappers following the existing Ninja
  interface, for both Born interference and loop-induced amplitude reduction.
- A repository-owned Rust bridge exposes versioned C-compatible context
  creation, evaluation, and destruction through Fortran `ISO_C_BINDING`, with
  explicit buffer lengths, error statuses, and panic containment.
- Use the fast tensor engine and `OneLoopNativeMasters` in double precision.
  Retain contexts/scratch by topology, rank, and precision; reset numerical
  inputs and accumulated results for every independent evaluation.
- Preserve tensor ordering, loop-momentum signs, propagator reversal, complex
  squared masses, scale, Laurent order, normalization, UV/R2 and split orders.
- Reuse MadLoop's DP/MP kinematic-matrix construction. Preserve dimensionless
  `OSThres` separately from absolute scalar-integral on-shell thresholds.
- Map unsupported inputs, failed masters, nonfinite results and unstable
  geometry into existing rescue. Never expose stale successful results.

### Genuine higher-precision rescue

- Recompute numerator and kinematics through MadLoop's QP path.
- Transport binary128 through fixed-width 40-significant-digit ASCII records.
  Recover the original 113-bit values before exact widening to 256 bits; return
  results through the same transport.
- Adapt the existing generic tensor numerator into the bridge, preserving its
  expansion arithmetic and upstream attribution.
- Use generic reduction and oneloopmaster-backed `SymbolicaMasters<Float>` at
  256 bits, with original Ninja's QP scalar-integral threshold.
- MadLoop owns retry order, rotations, acceptance, and final conversion.
  Add `UseQPIntegrandForNinjaRS`, default true.

### Configuration, installation, exports and reporting

- Register ID 8 in cards, capabilities, dispatch, checks, configuration and
  library discovery. Preserve default `6|7|1`; permit `8` and ordered mixtures.
- Provide repository-owned `install ninjars`, pinned downloads and explicit
  local NinjaRS/oneloopmaster source overrides.
- Pin NinjaRS `0d0d7e7a8033431b60a8580731084742e0655cda`, oneloopmaster
  `09b2c7497743db875ed446da95123175dd00b1c8`, required Symbolica patches and lockfile.
- Install the Rust library, Fortran module/interface and native link metadata.
  Honor prefix/compiler settings; run a compiled Fortran reduction smoke test.
- Support standalone and FKS/NLO exports, including copied dependencies.
  Generated processes must not require the reference clones.
- Keep all old return integers. NinjaRS stable DP uses `(H,U)=(2,8)`; rescued
  NinjaRS uses `(3,2)`. Update contextual decoding and dedicated usage counters.
- Without NinjaRS, old workflows still work; selecting unavailable ID 8 alone
  fails with an actionable diagnostic.

## Fortran optimization and independent audit

Separate agents review the numerical interface, Fortran performance and benchmark
correctness. Keep integration and optimization commits separable.

First measure bounded local coefficient scratch instead of per-call allocation,
generated coefficient-count lookup, fused copying/sign inversion, and contiguous
independent clearing/copying loops.

Then profile momentum construction, color contractions and reusable geometry.
Evaluate prefix sums, precomputed color factors/sparsity and explicitly
invalidated geometry caches; retain only numerically validated measured gains.
Do not weaken stability, omit contributions, disable rescue or alter benchmark
inputs. Compare optimized old backends with their original implementations.

## Validation and performance acceptance

Generate using actual `bin/madgraph` command files:

| n | `loop_sm`: QCD | `loop_qcd_qed_sm_Gmu`: EW |
|---|---|---|
| 1 | `u u~ > z [virt=QCD]` | `u d~ > w+ [virt=QED]` |
| 2 | `g g > t t~ [virt=QCD]` | `e+ e- > mu+ mu- [virt=QED]` |
| 3 | `u u~ > z g g [virt=QCD]` | `e+ e- > t t~ a [virt=QED]` |
| 4 | `u u~ > z z g g [virt=QCD]` | `e+ e- > mu+ mu- a a [virt=QED]` |

Use real masses and zero widths, fixed parameter cards/seeds, 1 TeV for n=2..4,
and on-shell energies with boosts for n=1. Retain model provenance/checksum.

### Correctness

- Compare Born, finite, single/double poles and split orders for all four
  backends at 100 fixed points/process.
- Pairwise tolerance: `|a-b| <= 1e-7 max(|a|,|b|) + 1e-10 S`, with S the common
  point's largest Born/Laurent magnitude. Investigate every failure.
- Add loop-induced gg->H/HH, mixed QCD/QED, complex masses, reversal, rotations,
  changed parameters, near-singular points and forced rescue controls.
- Verify binary128 transport below DP resolution and difficult points against
  higher-precision convergence and available analytic pole controls.
- Test installer retry/failure, missing libraries, standalone/NLO builds,
  legacy return codes and old/new statistics readers.

### Performance

- Retain a Fortran driver with identical point streams, helicity/color sums,
  pole settings, flags and initialization. `check timing` is only a smoke test
  because its Collier configuration omits poles.
- Time complete warmed MadLoop evaluations, including numerator, reduction,
  masters, stability and actual rescue.
- Pin one CPU/evaluation thread; alternate backend order; take at least seven
  repetitions. Record medians, dispersion and confidence intervals; increase
  repeats if the 5% boundary remains unresolved.
- Every process must have NinjaRS median <=1.05 times original MadLoop+Ninja.
  An uncertainty-straddled gate is unresolved, not passed.
- Report all backends and real backend/rescue counts, plus separate generation,
  compilation, initialization and component measurements.

## Milestones and benchmark report

Commit and push each validated milestone:

1. Goal/plan, reproducible toolchain, pinned dependencies and original baseline.
2. Installer and DP interface with initial generated-process agreement.
3. Own precision rescue, complete exports and status reporting.
4. Measured Fortran optimizations with regression evidence.
5. Full eight-process validation, independent audit and final report.

Create `docs/benchmarkfolder/benchmark.typ` and `benchmark.pdf`, raw data, command
cards, provenance and reproduction instructions. Reuse with attribution the
style from
https://github.com/MadGraphTeam/pyamplicol/blob/loop/docs/performance_reports/loop/benchmarks.typ:
identical typography/margins, navy headings, striped tables, pale panels and
logarithmic green-white-red cards. Compile and visually inspect the PDF.

Mark the goal complete only when correctness and performance gates pass, the
report is reproducible and all milestones are pushed to `madloop_ninjars`.

## Execution ledger

- Implementation started 2026-10-20. Baseline checkout was clean and on the
  requested branch. GitHub account `huashengshao` has repository push access.
- Bootstrap commit `7f883420e` was pushed to `origin/madloop_ninjars`.
- The original revision is archived in `.benchmark-work/original`. Original
  Ninja 1.2.1, OneLOop 3.6 and Collier 1.2.9 have built successfully with the
  pinned toolchain. Download checksums and a reproducible setup script are
  retained; the EW model was imported through the original CLI and its Python
  source checksum recorded. Host compatibility changes apply only to downloaded
  installer shebangs; original MadLoop numerical sources remain untouched.
- Reference setup milestone `165fd1b16` was pushed. A byte-for-byte check against
  `git archive` confirmed that all tracked baseline source files remain unchanged.
  The setup script handles IREGI's inherited-PWD build issue without source edits.
- `install ninjars --jobs=2` completed through the real CLI, including its compiled
  DP and binary128 Fortran smoke test. The first generated core process (`qcd-1`)
  passes all four backend comparisons at 100 points and the original-versus-
  candidate legacy-backend regression, including the final ABI2 installation.
  Nine Rust tests, the compiled Fortran
  interface/card/status tests, installer tests, statistics readers, benchmark
  acceptance tests and exporter reference tests pass.
- Independent review corrected MP box sampling scale and conditional cleanup.
  Separate C contexts now share one MP worker with isolated numerical state.
  ABI2 validates explicit lengths on every DP and ASCII buffer before access.
  An actual complex-mass rank-five pentagon converges between 256 and 384 bits
  at 31 relative digits, retaining a numerator perturbation below DP resolution.
- DP and MP each pass 21 retained tensor lanes against original Ninja's complete
  Laurent coefficients. The generic MP engine's separate rational diagnostic uses
  its documented basis-dependent split; MadLoop consumes the complete finite term.
- Integration commit `d47f0bc4f` was pushed. Final ABI2 copied-dependency standalone
  export survives relocation and removal of external library search paths.
  Full NLO export, compilation, all soft/collinear checks and 20/20 pole-cancellation
  checks pass. The actual local-source override also compiles and passes its
  installed DP/QP smoke test; primary installation and reference sources are
  unchanged. Commands and hashes are retained in `installer-validation.json`.
- Independent installer review corrected loop-induced OPP advice to recognize
  option 8 and documented upstream Symbolica worker licensing for concurrent
  processes. EW model reuse/import now verifies the original recorded file
  inventory and checksum, without changing that checksum's scope.
- Four focused processes (loop-induced gg->H/HH, mixed QCD/QED orders and complex
  masses) pass four-backend agreement and original-versus-candidate legacy
  regression. Default full-QP NinjaRS rescue uses status 322: H=3 and U=2, with
  the existing tens digit indicating QP numerator construction.
- Original-Ninja scratch-allocation optimizations were tested in copied generated
  outputs: all 100 DP/QP outputs match exactly for qcd-1/qcd-2, but exploratory
  timings show no resolved full-evaluation benefit. The production patch is not
  retained. Profiling instead identifies generated UPDATE_WL contractions as the
  dominant Fortran cost. An accumulation-order-preserving loop interchange passes
  2,880 DP and 2,880 QP exact comparisons and real-process regression. All eight
  candidate processes have been regenerated; retention still requires resolved
  timing evidence.
- Independent benchmark review added exact split-order dimension checks, actual
  parameter-card resolution checks, retained cards/build settings, numerical
  consistency across repetitions, adjacent original-Ninja/NinjaRS timing pairs,
  and rejection of mixed executables/settings. Initialization timing and
  incremental compilation are explicitly distinguished in the report.
- Installer/export validation milestone `ca62a8680` was pushed. Additional
  controls now pass rotations, changes of mass and renormalization scale,
  retained-versus-fresh context checks, and near-singular full-QP rescue against
  CutTools and original Ninja. A separate deliberately sub-ulp stability
  diagnostic rejects original Ninja; its returned coefficients agree, and the
  rejection remains visible in the retained evidence.
- Default stability settings do not guarantee the requested cross-backend
  accuracy at high multiplicity. The common core configuration now uses
  `MLStabThres=1e-9` and `COLLIERRequiredAccuracy=1e-10` for every backend,
  including the untouched baseline. Physical inputs and comparison tolerances
  are unchanged. All three original backends pass the strict 100-point QCD
  three-body calibration, including 52 CutTools and two Ninja QP rescues.
  Default-setting discrepancies and strict-setting checks are retained separately.
- Persistent Fortran benchmark workers are being validated against fresh
  processes so initialization is performed once per backend and all timed
  repetitions still evaluate the complete point stream. Promotion requires
  representative parity including genuine QP rescues, exact protocol identity,
  complete paired series, and all numerical/status checks. These changes avoid
  redundant initialization without excluding any timed physical evaluation.
- Focused-control and benchmark-validation milestone `f0a3da575` was pushed.
  Representative persistent-worker validation now passes all 18 combinations,
  including actual QP rescues, with bitwise-identical printed results. Native
  runtime linking is receiving an additional directed-rounding regression;
  representative parity will be refreshed after that correction.
- Compact and expanded wavefunction traversal changes preserve the arithmetic
  order of each output. All 41 polynomial tests and focused export comparisons
  pass, including 2,880 compact and 11,250 expanded exact comparisons per
  precision. Subsequent paired measurements resolve representative benefits:
  compact qcd-2 ratio 0.91643, expanded ew-2 ratio 0.933690 and qcd-2 ratio
  0.970802, with all 95% interval upper bounds below one. Final regeneration,
  all-process legacy regression and the NinjaRS runtime gates remain required.
- The complete eight-process correctness gate, measured optimization gate and
  all final performance gates remain outstanding. The styled report currently
  labels missing measurements as pending.
