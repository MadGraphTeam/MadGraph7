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
- Fortran traversal milestone `f934704f8` was pushed. All eight core candidate
  processes have now been regenerated through actual CLI cards and rebuilt
  from clean subprocess objects, preserving every physical input byte.
- A directed-rounding probe found that Rust's static archive could replace
  GNU Fortran's binary128 arithmetic helpers. Explicit pre-archive `-lgcc_s`
  metadata restores the original compiler runtime. The installed smoke checks
  directed rounding, DP and genuine MP; independent relocated standalone
  checks pass DP110, QP120 and stable218. The fresh full NLO fixed-order build,
  soft/collinear checks and 20/20 pole checks pass. Existing installation
  metadata without the link-order contract requires reinstallation.
  This correction was committed and pushed as `49b4af33d`. Interface-source
  packaging additionally retains `mninjars.f90` in installations and internal
  exports; the copied source recompiles to the same module hash.
- Regenerated representative fresh/persistent comparisons now pass all 18
  combinations, with 102 hash-bound evidence files. The refreshed ggH,
  mixed-order and near-singular results also pass independent cross-backend
  comparisons. Other focused controls remain explicitly queued for the final
  chosen implementation.
- A seven-repetition exploratory strict `Zgg` series agrees numerically across
  original/optimized Ninja and NinjaRS at all 100 points; retained original
  CutTools/Collier strict calibrations also agree. NinjaRS rescues three points
  versus Ninja's two and takes 2.269 times original Ninja
  (95% interval [2.259, 2.725]). This fails the performance target and is
  retained as diagnostic evidence, not promoted acceptance data. Exact
  per-evaluation scalar-result reuse and further Fortran QP sparsity work are
  being investigated in isolated prototypes without changing thresholds,
  precision or physical inputs.
- The first four core cases now pass all four-backend numerical comparisons,
  original-versus-optimized legacy regression, and seven-repetition runtime
  gates. NinjaRS/original-Ninja median ratios are 0.846545 (`qcd-1`),
  0.924495 (`ew-1`), 0.890040 (`qcd-2`), and 1.037710 (`ew-2`); their respective
  95% interval upper bounds are 0.852432, 0.931509, 0.900480 and 1.043532.
  Measurements use CPU 207 with other task-owned computation stopped.
  The shared host is not claimed to be exclusive.
- That series stopped during `ew-3` calibration when original Collier returned
  status 410 at points 68 and 74. Its entire failed run, including raw rejected
  outputs, is preserved. A separately audited extraction retains every completed
  case, every backend and all 196 repetitions without modifying the failed
  source run or its protocol. Frozen validators and eight extraction tests pass;
  promotion independently passes agreement, legacy regression and performance
  for all four completed cases. The Typst report now shows 4/8 correctness and
  4/8 performance gates, with the remaining cases outstanding.
- EW three-body diagnostics also find finite-part discrepancies at points 49
  and 78 between double- and higher-precision evaluations. Tighter Collier
  integral accuracy removes the two status-410 flags but not these discrepancies.
  Numerator and integral precision are being isolated using the original
  backends and unchanged physical points. These diagnostic variants are not
  acceptance evidence and do not relax comparison or stability thresholds.
- Those EW discrepancies have now been traced to generated MP model code before
  reduction. Nineteen default-kind `CMPLX` constructors round to single precision;
  six continuation constants contain a mistyped pi; the MP expression parser
  also emits default-kind `REAL` casts for real inputs. The production generators
  now specify kind 16 and the correct binary128 pi literal. Compiled mathematical
  controls cover axis inputs, logarithmic sheets, a threshold-crossing square-root
  trajectory, a massless B0 continuation, and inputs differing by `2^-100`.
  An independent audit reproduced the defects and reviewed the corrections.
- A fresh actual CLI EW3 export with both generator corrections builds, retains
  the exact original parameter card, and changes only `model_functions.f` and
  `mp_intparam_definition.inc` among generated model sources. Its 100-point
  four-backend diagnostic now passes all six coefficient comparisons; the
  largest tolerance fraction is 0.000913. NinjaRS uses 98 DP evaluations and
  two genuine QP rescues. Scoped original-plus-helper-only references also
  agree across CutTools, Ninja and Collier. Full final-policy recertification
  of this export and the other seven cases remains required.
  Existing QCD model counterterms also call these helpers, so previously accepted
  measurements are historical evidence for their recorded source, not final
  validation of the corrected generator.
- Benchmark accounting after this numerical repair must keep the original source,
  measurements and numerical discrepancies intact. The performance denominator
  remains the literal original MadLoop+Ninja timing. Correctness must additionally
  compare all corrected candidate backends with separately identified original
  exports carrying only the two generated model corrections, with exact source
  hashes and all 100 points/orders. No incorrect original QP value is silently
  relabelled or excluded. A versioned correctness policy and refreshed worker
  proof are required before promoting new final measurements.
- The next isolated Fortran optimization is validated: skip exact-zero products
  in QP polynomial updates only for finite IEEE binary128 operands under nearest
  rounding. Nonfinite operands, other rounding modes and unsupported layouts
  execute the unchanged dense routine. DP updater bodies remain unchanged.
  A quiet seven-pair QCD3 comparison against the previous candidate generator
  gives ratio 0.831836 (95% interval [0.806229, 0.877321]); EW2's all-DP control
  gives 1.008293 ([0.974855, 1.017739]). Every coefficient, accuracy and return
  code matches exactly. These use retained pre-model-correction process inputs
  and qualify the isolated optimization, not the final NinjaRS performance gate.
  The production port passes 43 polynomial tests, 30,000 additional compiled
  bit/IEEE/fallback comparisons, four polynomial exporter checks and the
  model-helper/internal-parameter exporter check, with independent code review.
- Completed scalar-cache, generic-tensor dependency-plan and color-contraction
  experiments are retained with their failed attempts and exact numerical
  comparisons. None established a sufficient runtime benefit for production.
  A warmed Zgg rescue profile assigns only 7.7% of samples to scalar masters,
  versus 44.1% to QP Fortran numerator construction and 33.5% to generic tensor
  and reduction work. Further performance effort follows those measured costs.
  The extra Zgg rescue remains an aggregate direction-check failure; same-input
  256-bit reductions show that original Ninja's smaller direction difference
  is not evidence of uniformly smaller per-reduction error. Stability thresholds
  and independent direction evaluation remain unchanged.
- Fresh policy-v2 CLI exports and clean builds now cover all eight core cases
  and all five generated controls at `fd72f8914`. Original physical cards,
  point streams and external masses match exactly. QCD model exports change
  only the verified helper file; EW exports change that helper and the MP
  internal-parameter include. Complete source inventories and unchanged
  dependency/protected-input checks are retained in the separate v2 namespace.
  Generation performs no loop-amplitude evaluation and is not numerical
  acceptance evidence.
- The versioned correction policy now anchors original exports to committed
  pre-correction evidence, verifies exact model postimages and forced rebuilds,
  and requires refreshed coverage of all three source roles. Publication checks
  all 21 corrected-cohort pairs while retaining original numerical discrepancies
  and the literal original-Ninja timing denominator. The separate report reader
  revalidates the dataset and independently recomputes numerical and runtime
  conclusions. Its seven-page Typst layout is visually checked using labelled
  fixture data only; the previous measured report remains intact. Representative
  collection, final eight-process measurements and performance acceptance remain
  outstanding.
- A fresh pinned-download CLI installation passed its compiled Fortran DP/QP
  checks and reproduced the production Rust archive. Relocated standalone
  exports with internal dependencies passed DP, QP and stability evaluations
  while the installation path and external library environment were unavailable.
  The corresponding clean NLO build exposed a classifier incompatibility with
  `-fno-automatic`; the failed build and raw evidence are retained. An `ASSOCIATE`
  expression now avoids forbidden saved local storage in the elemental function.
  The fix passes 42,000 compiled binary128/input/IEEE comparisons including NLO
  flags, 39 polynomial tests and two exporter checks. Clean NLO regeneration
  and execution remain required; no NLO success is inferred from the CLI's zero
  exit code.
- Clean regeneration at `85795ccbb` now passes the full relocated internal
  standalone and NLO checks. The private pinned-download installation is hidden
  during execution and library-path overrides are removed. Standalone DP/QP/
  stability results agree with the retained CutTools reference; the NLO build
  passes all 20 pole checks and its soft/collinear tests with zero failures.
  Versioned GNU binary128 imports resolve in both NLO executables. The successful
  attempt is retained separately from the initial failure. This closes the
  classifier/export regression for the recorded source; final eight-process
  acceptance and any later numerical-archive recertification remain outstanding.
- An isolated unconditional compensated T3 expansion passes full 100-point,
  all-order coefficient comparisons in QCD1, EW2 and QCD3. Zgg uses two QP
  rescues instead of three. Exploratory timing shows added cost on the all-DP
  cases, with substantial host contention; these runs are not performance
  acceptance evidence. The production archive is unchanged. A separate
  explicit DP retry is being tested after normal stability failure, retaining
  the same thresholds, full direction/rotation checks and genuine QP fallback.
- Legacy fresh/persistent validation now covers 26 eligible targets in 12 exact
  runtime groups across the untouched and helper-corrected original roles.
  Each group has the required precision coverage; both roles include
  loop-induced and mixed-order controls. Strict ordinary forced-Ninja failures
  remain retained failures. The full final-candidate proof is still pending.
  All eight corrected-original source copies have passed clean MODEL and
  subprocess builds; their complete numerical collection is prepared separately.
- The isolated explicit compensated DP retry now passes its full QCD3 stream,
  rotation controls, loop-induced and mixed-order controls. It retries each
  finite native stability failure once, keeps the same thresholds and checks,
  and sends unresolved points to genuine QP. The 100-point QCD3 stream uses
  98 DP results and two QP rescues; all coefficients pass comparison. Seven
  quiet alternating pairs show ordinary-path retry/native ratios of 1.005424
  for QCD1 and 0.987828 for EW2, with exact coefficients and return statuses.
  These establish the isolated retry's behavior and overhead, not final
  acceptance against original Ninja. The sealed experiment is retained before
  its independently reviewed production port and ABI update.
- Following the user's arXiv:1604.01363 reference, deepen the symmetric-tensor
  accumulation study. Compare the paper's recursive construction with the
  existing MadLoop coefficient graph and NinjaRS transposed Horner expansion.
  Test generated Fortran predecessor-based accumulation, selected straight-line
  rank kernels and provable vertex sparsity in an isolated namespace. Preserve
  current production ports while this research proceeds; require both kernel
  and complete-process measurements before retaining a further change.
- The measured terminal-QP optimization is now ported to the production
  generator. Whole-graph and rendered-consumer checks select only private
  wavefunctions consumed through their diagonal trace; unknown uses retain full
  updaters. Fresh CLI exports reproduce all QCD3/EW2 100-point rows exactly,
  plus complete ten-point mixed-order and loop-induced QP diagnostics. The port
  passes 108,000 compiled numerical/IEEE comparisons, 12 graph tests, 39
  polynomial tests and eight standalone/FKS/MatchBox/TIR exporter checks.
  Public DP/QP updater source remains identical in 96 checked combinations.
  These validate the port; final ABI3 source/runtime recertification remains.
- The compensated DP retry is now ported as ABI 3 with separate lazy state,
  while the ordinary DP implementation remains byte-identical. The installer
  rejects older ABI metadata and checks native/compensated/native DP followed
  by genuine binary128-input rescue. All 16 Rust tests, 13 focused Fortran/
  installer tests and 24 freshly generated retry-flow scenarios pass. A
  fixed-form formatting failure was caught, corrected and retained before
  commit. Main installation, fresh exports and full numerical/performance
  certification must now use this committed ABI 3 implementation.
- ABI 3 is installed and all thirteen fresh standalone core/control exports
  clean-build at `c5496ca9b`, with original physical inputs and dependency hashes
  preserved. A separate pinned installation and relocated standalone DP/QP/
  stability run pass. Its NLO compilation catches an empty-process-prefix bug
  in terminal-batch IEEE declaration detection; the CLI's zero exit does not
  mask that retained failure. The one-character predicate correction passes
  thirteen focused tests, including real compilation with NLO flags, and two
  FKS exporter checks. New archived standalone/NLO validation and fresh final
  generation use separate namespaces after this correction.
- All thirteen corrected exports clean-build at `b44d20e7`; an independent
  inventory verifies their sources, executable hashes and one shared complete
  reduction runtime group. The refreshed archived internal standalone and
  NLO checks pass, including all twenty poles and soft/collinear tests. The
  final representative and generated-control collection plans are ready.
  Rotation, changing-mass/scale and fresh-context controls are prepared with
  exact retained point streams and driver-only changes to isolated copies.
  A rejected read-only-module guard is retained separately from the corrected
  preparation. No new core numerical or performance gate is inferred from
  these builds.
- Diagnostic CPU profiling now binds to exact measured workers and retains
  raw outputs/recordings. The report reader requires profile membership in its
  dataset, rechecks source/runtime identities and recomputes rows and sample
  categories. Eighteen report tests pass, including byte-identical PDF builds
  with different ambient creation epochs; the eight-page layout fixture is
  compiled and visually checked. Final measured profiles and the completed
  benchmark PDF remain outstanding.
- The corrected archived installation/export milestone is sealed across 101
  files. Pinned downloads and explicit dual local-source installation pass the
  ABI 3 native/compensated/native DP and genuine MP smoke tests. Relocated
  standalone DP, QP and stability runs agree; the complete NLO build passes
  twenty pole checks and all soft/collinear tests. Sixteen focused tests pass.
  With NinjaRS absent, bundled CutTools still evaluates successfully and an
  exclusive selection of backend 8 reports an actionable error. Both reference
  source trees remain unchanged. Optional StdHEP shower support is outside
  this successful fixed-order validation; no new performance claim follows
  from these checks.
- Refreshed representative coverage now passes for all final candidate runtime
  groups: eight targets and sixteen persistent records have exact fresh-process
  parity over complete ten-point streams. All four backends have DP coverage;
  CutTools, Ninja and NinjaRS each have actual QP coverage. Loop-induced and
  mixed-order controls are included. The new proof combines these with the
  independently bound legacy groups, without mixing runtime signatures.
  Its 264-file retention package passes independent read-only verification.
  Full generated controls, the eight core streams and final performance gates
  remain outstanding.
- Complete ggH, ggHH, mixed-order and complex-mass generated controls now pass
  all twenty-one corrected-cohort pairings at each of two repetitions over
  ten points. Independent recomputation and a 406-file retention manifest
  verify the new runs, exact runtime bindings and reused representative data.
  The separate rotation control agrees across all four backends. The changing
  mass/scale control exposed a driver setup error: only the DP top mass was
  updated before `COUP()`, leaving its independent QP value unchanged. The
  CutTools-rescue disagreement and all raw rows are retained as a failure.
  The separate driver that synchronizes both precisions passes across all four
  backends, including the three CutTools QP points. Reused and fresh NinjaRS
  contexts agree exactly. All ten near-singular points pass with actual
  CutTools 329, Ninja 328 and NinjaRS 322 rescues, and every backend passes the
  rotated-versus-unrotated comparison. Independent audit reparses all twelve
  raw streams and recomputes every comparison. The generated NinjaRS rebinding
  test stays DP; separate unchanged bridge tests supply MP mass/scale coverage.
- The final installed ABI 3 physical direction-reversal control passes all
  thirty native-DP, compensated-DP and MP rows for a triangle and box. It
  includes distinct complex squared masses, odd numerator coefficients,
  retained/fresh exact replay and deliberately unflipped negative controls.
  Its initial synthetic input remains a separately retained unstable result.
  The focused source/evidence audit covers the final installation and controls;
  the eight core streams and performance gates remain pending.
- Symmetric predecessor/gather kernels are exact but show no useful speedup.
  Exact product sharing improves isolated kernels, but the seven-pair QCD3
  full-process ratio is 0.998449 with a 95% confidence interval of
  [0.966772, 1.026978], so it is not promoted. A new diagnostic trace confirms
  16.655% fewer complex multiplications in the selected helpers after zero
  skips; profiling and pure-real/imaginary-factor specialization are being
  investigated separately. No experimental sharing kernel is in production.
- The subsequent static-owner experiment passes 72,000 arithmetic/flag checks,
  forty trap controls and 4,000 changing-state calls. Its small extra kernel
  gain does not justify a production port. A complete QCD3 diagnostic profile
  attributes 18.33% of sampled cycles to the targeted QP rank categories, an
  upper bound for the selected vertices. The recorder's header-finalization
  failure, unchanged event bytes and separately recovered profile are retained.
- A distinct changed-rounding factorization now has exact source proofs for
  sixteen fermion/scalar maps (1,024 coefficient identities). It contracts a
  vertex row once before distributing it to four momentum coefficients, while
  preserving the original constant term. Static selected-product counts fall
  by 38.59% for QCD3 and 41.12% for QCD4 before dynamic zeros and runtime guards.
  Retained source snapshots allow independent offline proof verification.
  Compiled oracle, stability and performance validation are still required;
  the factorization is not part of production.
- The isolated compiled derivative-row factorization now passes 240 exact
  stored-input fixtures covering 273,280 complex binary128 outputs, including
  severe cancellation. Separate campaigns check 48,000 false-entry fallbacks,
  152 true-entry guard rejections and forty enabled-trap controls. An independent
  replay verifies the retained inputs, source hashes and arithmetic results.
  A complete 100-point QCD3 comparison preserves all 800 printed coefficients,
  accuracy estimates and statuses: 98 Ninja DP evaluations and two QP rescues.
  Its trace records 6,528 accepted factor calls and no fallbacks. Seven
  uninstrumented full-process pairs preserve all 5,600 compared coefficients
  and rescue counts, but their median runtime ratio is 0.967791 with a 95%
  interval of [0.946202, 1.008584]. This does not establish a speedup. The
  complete inconclusive series is retained. A separate fixed 21-pair series
  resolves a modest gain: ratio 0.968095 with a 95% interval of
  [0.958491, 0.982138]. All 16,800 compared coefficients, accuracy estimates
  and rescue counts match; all samples are retained. Independent audit
  recomputes the result from raw timings and binds source/runtime identities.
  The private Fortran production port is undergoing validation; the latest
  fully validated numerical revision remains b44d20e7.
- The production QP helper now reproduces all 250,880 retained FFV oracle
  outputs exactly. Forty-seven polynomial tests pass, including 408 compiled
  guard/IEEE cases; twenty exporter/terminal tests pass. Independent helper
  and exporter reviews find no blocker. A fresh CLI QCD3 export selects the
  expected 136 private calls and passes all 100 Ninja/NinjaRS points; Ninja
  reproduces the sealed original rows exactly, and both backends retain
  98 DP evaluations and two QP rescues. This dirty-source diagnostic is retained
  separately. Fresh NLO generation selects nine empty-prefix rank2 factor
  calls and passes the complete FO build, soft/collinear checks and twenty
  normal Ninja pole controls. A separately retained bounded driver passes
  twenty forced-QP pole controls each for Ninja and NinjaRS; these unchecked
  H1 results establish QP execution, not stability acceptance. The production
  port is ready for its intermediate commit, followed by refreshed clean
  generation/proofs and full-suite collection.
- The fresh QCD3 diagnostic also exposes an outstanding NinjaRS runtime cost:
  its single warmed traversal is about twice the candidate Ninja time, despite
  identical DP/QP counts. This is not final acceptance timing. A separate
  complete warmed-run profile attributes about half its samples to MP tensor,
  reduction, masters and geometry work; numeric transport is below one percent.
  An isolated allocation-only bridge study and an independently validated DP
  factorization study are investigating this gap. No runtime gate is claimed.
- Corrected-original reference collection completes the first six core streams,
  then reaches the unchanged 7,200-second startup timeout during QCD4 CutTools
  initialization, before any accepted QCD4 measurement. The failed full-suite
  run and its earlier results are retained. A fresh full-eight-case retry is
  running with an 86,400-second phase timeout and the same points, numerical
  settings, CPU and frozen collection tools. Copied pre-existing output files
  from the timed-out worker are not measurements.
- The guarded QP port is committed and pushed as f4bffc61e. All thirteen fresh
  core/control CLI exports and clean builds now pass from that exact revision.
  Their physical inputs are preserved, model postimages match the retained
  correction exactly, and every library/module/link signature matches the prior
  certified runtime. Independent live rehashing confirms one common complete
  runtime group per backend. Refreshed representative validation now passes
  eight targets across all four backends, with exact fresh-process parity and
  actual CutTools, Ninja and NinjaRS QP execution. A fresh relocated internal-copy
  build passes DP, genuine QP and checked-stability evaluations with its private
  installation hidden; all 48 coefficient comparisons pass. All four generated
  controls now pass all 21 corrected-cohort pairs at both repetitions. Twelve
  rotation, mass/scale rebinding, fresh-context and near-singular variant streams
  pass eight recomputed comparisons (1,600 coefficients), with fresh/retained
  NinjaRS rows exact. Complete eight-process validation remains outstanding.
- Separate DP factorization passes all six original/capture-only/factored
  100-point Ninja/NinjaRS streams. Capture-only results are exact; changed
  association remains within the original tolerance. Both backends still
  rescue points 44 and 80, so the rescue-reduction hypothesis is negative.
  Pure Fortran kernel timing has mixed results and does not justify promotion;
  an uninstrumented full-process timing plan remains held.
- An isolated allocation-only MP tensor prototype preserves all recorded values,
  tracked precisions and signed zeros in 1,392 actual reduction requests, along
  with exact 256/384-bit convergence outputs. Its original failing NaN assignment
  test and diagnosis remain retained: arbitrary NaN-source assignment is outside
  the implemented finite zero/one scratch-reset domain. The prototype is not
  installed or in production, and its runtime benefit has not been measured.
- A separate capture-only experiment preserves all 100 complete QCD3 results,
  accuracy estimates and statuses exactly, then records 1,392 complete native,
  compensated and QP calls at the two rescue points. All 464 semantic keys match;
  native/compensated input bits agree. Both coefficient and geometry bundles
  change through normal QP construction, so a later replay must separate them.
  No mixed-precision controller or reporting change is authorized by this probe.
- The fixed 21-pair DP factorization study completes on CPU185 after
  CPU5 and CPU212 fail the unchanged idle preflight without producing samples.
  One retained host-only survey selects the otherwise idle exposed physical
  core before measurement. All 42 invocations and 16,800 compared coefficients
  pass, with unchanged rescue points. Runtime ratio is 1.013546, with a 95%
  interval of [0.996598, 1.030949]: no demonstrated gain, so DP is not promoted.
  All samples and rejected preflights are retained. The quiet window is released;
  the corrected-original collector continues unchanged on CPU207.
- Source audit finds that MP transport's exact-rational widening canonicalizes
  negative zero. Direct precision widening can preserve both the original finite
  dyadic value and zero sign. The physical effect on retained amplitudes is not
  established. The separate correction now passes fourteen finite boundary
  fixtures at 256/384 bits, complex branch and nonfinite controls, GNU REAL16
  read/pack/TRANSFER checks, all eighteen bridge tests, and all 1,392 stored
  requests with 11,136 unchanged output fields. Independent source review passes;
  the exact validated patch is applied. Affected installation/runtime refreshes
  remain outstanding, and the currently installed archive is still unchanged.
- The signed-zero correction is committed and pushed as `9f13d10cc`.
  A fresh actual-CLI installation and relocated standalone/NLO refresh is
  prepared under `policy-v2/install-export-abi3-zero-sign-v1`, with one build
  thread and explicitly recorded CPU affinity. It remains held during the
  isolated tensor timing window. The already sealed eighteen-test Rust proof
  is reused only after checking exact archived and installed source identities.
- A new isolated Float256 study freezes interpreted, native-denominator and
  full-native factory selections in one private test binary. It keeps the
  corrected parser, tensor recurrence, masters and thresholds fixed. The full
  native path's different reciprocal arithmetic requires convergence and
  analytic pole checks. No native prototype, compilation or timing result is
  claimed by the source review and held plan.
- The matched eight-cell stored-input replay now completes all 3,712 calls,
  with 1,392 exact reproduction controls and an independent exact-dyadic audit.
  Rounded QP coefficients on unchanged DP geometry leave the dominant native
  versus generic discrepancy essentially unchanged. Distinct master thresholds
  and the historical zero-sign parser remain explicit confounds. This does not
  justify a numerator-only DP retry; no retry policy is changed. The replay's
  separate seal retains all calls and the offline verifier's serialization
  corrections without changing or rerunning numerical results.
- A separate source-only Fortran census finds adjacent producer/updater fusion
  opportunities, but neither proves their stored arithmetic/exception behavior
  nor measures a speedup. Strict environment-query sharing covers only modest
  regions. Both remain research leads, with the negative DP timing unchanged.
- The isolated MP allocation patch now has a fixed twenty-one-pair callback
  measurement: ratio 0.907561, with a paired 95% interval of
  [0.898288, 0.912750]. All forty-two blocks reproduce 86,436 complex outputs
  exactly in value, tracked precision and signed zero. Unsubtracted dispatch
  control overhead is 0.00070086% of the baseline median. Independent and root
  offline audits reproduce the result. The old-bound parser, rejected CPU
  preflights and explicit affinity change remain in the sealed evidence.
  This is a 9.24% kernel gain, not a full-process result or production change.
  A separate corrected-parser combination must pass its own validation first.
- User requested a pause on 2026-10-20. The goal is paused after preserving
  active compilation/reference jobs with SIGSTOP and recording their exact
  identities. Fresh corrected-transport installation and both CLI exports
  passed; copied-output runtime checks remain pending. See `references/PAUSE.md`
  before resuming; do not silently restart jobs or reuse timing intervals that
  span the pause.

## Final improvement round and validation freeze (user resumed 2026-10-21)

The user resumed the preserved work and asked to wind down optimization after
the current round. Complete only the already prepared allocation/transport and
Float256 native-dispatch studies, accepting changes only with numerical and
measured evidence. Finish the offline reuse census without starting another
cache or tensor algorithm prototype. Then freeze the implementation for an
extensive regression and benchmark campaign.

Add meaningful MG7 regression tests for uncovered interface, export, installer
and reporting behavior. Retain the eight-process correctness/performance gates
and extend the final report with loop-induced amplitudes, numerical stability
comparisons, actual backend/rescue usage, and difficult-point precision controls.
Keep performance samples separate from correctness/reference diagnostics and
never count initialization intervals spanning the pause as measured speed.

Consolidate all final changes on `madloop_ninjars`, retain reproducible inputs
and measurements, and clean temporary/build files out of the publication set.
Commit and push tested milestones. Claim achievement only after the original
gates and the expanded validation pass; leave failed or unresolved gates visible.
After achievement, wait for the user's instruction to open a pull request.

The preserved job identities were checked before resuming the same processes.
See `.benchmark-work/resume-20261020T2217Z.json` and each experiment's separate
resume metadata. The earlier pause note is a historical checkpoint.
- The corrected transport's private actual installation, relocated internal
  standalone DP/QP/stability calls, NLO twenty-point pole cancellation and
  soft/collinear checks now pass. Its sixty-three-file evidence seal and live
  archive/executable rehash pass; earlier proofs retain their earlier identities.
- The user also requested upstream monitoring. NinjaRS and oneloopmaster's
  default/main refs still equal the pinned revisions at this checkpoint. Add a
  read-only repository checker and repeat it before final freeze/publication;
  a changed pin requires isolated compatibility and numerical validation, a
  new committed lockfile, and fresh runtime/benchmark provenance. Do not float
  dependencies or silently update a frozen measurement. No external recurring
  schedule has been configured in this session.
- The user requested an immediate second pause. Active native compilation,
  original-reference collection, broad regression children and the temporary
  read-only upstream watcher are suspended; completed proofs and new tests
  remain saved. No final improvement was promoted. The latest
  `references/PAUSE.md` checkpoint identifies the new records and outstanding
  IO-comparison failures for investigation on explicit resume.
- The user resumed on 2026-10-21 and requested orchestration across the expanded
  agent team. Final work is delegated to native-kernel validation, allocation
  process measurements, legacy regression repair, reference recovery, loop-induced
  and stability supplements, reporting, upstream/integration checks and independent
  review. Root controls commits, numerical promotion, MP-slot/quiet-window handoffs
  and completion gates. No new optimization families are added.
- Paused controllers disappeared before this resume. Exact surviving compile
  identities are recovered without relabeling compilation duration; the orphaned
  reference run is preserved as interrupted and replaced by a fresh unchanged
  full-reference collection. All paused regression descendants are gone; new
  regression runs are independently logged. See the new resume ledger before
  using any older PID checkpoint.
