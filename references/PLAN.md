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
- Support `install ninjars --update` to rebuild the local installation from the
  current checkout's pinned dependencies after `git pull`. Keep `--force`
  compatible and preserve the old installation if build or validation fails.
- Pin the final upstream versions confirmed on 2026-10-21: NinjaRS
  `aa6a6307dd081dc3714f9accebc0df9e5561ae73`, oneloopmaster
  `8ec256fc26cb361b20e10f59c19beb35717ef32c`, and Symbolica/numerica/graphica
  `9cdc2949cdb153478c1eca0d40690dc0abfc616e`, with a committed lockfile.
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
- Root reverified and pushed the corrected-parser/allocation candidate's
  twenty-two tests, twenty-one golden lanes and both complete 1,392-request
  exact replays. This is candidate correctness evidence; production still has
  the eighteen-test original tensor bridge until a measured promotion.
- Nineteen new MG7 boundary tests and the read-only upstream checker now pass
  together with the existing installer/statistics tests (thirty tests total).
  A three-line direct-Python diagnostic fix rejects an absent NinjaRS path with
  the actionable installer message. Both requested upstream main pins remain
  unchanged; a separate newer Symbolica arithmetic patch is recorded but not
  independently substituted for NinjaRS's explicit dependency.
- The controllerless original-reference worker was terminated only after exact
  identity and missing-pipe-peer verification. Historical data are unchanged.
  One fresh full-eight helper-reference collector runs on CPU207, QCD4/EW4
  first, under the unchanged 24-hour timeout and frozen protocol. Its sealed
  recovery ledger is separate from the live incomplete report. Final timing
  remains held until its references and the final numerical source are ready.
- The bounded MG7 regression campaign now passes ninety cases: eighty-nine
  passes and one explicit existing GoSam Python 3 skip. Exactly twenty-two
  individually reviewed expected-output fixtures and one deterministic FKS
  test-helper setting were updated; no production arithmetic was changed.
  The sealed logs, failed earlier attempts, fixture review and report adapter
  bind 102 tested production paths to the final candidate. Rust reduction,
  amplitude agreement and performance remain separate outstanding gates.
- Final report, precision-reader and loop-induced supplement preparation now
  pass fifty-five independently rerun MG7 tests. The tools reject stale-runtime
  relabeling, normalize convergence using physical Laurent coefficients, and
  report estimated accuracy separately from observed discrepancies. Two literal
  original loop-induced CLI exports, two private corrected helper builds, two
  hundred fixed points and twelve threshold points are prepared. Numerical runs
  await final source/install freeze; synthetic PDF tests certify rendering only.
- The user additionally requested two more challenging 2-to-4 tests and complete
  genericity. Add `g g > t t~ g g [virt=QCD]` with `loop_sm` and
  `u u~ > e+ e- mu+ mu- [virt=QED]` with `loop_qcd_qed_sm_Gmu`, beyond the
  unchanged eight core processes. Freeze twenty deterministic ordinary points
  per process and a small fixed rotation/scale control set at 1 TeV, real masses
  and zero widths; compare all four candidate backends with the existing
  Born/Laurent/split-order tolerance. Record actual rescues and costs. These
  supplemental correctness tests do not replace the core performance gates or
  modify the already running private timing study. Independently audit that
  production optimizations depend only on generic rank/algebraic structure with
  guarded general fallbacks, never process identities or benchmark inputs.

- User requested another pause on 2026-10-20 at approximately 23:55 UTC.
  All active task jobs and the read-only watcher are suspended with verified
  identities in `references/pauses/20261020T235607Z`. Twelve allocation pairs
  completed exactly; the current timed arm is interrupted and unaccepted.
  Installer/stress/report extensions remain explicitly untested drafts.
  No final optimization or performance gate is promoted; await user resume.
- The user explicitly resumed on 2026-10-21, retaining the final-round wind-down,
  full validation/report/cleanup requirements and the hold on opening a PR.
  Verify exact preserved identities before any process action. Retire the
  interrupted allocation timing worker without resuming its timed interval and
  collect a fresh fixed series. Resume the original correctness collector only
  after confirming its live controller/pipes, sources and deadline. Complete
  the already drafted software regressions before the next quiet timing window.
- The resumed installer transaction regression reproduces three original
  failures and passes all thirty-seven focused tests, including seven new
  failure/retry/interruption cases. Independent review binds the final sources.
  The final report/stress/precision/supplement software suite passes all
  seventy-six tests under independent rerun; physical measurements remain
  separate. The interrupted allocation jobs were retired without SIGCONT.
  A fresh byte-identical fixed twenty-one-plus-seven-pair study received its
  quiet/MP handoff on 2026-10-21 at 00:05:46 UTC. No historical sample is reused.
- The fresh allocation study completed all twenty-one QCD3 and seven EW2
  pairs with fifty-six successful fresh workers and exact coefficients,
  accuracy estimates, statuses and raw rows. Independent 100,000-resample
  analysis reproduces QCD3 ratio 0.9685183929 with 95% interval
  [0.9568779827, 0.9936397041], and EW2 ratio 0.9976395568 with interval
  [0.9947156586, 0.9993386992]. Both local promotion gates pass. Root reverified
  the 751-file evidence seal; these results do not pass the separate final
  original-Ninja runtime gates. The allocation patch remains held until the
  final combined source freeze and numerical tests.
- The final native screen completed all seven triplets and three warmup
  streams with exact own-arm controls and unchanged inputs. Independent paired
  analysis found no demonstrated gain: denominator/interpreted ratio
  1.02169807, interval [0.973591667, 1.045665774]; full/interpreted 1.007070304,
  interval [0.981059834, 1.040109539]. The declared rule retains interpreted
  kernels; neither native feature nor factory selection is promoted.
  This closes optimization research for the requested final round.
- Apply the measured tensor-allocation change and test-only panic/final
  precision modules, then freeze the exact committed candidate for one Rust
  test binary, actual installer/copied standalone/NLO proof, and fresh core,
  loop-induced, threshold, stability and extra two-to-four campaigns. The new
  final test modules are not counted as passed until compiled and executed.
- The first final Rust attempt stopped at four test-only owned-left Float
  operator errors before producing a test executable. Preserve its failed
  six-file proof unchanged. Correct the three expressions without changing
  arithmetic order, thresholds or production code, then retry from a new
  committed revision and fresh proof/work namespace. An explicit repository-
  local Cargo cache option reuses compiled dependencies; source/executable
  proof remains bound to the corrected revision and no failed run is passed.
- Corrected candidate `d05b59f3277ce0d727cb2816d8b7ad45107e58c4` passes
  the final one-binary proof: twenty-four ordinary tests, sixty-three physical
  fixture calls, and all 1,392 retained requests at 256/384 bits. All 11,136
  historical ASCII fields match exactly; exact-rational physical Laurent,
  rational diagnostic and finite-split checks pass without raw sign changes.
  Root independently verifies the seventeen-file successful proof. The first
  failed attempt remains unchanged. The actual CLI installation has started
  from this exact archived source, followed by relocated standalone/NLO tests.
- The final actual CLI installation and copied-export proof passes from that
  same source: ABI 3, compiled Fortran native DP/compensated DP/binary128 calls,
  relocated standalone return codes 110/120/218 with CutTools agreement, twenty
  NLO pole checks and soft/collinear checks without failures, and forty-two
  archived-source Python checks. Root verifies the sealed sixty-eight-file
  proof and publishes archive
  `800a941faf3cdfdab764fdffe1d5ab73f84984a786aa194596b753af6b6f5d42`
  and its nine matching artifacts through the tested installation transaction.
  Both complete copies of the former installation and the alias journal are
  retained. Earlier numerical proofs keep their original identities.
- Freeze the resulting committed generator and installed artifacts for fresh
  eight-process, loop-induced and extra challenging two-to-four generation.
  Generation/build durations remain separate from warmed performance. Final
  correctness, stability, original-Ninja runtime gates and the final report
  remain outstanding; no goal-completion or PR claim is authorized by these
  installation checks alone.
- Fresh CLI exports and clean builds for all eight core processes pass from
  generation revision `b0ea105a04afa34304d437d2956e4d4672360670`. All physical
  cards, fixed points, model correction postimages, dependencies and 6,009
  literal-original archive members remain unchanged. Every process shares the
  complete runtime descriptor
  `8b454519cb6818ef1fafe37378930c276f7e85394a2333ebc9fdb00e75726cff`.
  Independent re-execution of the inventory audit reproduces it exactly.
- All sixty-four public C ABI precision calls against the installed library
  pass. The beyond-double complex control reproduces its final-source 256-bit
  outputs and passes the independent 384-bit convergence comparison. The
  refreshed Fortran representative proof passes eight new candidate targets
  with exact persistent/fresh parity, genuine rescue, loop-induced amplitudes
  and mixed split orders; original/helper roles are revalidated separately.
- Both challenging two-to-four processes, both loop-induced processes and the
  complex-mass control have fresh actual CLI exports and clean builds. Retain
  the stress preparation's first failed path precheck and the one-line
  operational correction for existing broken template links. It changes no
  generator or numerical code. Subsequent source-bound numerical collections
  may retain raw stress streams before the core dataset exists, with no
  placeholder core identity; attach the real identity in a new manifest and
  validate through the unchanged final reader before any campaign claim.
- Full core numerical comparisons, warmed timing gates, remaining focused
  controls, final profiling and the reproducible final PDF remain outstanding.
  Generation, installation, representative checks and diagnostic timings do
  not certify those final gates. Do not open a PR before the user's instruction.
- Candidate/helper loop-induced and threshold diagnostics pass all 35,616
  comparisons at the fixed 100/100/12 points. Complex-mass and mixed-order
  controls pass all 11,760 comparisons, including every split order and fresh
  candidate parity. Rotation, mass/scale rebinding, context reset and near-
  singular controls pass a further 1,600 comparisons.
- A separate explicit-quadruple Fortran mass/scale test passes 240 comparisons:
  retained and finalized contexts agree exactly and agree with the ordinary
  DP results. Its documented direct mode reports 120 with accuracy -1; it is
  a precision/state test, with no stability or performance acceptance claim.
  Normal benchmark rescue and stability settings remain unchanged.
- Actual Collier data exposed a postprocessing mismatch: its shared-setting
  comparison incorrectly included a derived precision annotation absent for
  that backend. Preserve the first rejection and raw data. A bounded reader
  correction follows the existing core-policy rule, validates every applicable
  QP flag and preserves all physical settings. Eighty-one software regressions
  and an independent source review pass. Keep this analysis-only change
  separate from the frozen numerical source and retain its exact patch for
  reproduction; final assembly still requires the actual core dataset.
- Final publication software and evidence are ready for the remaining physical
  campaign: all forty-three report tests pass, including loop-induced timing
  ratios, confidence intervals, DP/QP usage and optimization attribution.
  These synthetic rendering checks do not certify a final numerical report.
  The six approved native plaintext ignore rules are applied; the actual
  source guard still reproduces all 2,601 frozen production paths. The exact
  EW-model source archive and self-contained analytic leading-pole capsule
  have reproducible retained inputs and explicit provenance limitations.
- Early challenging-QCD two-to-four calibrations agree across CutTools, Ninja
  and Collier in all 480 comparisons at twenty physical points, counting total
  and sole split rows separately. The corrected-original QCD4 CutTools/Ninja
  calibrations agree in all 800 comparisons at one hundred points. Preserve
  finite point 99 at 0.875 of the permitted tolerance, with its actual accepted
  statuses and separate internal estimates; no rows or thresholds are changed.
  Ninja also passes the source-derived leading-double-pole diagnostic. These
  calibration checks do not replace the pending four-backend repeated streams,
  rotation/scale controls or warmed runtime acceptance campaign.
- Keep the numerical source, generated outputs, installed runtime, physical
  cards and points frozen while those collections run. Hold orchestration HEAD
  at `9d33d4650a2e6079d007c365f1c12bb74a2def3f` until the three declared stress
  variants release it. Documentation and analysis-only edits remain unstaged;
  original collection source snapshots and every failed attempt stay retained.
- The final report still requires complete eight-process and ggH/ggHH timings,
  both challenging stress processes, the actual core-linked runtime provenance,
  nonempty validated final profiles, and two byte-identical actual renders plus
  visual inspection. Profile presence is an explicit publication requirement
  beyond the report's scientific campaign Boolean. Preserve all measured-worker
  binding chains until profiling finishes. Commit and push reviewed final work
  on `madloop_ninjars`; claim success only after every required gate passes,
  then wait for the user's separate instruction before opening a PR.
- The corrected-original QCD4 reference subsequently stopped with a real
  pure-Collier failure: points 17 and 52 return 410 above the fixed stability
  threshold, and point 99's finite CutTools/Collier difference is 1.158 times
  the approved tolerance. The latter is one physical event duplicated in total
  and sole split rows. All values are finite; independent source/card/model
  audits and root's full 2,400-comparison recomputation confirm both failures.
  Preserve the complete failed attempt and do not relax thresholds, drop points,
  relabel failure statuses, add an undeclared fallback, or claim a passing core
  cohort. The existing strict publisher cannot accept that failed reference.
- Investigate with a separate frozen-candidate pure-Collier diagnostic on the
  identical full stream and public direct-QP controls for the difficult points.
  The candidate diagnostic is authorized once on CPU 207; its first launcher
  affinity precheck failed before any build or physics, and the separately
  authorized corrected attempt preserves that record. Native NinjaRS stress on
  CPU 190 remains the sole Symbolica/MP worker. Higher-precision controls are
  being prepared, without substituting them for normal acceptance or timings.

### User priority amendment — 2026-10-21

The user instructed: “Continue as planned, but focus first on getting a
validated MadLoop+NinjaRS version. Then if you uncovered issues with other
reduction tools (e.g. COLLIER), then just post an issue with an MRE to the
corresponding github”.

- Put the eight-process NinjaRS validation first. Keep every phase-space point,
  Born/Laurent/split-order comparison, stability threshold, contribution and
  genuine QP rescue requirement. Validate NinjaRS against accepted, source-bound
  CutTools/Ninja references; reuse matching retained reference data with its
  actual calibration/repetition scope before scheduling new reference work.
- Retain the original/candidate Collier failures and all four-backend comparison
  results as reported findings. Do not mark Collier successful, omit its bad
  points, or require an external Collier fix before validating NinjaRS. Preserve
  the historical strict v2 policy and its failed attempts; make this new
  prioritization explicit in the next versioned validation/report policy.
- Retire the long auxiliary ordinary-stress attempt at its complete calibration
  checkpoint, preserving the 960 passing all-four comparisons and the interrupted
  repetition as user-priority retirement, not a numerical failure. Release the
  sole Symbolica worker for the eight-process NinjaRS correctness campaign;
  defer the remaining challenging-process repetitions/rotation/scale work.
- Use the already generated frozen exports for the NinjaRS-only correctness
  phase, with 100 unchanged points per process and two retained commands.
  Keep actual correctness CPU/runtime/source identities distinct from the later
  warmed seven-pair performance gate against literal original Ninja. Its 1.05
  limit and uncertainty requirement remain unchanged.
- Once the NinjaRS version is validated, prepare and run minimal reproductions
  for genuine issues found in other tools, identify the responsible component
  and official upstream repository, check for duplicates, and post the issue
  with its reproducer. The user authorizes these GitHub issue posts. Do not
  assume a Collier library defect from a conservative stability rejection or
  a full-amplitude discrepancy before isolating numerator/contraction effects.
- Keep the final four-tool results, numerical stability discussion, loop-induced
  controls, reproducible styled PDF and tested milestone pushes in scope. Report
  external-tool failures honestly and link their upstream issues. Continue to
  wait for the user's instruction before opening a PR.

### Final upstream release and resume — 2026-10-21

The user confirmed that both upstream `main` branches are ready, authorizing
their update and the remaining implementation, validation and finalization.
Repository-owned clones were fetched and fast-forwarded to the exact pinned
revisions above; sibling reference clones remain unchanged. The existing goal
and its correctness, performance and publication requirements remain in force.

- Rebuild and validate the coupled NinjaRS/oneloopmaster/Symbolica update in an
  isolated prefix before publishing it as the installed backend. Final NinjaRS
  includes the MP soft-D0 relabelling, so inspect and use that implementation
  before considering any repository-owned master-provider adaptation.
- Preserve every old source/runtime-bound result and interrupted attempt.
  The previous NinjaRS core collection was paused after QCD4 calibration and
  one repetition; it is incomplete and must never be finalized as successful.
  Completed accepted CutTools/Ninja references can be reused with their actual
  scope and input bindings. Their parent statuses remain truthful.
- Rerun precision transport, failure/retry, actual installation/export and
  genuine QP controls against the new dependencies. Generate and bind a fresh
  candidate for all eight processes, loop-induced checks and challenging
  two-to-four processes. Old NinjaRS results do not certify the new backend.
- Keep one numerical MP worker at a time, and separate correctness runs from
  the quiet, pinned-CPU, alternating seven-or-more-pair timing campaign. Preserve
  the existing thresholds, full contributions, rescue and 1.05 runtime gate.
- Update the final report only from actual validated results; include failed
  external-tool comparisons and unresolved requirements explicitly. Complete
  the goal only after the required gates, reproducible PDF and milestone pushes
  pass, then await the user's instruction to open a PR.

### Four-tool timing and stability comparison — 2026-10-21

The user explicitly requests comparisons among CutTools, Ninja, Collier and
NinjaRS in `benchmark.pdf`, with tables and figures analogous to Tables 2–3 and
Figures 4–7 of https://arxiv.org/pdf/1604.01363. Inspect the optional public
script archive linked by the user for reusable measurement/plotting methods;
record its provenance and inspect scripts before running or adapting them.

- Add process generation costs, loop-diagram/group counts, maximum numerator
  rank, numerator construction and each backend's reduction cost. Label the
  paper's single-helicity, color-summed profile separately from the agreed full
  helicity/color, full-pole, rescue-enabled end-to-end timing gate.
- Plot cumulative accuracy tails for all four tools, with consistent point
  samples, denominators, uncertainty and a Ninja comparison panel. Measure
  unrecovered double-precision direction/rotation estimates separately from
  accepted rescued results; do not infer those estimates from post-rescue data.
  Include loop-induced and challenging-process comparisons. Preserve the agreed
  core inputs and tolerances; any extra study has its own explicit configuration.
- Keep rejected/nonfinite points visible and record real backend/rescue counts.
  A failed tool is a reported finding, not a successful timing or an omitted
  sample. Retained calibration elapsed times cannot stand in for a new timing
  study. The full evaluation performance gate remains unchanged.
- Continue the new NinjaRS correctness campaign first. Separate correctness
  controllers for legacy-only selectors 1/6/7 may run on CPU12 while the sole
  NinjaRS/Symbolica engine uses CPU190: source audit confirmed lazy NinjaRS
  initialization, selector-isolated retries and private process/cache paths.
  Use separate Python processes and work/report trees, exact single-backend
  selectors, one thread, and owned child groups. No concurrent final timing or
  profiling is permitted; those phases require a new explicit resource handoff.


### Latest user steering — 21 October 2026

The remaining benchmark and report work focuses on Ninja (6) and NinjaRS (8).
Do not start new CutTools or COLLIER investigations, benchmarks, debugging or
issue work. Preserve previously collected evidence with its original scope.
The integration stays generic and legacy behavior remains covered by the
existing software checks.

Use the actual MadGraph7 `profile` CLI with explicit backend selection to
measure its reduction stage and total MadLoop time per phase-space point.
Document the precise timing boundary and keep this stage study distinct from
the full warmed original-Ninja/NinjaRS performance gate. Preserve exact command
cards, settings, source/runtime identities, seeds, raw outputs and repetitions.
Use only Standard Model processes, including the agreed loop-induced and
challenging cases. Add seeded random-point native-DP stability distributions;
retain the fixed100-point correctness sample separately, with true denominators
and explicit failure/zero-event treatment.

Make timing and stability plots analogous to Figures1–3 of
https://arxiv.org/pdf/1604.01363. The latest visual reference is
https://github.com/alphal00p/ninjars/blob/main/docs/benchmark/benchmark.pdf:
large readable coloured cells, detailed numerical information, elegant tables
and legible scientific plots. Inspect its actual PDF/Typst source, retain
attribution, and compile/visually check the final benchmark.pdf reproducibly.
Do not open a PR; push the tested final branch and await further instructions.

Resource isolation: normal/rescued NinjaRS numerical work retains the sole
Symbolica/MP slot on CPU190. A separate, audited native-f64-only NinjaRS study
may use CPU12, with private GNU link guards aborting every MP/compensated call
from initialization onward. Native f64 context/master code has no Symbolica
state or shared runtime file cache. Correctness-only optimized Ninja6 can use
CPU14 with guards rejecting NinjaRS numerical entry. These parallel controls
are not timing acceptance; final timings require the quiet reserved CPU207.

Keep the validated numerical source at `856e76e669618ec8d0f799b9c8e6f86819dcd986`
through its correctness, focused controls and full-sum core/loop timing runs.
Their source guards bind the complete generator and vendor trees. Prepare the
CLI stability correction and timing metadata privately, then apply, test and
push them after those runs release the source hold. Run the requested actual
CLI profiles against that committed diagnostic revision, retaining the separate
source identities in the report. Do not weaken the guards or relabel earlier
measurements to accommodate the later CLI change.

### Finalization checkpoint — 21 October 2026

The final pinned NinjaRS core correctness campaign has completed all eight
processes, with 100 fixed points per process and calibration plus two repeated
traversals. Born, Laurent and split-order results pass the retained numerical
comparison; repeated rows are identical. Each traversal uses native DP at 768
of the 800 points and genuine higher-precision rescue at the other 32. The
terminal correctness receipt is retained under
`docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/core-correctness/`.
This certifies numerical correctness for those inputs, not performance.

The optimized legacy Ninja campaign has also completed all eight processes;
its independent final audit is pending. A separate 100-point random boosted
Z-production pilot passes Ninja/NinjaRS agreement and recorder parity, with safe
initialization and no higher-precision calls. The remaining random processes,
focused controls, challenging processes and full-sum loop-induced checks still
need their final evidence. Pilot sample sizes are not the final stability study.

Keep the seven-pair quiet timing gates, actual MG7 CLI profiles, final scientific
figures and reproducible PDF outstanding until they have actually run and been
audited. Do not claim goal completion or open a PR from these intermediate
correctness results. The latest upstream NinjaRS main adds report/documentation
changes only; the retained numerical pin has identical source/build inputs.

### User scope reduction — 21 October 2026

The user explicitly rejected the proposed thirteen-hour campaign and requested
lighter testing setups, without matching the reference paper's multiplicities.
This supersedes the exhaustive eight-process timing requirement. Preserve the
completed eight-process numerical qualification and all truthful partial runs;
do not launch the old long timing specification or expand expensive stress runs.

The new timing study uses QCD and EW 2-to-1 and 2-to-2 processes, plus the
loop-induced gg-to-H and gg-to-HH controls. Use the same first ten fixed points
for each paired backend, full helicity/color sums, poles and normal rescue,
seven paired repetitions and a half-second calibration target. The observed
measured commands may be shorter after warmup; report their actual durations. Retain
the literal original-Ninja baseline, exact source/runtime bindings, uncertainty
and backend/rescue counts. Assess performance for this declared lighter scope;
do not imply that the abandoned eight-process timing gate has passed.

Use actual MG7 CLI profiles for the same lighter processes to report reduction
and total time. Retain the matched 100-point native-DP random studies for the
affordable processes; small 2-to-3 diagnostics may be included with their actual
scope. No larger high-multiplicity random campaign is required. Keep the report
reproducible, readable and explicit about sample sizes and unresolved results.
The final source, tests, report and pushes still require review before goal
completion. Continue to wait for instructions before opening a PR.

### Installed-backend default — 21 October 2026

The user now requests NinjaRS as the default when it is installed. For ordinary
one-loop exports, prefer `8|6|7|1` when NinjaRS is available and retain `6|7|1`
otherwise. Preserve explicit user selections. Leave the existing loop-induced
reduction strategy and its `7|6|1` export default unchanged.

Prepare and test this change privately during the remaining frozen numerical
checks. Apply it together with the CLI correction after the lightweight timing
and focused checks release their source hold, then verify installed/absent and
loop-induced defaults before pushing the final production revision.

### Final profile scope expansion — 21 October 2026

The user subsequently requested the Standard Model 2-to-4 processes as well
as the loop-induced `g g > h h h` and `g g > h h h h` processes. This expands
the actual CLI profile study to the eight core QCD/EW processes and four
loop-induced Higgs multiplicities (H, HH, HHH, HHHH). Compare explicit Ninja6
and NinjaRS8, with three timing repetitions per backend and initially twenty
stability points. Run the first QCD1 pair as a cost and metadata pilot before
the remaining calls. Use the existing CLI timing method and report reduction
time as the primary comparison, with numerator construction and total time per
sample for context. State that the reduction estimate uses the CLI full-minus-
SKIP construction and its initialization subtraction. Preserve raw signed
estimates, actual adaptive sample counts, failures and uncertainty.

This expansion does not reinstate the retired thirteen-hour seven-pair
full-sum campaign. The completed six-process ten-point full-sum timing sample
and eight-process fixed-point correctness and native-DP random stability
studies retain their original scope. Do not relabel them as the expanded
profile study. Leave loop-induced production defaults and its existing TIR
strategy unchanged. No CutTools or COLLIER investigation is authorized.

The six-process timing run completed in 250.157 seconds. All numerical
comparisons pass, and all six observed paired 95% confidence intervals have
upper limits below 1.05 relative to the original Ninja baseline. This result
applies only to that declared sample. The numerical source hold is released
for the independently reviewed installed-default and CLI diagnostic changes.

### Explicit-output profiling repair and multi-Higgs extension

The additional `gg > HHH` and `gg > HHHH` processes were generated and
compiled through the actual CLI and checked at two common 1 TeV points each.
Ninja and NinjaRS agree for Born, Laurent and split-order results; maximum
finite-part relative differences are 2.42e-14 and 3.87e-13. Preserve the
original data-analysis import failure separately from its corrected read-only
finalizer and independent numerical audit; no amplitudes were rerun for that
correction.

The first actual profile pilot completed Ninja timing but failed during
stability because the existing MadLoop code used the installation directory
in place of the explicit output directory. The same defect exists in the
original baseline. Retain this pilot as incomplete, with no accepted timing
comparison, and retain its zero-exit child/reaped-resource evidence.

Repair only the stability output-directory base and retain the actual private
work directory's debug log on failure. Four portable tests exercise the real
driver-path handling; the coupled affected software suite passes 110 tests.
Keep timing, RNG and numerical algorithms unchanged. Commit and push this
source revision, then use a new source-bound twelve-case specification and
new output namespace for the two-call pilot and subsequent profiles.

### Final profile workflow qualification

The successor pilot exposed a second pre-existing two-to-one stability issue:
the existing card helper was given a subprocess directory instead of the export
`Cards` directory. Preserve that failed Ninja-only attempt separately; no
NinjaRS comparison or timing acceptance follows from either failed pilot.

Use the established Cards interface and fix stability report handling for
NinjaRS, exact-zero and extreme finite/infinite accuracy diagnostics. Preserve
raw values and the existing ordinary nonzero formulas. Reject invalid historical
cached records actionably, while allowing legitimate positive infinite accuracy
and absent QP observations. These reporting changes do not alter the numerical
engines, stability thresholds, timing stages, RNG or reduction calls.

Before the next native pilot, exercise the whole stability method using real
filesystem/card aliases, the actual checker-output parser, driver rendering,
binary pickle/cache reuse, and complete profile reports for both backends with
a mocked checker. Include genuine QP records and zero/cancellation/extreme and
invalid-history controls. Push the tested source and use a fresh source-bound
specification/output namespace for the twelve-process timing campaign.

The affected live suite passed all 135 tests without skips or errors, including
15 full stability I/O/cache/report tests and ten multi-Higgs control tests. A
subsequent one-byte whitespace cleanup leaves its tested AST unchanged. Preserve
the exact tested/final source identities in the software proof.

### Actual timing-observer and collector contract

The third pilot completed the native Ninja profile and stability command with
exit zero, but its benchmark adapter rejected missing legacy flat-card metadata.
Do not accept its timing comparison: NinjaRS was not launched. The generated
Fortran reader resolves the physical card directly or through `Cards` in the
working directory and up to five parents. Mirror that exact lookup in metadata
capture. Keep the canonical physical-card identity checks strict.

The complete retained native output also exposed a collector assumption about
bare archive filenames. Accept relative library keys by exact basename, require
at least one match, and require every matching hash to equal the pinned archive.
Preserve the original library map. Before another native pilot, replay the entire
actual-data collection and strict report reader offline, including cache,
adaptive counts, runtime/source/card identities and transitive file inventories.
The original failed result remains unchanged. These observer/collector changes
do not alter the physics, RNG, timing formulas or numerical acceptance.
