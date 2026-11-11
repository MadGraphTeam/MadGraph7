# TIRust / MadLoop benchmark

**Current status.** Production registration `0fcd2cd873d6` has eleven closed
ordinary cases: all 66 lanes pass `3e-6`, while 54 pass the separate `1e-8`
target. The [41-page benchmark PDF](benchmark.pdf) preserves the original 35
report pages and four separately bound CMS chronology pages, then appends
[two pages of actual shared-host timings](experiments/current-source-five-backend-timing-report-1/rendered-1/README.md)
for four SM processes and all five reducers, including a fresh QCD 2→4
four-lepton process. All twenty timing lanes pass the retained `3e-6` and
ordinary `1e-8` numerical gates; these descriptive timings are separate from
local performance acceptance. The reusable
[20-row timing CSV](experiments/current-source-five-backend-timing-report-1/rendered-1/timing-summary.csv)
retains the exact medians, unscaled MADs, paired-cycle ratios and accepted
precision counts. The
later [private matched-input QP comparison](experiments/matched-native-input-qualification-1/README.md)
agrees in Born, finite and both poles at the saved forward fixture, with
maximum relative disagreement `2.014e-16`. Its explicit backend-8 input-policy
derivative is unpromoted; original legacy 8's failure remains historical.
A later [private checked CMS20 replay](experiments/normal-checked-cms20-qualification-1/README.md)
also passes MadLoop's automatic DP-to-QP rescue and independent stability checks,
retaining six failed DP reductions before the accepted QP result. That fixture
uses cache disabled. Independent amplitude truth and generic production coverage
remain unqualified. Final precise timings remain
[Hua-Sheng’s local TODO](campaigns/entry-latch-cache-guard-1/timing-handoff-1/LOCAL_TIMING_TODO.md).

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
The worker suppresses Fortran unit 6 for post-READY timed `RUN` batches;
stdout diagnostics from those batches are unobserved. The current ordinary
numerical matrix instead uses the final steady warmup-only traversal, completed
before that redirection, so its console diagnostics are retained. The numerical
parser checks statuses and coefficients without attributing or gating every
console warning per call. Internal retry counts and hardware FPE flags remain
unobserved; accepted statuses and independent coefficient agreement remain the
numerical evidence.
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

The [archived 19-page PDF](history/report-before-invariant-symmetry-qualification/benchmark.pdf) presents closed evidence for the older measured candidate
`split-dp-terminal-1`: all eleven processes, all six lanes per process,
and 100 saved points per lane pass the `3e-6` criterion. The complete matrix
contains 66 lanes. The `1e-8` misses in QCD three- and four-particle cases
and the EW four-particle case remain visible; accepted precision
is reported separately from internal retries, which this dataset does not observe.

The [sealed shared-host appendix](campaigns/split-dp-terminal-1/provisional-retention-2/sealed-1/index.json)
retains three alternating pairs over all 100 points: seven chain comparisons and
22 standalone comparisons complete, five comparisons time out, and 32 are not
started within the budget. Total elapsed time is 1,790.210 seconds. These are
descriptive observations for the older candidate, without CPU reservation,
confidence intervals or a `≤1.05` acceptance claim. Initialization, actual batch
durations and accepted backend/precision counts are retained separately.

Final precise timings are **deferred to Hua-Sheng on his local machine**, as
recorded in [the user amendment](PLAN-AMENDMENT-2026-11-10.md). The latest generated
Fortran resets reduction-failure flags at each independent amplitude call and
avoids creating an unused TIRust context during event clearing. Its
[promotion checks](experiments/fortran-lifecycle-promotion-1/README.md) pass;
the successor registration `entry-latch-cache-guard-1` binds revision
`0fcd2cd873d6` and numerical identity
`3c13ad88b825d9d19b1030b62b317c0f31759e002d8dfdb9050142301a7cae24`.
The runnable [successor local handoff](campaigns/entry-latch-cache-guard-1/timing-handoff-1/LOCAL_TIMING_TODO.md)
provides fresh installation, all eleven six-lane numerical qualifications,
precise timing and a separate local PDF. Its 66 fresh local numerical lanes and
precise timing acceptance remain pending; shared-host observations are separate.
The [fresh shared-host QCD/EW 2-to-1 checks](campaigns/entry-latch-cache-guard-1/n1-closed-replay-1/README.md)
pass at both `3e-6` and `1e-8`: twelve lanes, 100 saved points each, all accepted
results in the selected backend's DP mode. Five more successor processes now
have [independently audited closed checkpoints](campaigns/entry-latch-cache-guard-1/QUALIFICATION_PROGRESS.md):
the Higgs and Higgs-pair controls, QCD top-pair production, EW muon-pair production,
and the mixed QCD/QED control. Both three- and four-particle processes have also
closed: all eleven cases and 66 lanes pass `3e-6`; 54 lanes also pass the separate
`1e-8` target. QCD three- and four-particle production retain finite target
misses; EW four-particle production also retains pole target misses for some
reducers. TIRust accepts QP results at respectively 13, 86 and 60 points in QCD
three-particle, QCD four-particle and EW four-particle production. The final EW
checkpoint selects five completed original lanes and a separate fresh 100-point
TIRust run, retaining the user-interrupted 74-point attempt without concatenation.
Precise timing remains pending.
The preceding 31-page PDF covers the `invariant-symmetry-1`
registration, at revision `2c13246ffa63`, with numerical
identity `f217762db032b8b755c0f62171bedcb0e9f6e09eac8128d7f8529d15e5dac8c8`.
Its numerical report is bound to a sealed terminal eleven-process checkpoint
and immutable source-specific selection/dossier. Those 66 lanes do not qualify
the changed Fortran source. The preserved 35-page PDF retains that evidence and
adds the separately bound eleven-case current-source supplement and the
current-source QCD three-particle raw stability study. Its immutable progress-3
selection includes both four-particle checkpoints, all 66 regression passes and
the retained ordinary-target misses. Precise timing remains deferred. The runnable
[preceding-source local timing handoff](campaigns/invariant-symmetry-1/timing-handoff-4/LOCAL_TIMING_TODO.md)
documents fresh installation, local registration, numerical replay, timing and a
separate local PDF for that preceding source. Handoff 4 binds the tracked `bin/madgraph` launcher;
handoff 3 remains preserved as its predecessor. Its tests are not a local measurement. The preserved
[handoff 2](campaigns/split-dp-terminal-1/timing-handoff-2/LOCAL_TIMING_TODO.md)
belongs to the older source; the historical quiet-window controller remains unrun.

The production QP adapters receive denominator masses already formed as
`M2L=M**2` in complex DP by the optimized callers; Ninja, NinjaRS and TIRust
share those DP values. TIRust widens them exactly but does not yet reconstruct
them from the MP model. The low-level `mtirust` `tensor_mp` entry accepts genuine
complex-QP masses, and its 113-to-256-bit transport and contraction remain QP.
Scale construction also differs: TIRust uses `REAL(MU_R,QP)**2`, whereas NinjaRS
uses `REAL(MU_R**2,QP)`. Native scalar inputs across adapters are therefore not
claimed bit-identical; the retained backend-8 scale discrepancy stays explicit.

The [private QP mass correction](experiments/fortran-qp-mass-current-qualification-1/README.md)
passes six qualification stages for the frozen QCD three-particle process.
Six fresh workers retain all 100 learning and 100 final points, with identical
amplitudes, accuracy, statuses and observed exception states across the mass
variants and native-cache settings. Paired native journals confirm 1,118
recovered 105-bit bottom-mass squares and 72 changed mass-dependent invariants;
shifts, scales and 95,030 complex numerator coefficients remain identical.
Separate recording controls preserve genuine 113-bit mass arguments. The patch
remains unapplied, and the packet retains its failed inventory build and a
separate fresh-CLI startup failure. Broader models, independent physical finite
truth and performance remain unqualified.

A later [fresh-CLI follow-up](experiments/fortran-qp-mass-fresh-cli-qualification-1/README.md)
closes actual MG CLI generation, explicit source review, compilation and a fresh
normal checked 100-point run for `loop_sm; u u~ > z g g [virt=QCD]`.
All seven changed driver/numerical units compile with the original flags; all
800 coefficient and 200 accuracy words agree with the corrected reference,
including 87 DP and 13 automatically rescued QP points. The historical startup
failure and generation's pending-review status remain unchanged; separate later
reviews qualify this successful chain. This uses genuine QP integrands and
cache disabled. The five-file patch remains unapplied; other models, backends,
loop-induced processes and performance acceptance remain open.

A [closed ordinary-QP transport profile](experiments/fortran-qp-transport-profile-qualification-1/README.md)
retains three fresh workers, each with the unchanged 100-point learning traversal
and the same DP and automatically rescued QP anchors. All coefficient, status,
accuracy and observed IEEE state words agree. At the QP anchor, native reduction
occupies 92.33% and output decoding plus mapping 5.32% of the recorded sequential
QP intervals; this excludes host/numerator work and is not a full-amplitude timing.
Seven exact primary QP repeats already hit the native cache. These observations
establish neither a new optimization nor performance acceptance. The portable
file-only verifier reproduces all three modes and preserves both preparation
failures separately.

The optional [Fortran QP bulk-decoder experiment](experiments/fortran-qp-output-bulk-parser-qualification-1/README.md)
passes its compiled recording-provider unit gate: 135 executed cases and 14
lexical-only controls, with all 401 executable invocations retained. It checks
all 42 topology/rank shapes, exact binary128 words, malformed records, scalar
fallback, status/error/cache behavior and observed IEEE state. The intentional
inexact trap agrees across original, candidate and observer variants. Both the
earlier runtime-binding failure and corrected native-error expectation remain
recorded. Portable raw replay and twelve independent integrity/relocation tests
pass. Production is unchanged; the subsequent physical study is recorded below.

The [physical decoder study](experiments/fortran-qp-output-bulk-physical-qualification-1/README.md)
uses `u u~ > z g g [virt=QCD]`, five fresh full100 learning workers, and three
102-call parity streams. Original, candidate and wire observer agree exactly
in public amplitude, accuracy, status and observed IEEE words, including
automatic QP rescues. All 3,248 successful native QP batches, containing 818,720
records, meet the proposed bulk eligibility conditions; these are potential
eligibility counts, not instrumented candidate branch counts. Seven alternating
timing pairs per anchor give median candidate/original ratios `0.999108` for
native DP and `0.996739` for automatic QP. The effects are smaller than paired
scatter, so the candidate remains unpromoted. Portable replay retains every
build, learning, parity, wire and timing observation. These shared-host anchor
measurements qualify neither whole-process performance nor local acceptance.
The separately frozen 41-page PDF is unchanged.

The [current-source small-process DP profile](experiments/fortran-dp-n1-profile-qualification-1/README.md)
uses `u u~ > z [virt=QCD]`, four instrumentation modes and eight fresh full100
learning traversals. All 28 timed batches, covering 358,400 amplitudes, return
checked TIRust status 219. Exact-word checks cover every pre-timing point and
each batch's final100 traversal; status and counter checks cover every call.
The clock mode adds a median paired `1.94%` overhead. Its median scopes per
amplitude are `2.195 us` for kinematic preprocessing, `1.425 us` for mapping,
and `9.131 us` for the combined primary/check C-call scope. The latter is nested
inside the `9.720 us` complete module scope and must not be added to it.
Empty-probe estimates are retained without subtraction. The ordinary Fortran
interface recipes lack optimization flags; this profile motivates a separate
controlled compiler experiment. It establishes no optimization gain, current
NinjaRS comparison or local performance acceptance. Portable replay preserves
the initial missing-tool build failure and all subsequent qualification records.

The [phase-space repair profile](experiments/fortran-host-repair-profile-qualification-1/README.md)
measures the existing Fortran repair in the same small process, preserving its
bodies, compiler flags and numerical results. Four controls and 358,400 timed
calls give a median inclusive repair scope of `6.075 us`, containing `5.627 us`
of nested MP repair. Its median paired fraction is `6.7567%` of the clocked
whole-amplitude time. Every timed status passes; retained final-point words and
IEEE observations match the baseline. These overlapping scopes and separate
probe measurements are retained without subtraction. This is runtime
attribution, with no optimization gain or performance acceptance claim.

The subsequent [strict wrapper compiler experiment](experiments/fortran-tirust-interface-o3-qualification-1/README.md)
changes only `TIRUST_interface.f` to add `-O3 -fno-fast-math` in private builds.
All coefficient, accuracy, status and observed IEEE words match on the full100
small-process cohort and the full100 `u u~ > z g g` cohort, including its 13
automatic-QP points. Seven paired timings give candidate/original medians
`0.988074` for the small process, `1.008736` for the three-particle DP anchor and
`1.002603` for its automatic-QP anchor. These shared-host samples show no
consistent overall benefit. The 179,424 timed calls and all raw comparisons are
retained; the three-particle measurements time anchors, not its full cohort.
The compiler change remains unpromoted. Prospective makefile patches remain
unapplied, and build registration, formal compiler-policy compatibility and
broader qualification remain open. Default flags and the frozen PDF are unchanged.

The experimental adaptive-precision bridge is an isolated development build,
not a private repository or the installed backend used in this campaign. It
tries 256-bit MPFR first and retries an eligible conditioning refusal once at
512 bits. Its [debug Fortran runtime](experiments/adaptive-qp-fortran-qualification/runtime1/evidence-index.json)
and [release Fortran runtime](experiments/adaptive-qp-release-fortran-qualification-v2/runtime1/evidence-index.json)
now supply bounded transport, dispatch and cache proofs. A separate
[fixed-mode physical replay diagnostic](experiments/adaptive-qp-physical-replay-qualification-1/README.md)
is retained with unchecked fixed-QP status 120. The separate
[masked checked replay](experiments/adaptive-qp-checked-trap-proposal-1/qualification-1/evidence-index.json)
accepts its threshold point with status 323 and serves 600 QP requests, while
both trapped workers fail with SIGFPE; its combined trap gate remains false.
The fixed-mode result and the checked result have distinct stability scopes.
Neither establishes independent physical truth, per-call working precision or
performance. Actual backend 8/9 real-Z Born controls also retain their unchecked
fixed-QP status 120.

A later [guarded private release build](experiments/cayley-guarded-adaptive-build-qualification-1/evidence-index.json)
completed successfully and produced DSO SHA-256
`69722452c234f494f0aa436fa24bbabfeae59100c03b05a1f0b76336cc17886b`.
The compilation package preserves the prior 4 GiB allocation failure, the
separate 600-second timeout and the earlier `c5de24e6…` library's identity.
Build success establishes source and compilation provenance. The separately
[closed guarded physical replay](experiments/cayley-guarded-adaptive-physical-qualification-1/evidence-index.json),
index SHA-256 `715f436c704fe1cf1f7de9c2f15a2ca1b17ba8f7453956196ceb88bc4828cd90`,
retains masked checked and fixed-QP outputs bit-identical to their respective
old adaptive answers, alongside two trapped SIGFPE failures at the later
Gram-indicator and MPFR dilogarithm sites. Combined trap qualification remains
failed; actual register operands and the precise IEEE exception subtype were
not captured. Its rejected-original comparison remains a nonreference
diagnostic rather than independent truth. The main installed `eb3d67fd…`
bridge is distinct from both private builds. No adaptive production promotion,
general physical-stress qualification or performance acceptance follows.

The [symmetric-invariant Fortran template is promoted](experiments/ew-invariant-symmetry-prototype/v2/promotion-qualification-1/index.json).
Four tests pass, including three compiled recording-fixture methods and one
Python permutation method; this does not requalify earlier exports. The
[strict installer version-type correction is applied](experiments/installer-metadata-promotion-1/index.json),
with 27 tests and 37 subtests recorded as passing. That receipt explicitly notes
that no raw pytest log was captured. Earlier ABI/interface rows retain their
original test scope.

The [all-component independent Feynman-parameter check](experiments/rank6-feynman-parameter-all-components-qualification-1/index.json)
corroborates all 210 finite Cartesian components of one retained synthetic
N=6/rank-six fixture. All 420 real/imaginary point estimates pass both `3e-6`
and `1e-8`, with maximum TIRust residual `9.625e-20`. The analytic-tail-only
classification separately gives 420 agreements at `3e-6`, and 357 agreements
plus 63 inconclusive parts at `1e-8`. It excludes finite node, weight and
arithmetic error; there is no whole-program floating-point certificate.
The earlier seven-component calculation, both native answers and every
COLLIER disagreement remain preserved. This fixture corroboration does not
qualify physical high-rank amplitudes or general kinematics, stability rescues,
threshold, soft/collinear or complex-mass stress. Those physical gates remain open.

A separate [exact rational pole oracle](experiments/rank6-rational-pole-oracle-1/README.md)
corroborates all 210 nonzero UV components of the retained massive two-propagator,
rank-six fixture. Its 15 full/partial and pole-selection rows pass an independent
complex symmetric `1e-8` coefficient gate. Nine algebra tests and portable raw
replay pass. This retrospective audit preserves the inherited IEEE mask 37 and
all historical failed gates; the massive fixture has zero IR residues. The
included all-component triangle IR specification was uncompiled and unrun at
that packet's freeze; its subsequent qualification is recorded separately below.

The [independent rational UV/IR qualification](experiments/rank6-rational-ir-poles-qualification-1/README.md)
passes 24 DP/QP, full/partial and pole-selection requests for synthetic massless
triangle and scaleless bubble fixtures, each covering all 210 Cartesian
components through rank six. All 12 full/partial pairs have identical pole
words. The triangle has nonzero double-pole witnesses in every component;
the scaleless bubble retains separated, cancelling UV and IR residues.
Maximum scaled errors are `3.698e-15` in DP and `1.258e-34` in QP. Fresh initial
IEEE flags are zero; model initialization raises only inexact. Portable replay
preserves the original failed build and separate successful validation, and
independently joins recorded native outputs to Fortran words. Scope remains
the historical generated adapter with caching disabled; physical amplitudes,
checked rescues, general stress and performance acceptance remain open.
Production and the separately frozen 41-page report are unchanged.

The [current-source relocated deployment check](experiments/current-source-deployment-runtime-qualification-1/README.md)
uses actual standalone and FKS/NLO CLI exports at `0fcd2cd873d6`, the installed
`eb3d67fd…` TIRust library and the unchanged NinjaRS archive. After private
dependency prefixes were removed and both exports moved, nine fixed-mode
workers completed; seven comparisons against the two same-export CutTools
controls agree within `4.34e-16`. The NLO entry point rejects an incomplete
finite result. The actual ELF interpreter and initialized runtime providers,
card restoration and cleanup are verified. These unchecked 110/120 results
retain one final point per worker after four driver evaluations; they qualify
internal deployment, relocation and the virtual-entry guard. Stability, rescue,
cache invalidation, Monte Carlo integration and runtime performance remain
outside this check.

The [current-source kinematic-trace build packet](experiments/current-kinematic-trace-build-qualification-1/README.md)
retains fresh CMS generation, signed corner mappings and four unmodified or
instrumented CMS/EW executables at `0fcd2cd873d6`. All 258 build commands and
four compiler-selected loader-resolution queries passed with unchanged protected
inputs and clean process shutdown. This closed packet contains zero amplitude
evaluations and qualifies compilation and library resolution only. Later
observational and physical checks remain separate from this build record.

The separate [current-source trace runtime packet](experiments/current-kinematic-trace-runtime-qualification-1/README.md)
retains 30 fresh fixed-mode workers: four ordinary CMS/EW DP/QP comparisons
pass exact tracing parity, while all six stress point/precision comparisons
retain failed-contribution flags despite identical variant observations. The
failed stress stage exits 1. Its CMS traces show spacelike invariants near
`-2.5e-7` and `-2.5e-11 GeV²` projected to zero before native refusals; EW traces
retain the mass-shell projection and photon control. These observations do not
establish refusal causation or an accuracy fix. All ten input/status analyses
replay from retained bytes; no failed tensor output supplies an accuracy claim.
The 35-page PDF remains its separately frozen ordinary-campaign snapshot.

The subsequent [isolated causal replay](experiments/kinematic-projection-causal-replay-1/README.md)
reproduces the CMS rank-two, four-propagator refusal at both stress points in
DP and QP using the original installed Fortran adapter and TIRust library.
Replacing only the two symmetric zeroed invariant entries with their captured
spacelike values changes all four refusals to successful statuses. Exact input
reconstruction and the error's two-by-two Gram matrix establish this scoped
projection cause. No tensor coefficients or amplitudes were inspected by the
replay, so a general Fortran correction, amplitude accuracy and performance
remain unqualified. Its portable verifier checks the retained inputs, ten raw
protocols and twenty loader maps without running native code.

The [compiled integer-certificate check](experiments/spacelike-kernel-native-qualification-1/README.md)
passes 4,287 synthetic DP/QP and malformed-input cases, including all 32
combinations of the five standard sticky IEEE flags. All 1,572 accepted cases
satisfy the conditional exact error enclosure. The sampled raw values are not
necessarily produced from their vectors, so this check does not establish
physical kinematics or qualify the source-DAG bounds and amplitude caller.

The [standalone caller-bound qualification](experiments/caller-bound-native-qualification-1/README.md)
then checks the compiled Fortran bound factory and read-only environment probe:
174 cases cover all 104 route/mode combinations plus invalid-input, boundary
and boost controls; eleven token controls also pass. Actual bounds and table
orientation match the reference, with unchanged observed flags and controls.
This qualifies the standalone factory on the present GNU/Linux host. The full
amplitude's entry-to-use environment and numerical results remain unqualified.

The later [generated-wrapper/module ABI fixture](experiments/generic-geometry-wrapper-abi-qualification-1/README.md)
passes all 240 direct calls: 180 available bounds and 60 conservative rejections.
It checks exact source-bound slots and integer bounds, DP/QP storage words,
mass arguments, legacy forwarding, tokens and observed IEEE state. Seven build
commands and one fixture worker close successfully; portable raw replay and
fourteen integrity/relocation tests pass. This uses a recording legacy target
and supplied histories, without a physical amplitude. The included source
counterexamples expose a remaining parent-caller interval gap between capture,
producer batches and consumption. Full caller lifetime and scientific admission
remain false until those parent effects and possible momentum writes are proved.

The [standalone exact-geometry helper](experiments/generic-equality-postcondition-qualification-1/README.md)
checks a conservative alternative: integer arithmetic proves that each supplied
momentum component equals its signed external-momentum sum. Both checked and
optimized Fortran builds pass all 1,698 cases in twenty explicitly observed
rounding, exception and FTZ/DAZ settings: 67,920 calls, with no skipped settings.
Their complete transcripts are identical, and observed environment fields and
restored state remain unchanged. Independent rational and integer-capacity
checks retain ten exact equalities that the bounded helper conservatively
refuses. Snapshot ownership, actual caller inputs, the final floating-point dot
product, physical amplitudes and performance remain separate open gates.

The [standalone snapshot fixture](experiments/generic-owned-snapshot-qualification-1/README.md)
tests the Fortran module's word transport, staged publication, reset and separate
DP/QP state. Checked and optimized builds agree on all 1,480 observations,
3,680 API outcomes and 94,720 integer state words across twenty environment
settings. Actual runtime mappings and observed environment restoration pass.
These unit observations do not establish physical constructor completion,
current-point provenance, q/dot behavior or full amplitude correctness.

The [private CMS amplitude trial](experiments/kinematic-invariant-amplitude-qualification-1/README.md)
compiles the Fortran prototype and preserves the ordinary point's exact
original/candidate observations. The first stress point retains failed DP
contributions and a QP disagreement with CutTools; the next CutTools DP
measurement times out, leaving seven requests unrun. The failed comparison and
incomplete stage remain explicit. The prototype is not promoted.

A separate [universal QED double-pole audit](experiments/cms-universal-qed-pole-audit-1/README.md)
reconstructs the existing model and supports the prototype's QP double pole
at that stress point; CutTools returns approximately half the predicted value.
This checks one coefficient under the documented reconstruction assumptions.
At that earlier boundary the remaining coefficients still required
adjudication. The subsequent single-pole and later matched-input results below
provide separate evidence; the failed original stress stage remains failed.

The subsequent [single-IR-pole audit](experiments/cms-universal-qed-single-pole-audit-1/README.md)
also supports the candidate QP coefficient, within `5.23e-16` of the FKS
prediction. It explicitly relies on the retained restored QP inputs and the
intended UV-renormalized amplitude; runtime coupling bits and UV cancellation
were not independently measured. These analytic pole audits do not determine
the finite term; the later matched-input probe supplies finite agreement under
its explicitly private input policy.

The [legacy Ninja/NinjaRS reference probe](experiments/kinematic-legacy-reference-adjudication-1/README.md)
completes both workers but obtains no valid reference: NinjaRS reports five
failed box reductions, while Ninja's final Laurent coefficients are nonfinite.
The logs establish QP interface use and preserve every blocked comparison.
The later [matched-input probe](experiments/matched-native-input-qualification-1/README.md)
uses a separate private backend-8 derivative with the candidate certified
invariant/symmetry policy and aligned scale expression. All 52 native pairs,
104 native adapter calls and both whole-worker/bootstrap gates pass. All
point-20 Born/finite/pole orders agree within `2.014e-16`; finite and single-pole
returned binary64 words differ by one ULP, while Born and double-pole words are
identical. Candidate-9 output words also match the earlier eligible candidate.
Its recorded canonical numerator is the Fortran contraction buffer, not an
argument to TIRust’s tensor-only API. Original legacy backend 8 is unchanged.
This establishes matched-input agreement, not an algorithmic winner, an analytic
finite oracle, normal checked stability, generic source qualification or performance.

The separate [original-input EW replay](experiments/ew-original-refusal-replay-1/README.md)
reproduces both point-2, rank-three refusals and retains their complete original
errors: ill-conditioned Gram subsets `[1,2,3]` in DP and `[0,1,2,3]` in QP.
The inputs, original adapter and library are unchanged. This identifies the
reported refusal conditions; it establishes neither their numerical cause nor
a mass-projection fix or amplitude accuracy. Both failed results remain open.

The [current-source cache lifecycle proof](experiments/subprocess-cache-lifecycle-successor-qualification-1/README.md),
index SHA-256 `e9df673cf450c07a9d28ee8aa50f31c075abfb8dc8fdc13f98b28bb6d13c4a57`,
uses fresh CLI exports and full numerical builds at `0fcd2cd873d6` with the
unchanged `eb3d67fd…` native DSO. Four cache-disabled/enabled, plain/observed
workers agree exactly across 76 checked native-DP physical calls, eight warmup
calls and 112 separate DP/QP tensor-key requests. Seventy-two physical calls
enter with deliberately poisoned software failure latches; the called prefix
resets its own latches and preserves the other prefix. Raw traces require
successful primary/independent-check DP passes, event invalidation, distinct
subprocess registrations and guarded context recreation. Independent tensor
checks bypass primary stores, whose following primary requests still hit.
All 83 external commands close successfully; portable raw replay and 19
integrity/relocation tests pass.

The [preceding-source proof](experiments/subprocess-cache-lifecycle-current-qualification-2/evidence-index.json)
remains unchanged at `2c13246ffa63` with 96 tensor-key requests. The new proof's
physical scope remains real-SM N≤3/rank≤2, while QP coverage uses exactly promoted
DP tensor inputs. Physical-QP rescues, rank six, EW, loop-induced, poles-only,
general stress accuracy and performance remain separate qualifications. The
41-page PDF is a separately frozen snapshot preceding this lifecycle replay.
Native-DP/rescued timing cohorts and all precise performance gates remain
unqualified. Complete delivery remains false.

The [preceding-source physical-stress terminal capsule](experiments/physical-stress-current-portable-4-terminal-2/evidence-index.json),
index `6fc2035649ba0387f0d4457784ae883592f13bb7e3b6f9c33b99e038b81f7146`,
reuses the exact [preparation/preflight snapshot](experiments/physical-stress-current-portable-4-preflight-1/evidence-index.json),
index `49cb1ec298bbd2f8b942b02c3e81b60b53cd2f14a5cdaa56d0fa93452c175c22`.
Eight fresh CLI exports, full builds and mass preflights belong to that earlier
snapshot at `2c13246ffa63` (`invariant-symmetry-1`), not the successor
`0fcd2cd873d6`. The terminal native stage exited 1 on a lock timeout before backend 8
launched. Four threshold streams were host checked, one request was deferred
before launch and 19 were not started. The four checked streams contain eight
qualified final point evaluations and 12 observed warmup calls, without proving
the hidden evaluation count. Five of their six pairwise comparisons fail the
`3e-6` criterion, including retained single-pole disagreements; baseline and
current CutTools agree. Full six-lane coverage and agreement remain unqualified.
The comparator's historical `ordinary_target_passed` key applies the `1e-8`
target to the whole two-point stream. Neither the earlier preparation-only
snapshot nor this checked subset closes numerical stress accuracy. The EW
analytic oracle has not been requalified for these physical inputs.

The separately closed [preceding-source CMS attempt-1 retention capsule](experiments/current-cms-replay-retention-1/evidence-index.json),
index `7ef5f6a3f7a88449d67d56cb21e80e5ad1e75433e8e066ab07e3abf2744237ed`,
retains successful fresh source1 preparation/preflight and a failed native
orchestrator (exit 1). Private-link validation rejected a generated external
directory symlink before any RUN-stage physics loader, worker or amplitude
evaluation. All six planned 21-point streams, including 42 order outcomes per
final stream, remain unexecuted in this attempt; its result is failed rather
than prepared or passed. Its exact failed ancestry remains preserved.

The separately admitted [CMS attempt-2 capsule](experiments/current-cms-replay-retention-2/evidence-index.json),
index `aa39809c5d6cde25367454bb1fd9c3500b9aaa9b14b976fc1ff42bb0783714d3`,
completes six controller requests with orchestration exit 0 but remains
numerically incomplete. Both CMS attempts bind the preceding `2c13246ffa63`
source and are not successor stress qualification. Two workers timed out during startup and four were
rejected during warmup; none reached READY, RUN or final traversal. All 252
final order outcomes remain unexecuted, with zero complete final streams or
host-qualified requests. The warmup partition has 234 accepted, eight rejected
and ten unexecuted outcomes; those individual accepted outcomes do not qualify
a request. Twelve bootstrap order rows, 242 observed warmup order rows and five
W03 lines are separate retained observations, not a count of hidden amplitude
evaluations. Preparation/preflight are inherited from failed attempt1 through
checkpoint `b3e38892b2cc86b679c9ed6c05bbad0b125a65ba6a67cb41eca6824dabb09949`;
this successor performs no new preparation builds or model probes. The failed
attempt1 capsule and all its unexecuted rows remain visible. Broad CMS accuracy,
an independent finite-term oracle and performance remain open.

The [rank-two threshold geometry audit](experiments/ew-threshold-rank2-geometry-audit-1/README.md)
checks 72 contraction identities from two closed historical tensor cases.
It reproduces supplied-invariant recurrence consistency, while retaining the
different momentum-derived geometry and missing independent pinch controls.
Ten pure tests and independent replay pass. This does not resolve the current
EW threshold refusal or justify relaxing production conditioning guards.

The [invariant-classification audit](experiments/invariant-classification-audit-1/README.md)
recomputes two saved forward CMS points and the separate EW threshold group.
The CMS transfers have zero time difference and genuine nonzero spacelike
squares, but satisfy the existing energy-normalized zeroing test. The EW group
instead exposes a DP-formed mass-square mismatch after massive-shell snapping.
Both pure recomputations pass independent replay; no threshold or numerical
guard change is qualified. The [symbolic-routing research note](qualification/benchmark-audit/kinematic-provenance-research.md)
identifies existing topology metadata and a trace-only next experiment; it
separates routing provenance from the unresolved DP/QP mass contract.

The bounded [closed QCD n3 investigation](experiments/invariant-symmetry-qcd-n3-target-analysis/portable-analysis2/index.json)
retains the same 1,200 rows and 4,800 coefficient values as its prior closed
result, including 28 finite-only `1e-8` target-miss points. Born and pole
comparisons pass `1e-8`; the largest CutTools/TIRust symmetric finite residual
is approximately `2.806420647e-6` at point 93 and the largest non-CutTools pair
residual is approximately `8.595156396e-8` at point 39. This seven-input offline
analysis supplies no independent truth claim, new native/QP evaluation or main
checkpoint replacement or new QP result.

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

The report uses an explicit immutable closed-evidence selection. Its selected
dossier binds the checkpoint inventories and excludes active numerical
directories before opening them. The old 19-page PDF and complete 4,294-input
ledger are preserved under
[report-before-invariant-symmetry-qualification](history/report-before-invariant-symmetry-qualification/README.md).
Reconstruct its original renderer inputs from the pinned Git commit to
reproduce that historical PDF; current helpers do not replace those inputs.

The preceding `invariant-symmetry-1` campaign closed all eleven ordinary cases at `3e-6`. Its
sealed checkpoint is `8f5f7110…`; performance acceptance remains deferred.
The additive retrospective view binds all eight plots and all twenty accepted
backend streams from the [closed four-case plot capsule](stability_appendix_c/retrospective-main-closed11-1/README.md),
index `1138e7f644aec5a0eda96959b47eb17694344a2ad92caa09b2abee6202e23dab`.
The post-rescue composite estimates and finite disagreement against current
candidate CutTools use ordinary 100-point streams, not raw-DP probes or asserted
paper-cut-qualified inputs. They establish neither true error nor performance.
The preceding [17-page report milestone](https://github.com/MadGraphTeam/MadGraph7/blob/4d24cebce1/docs/benchmarkfolder/benchmark.pdf)
remains available at its recorded commit.
The separate [closed raw stability pilot](stability_appendix_c/raw-dp-pilot-5-1/README.md)
and [four raw figures](stability_appendix_c/raw-dp-pilot-5-figures-1/README.md)
use 100 cut-qualified points per process and host forced-DP/card-selected reducers.
They measure three-probe internal consistency, not true error or observed reducer
identity/internal precision. Eighteen lanes have 100 available diagnostics;
TIRust has 24/100 in QCD Zgg and 0/100 in QCD ZZgg. Their unavailable outcomes
remain unknown bounds, not measured failure fractions. Nineteen completed requests
are not nineteen accuracy passes. The preceding [25-page retrospective report](https://github.com/MadGraphTeam/MadGraph7/blob/ba047d9165/docs/benchmarkfolder/benchmark.pdf)
remains available at its recorded commit.

The separate [current-source QCD Zgg stability study](stability_appendix_c/current-source-qcd3-raw-delivery-2/README.md)
adds a [reproducible comparison plot](stability_appendix_c/current-source-qcd3-raw-figure-1/current-source-qcd-n3-raw-dp-stability.pdf)
for the successor Fortran source. CutTools, Ninja, COLLIER and NinjaRS each have
100 available diagnostics; TIRust has 95. Its five unavailable points remain in
the denominator and give unknown-tail bounds, including [0, 0.05] at 1e-3.
These three-probe, host forced-DP/card-selected diagnostics measure internal
consistency; they establish neither true error nor observed internal precision.
The exact data, failures and independent replay are retained separately. The
current PDF adds this study on page 35; the immutable 33-page report's
progress-1 selection remains a separate reproduction anchor.

The [first-failure capture and standalone replay](experiments/tirust-first-failure-capture/closed-2/COMPACT_README.md)
retains six exact Zgg tensor inputs. All six reproduce their reported
Gram-conditioning rejection in fresh and repeated contexts; failed tensor
outputs were never read. ZZgg times out during bootstrap, leaving six primary
inputs missing. The original capture failure and later inventory-admission
failure remain visible. These boundary diagnostics establish neither independent
accuracy nor the exact cause of every internal fallback failure.

The optional private diagnostic bridge now has a separately
[completed build](experiments/tirust-first-failure-capture/diagnostics-private-build-complete-3/README.md)
and an [audited twelve-call metadata replay](experiments/tirust-first-failure-capture/diagnostics-replay-closed-1/README.md).
The replay preserves all twelve original refusals for six captured Zgg requests,
with exact runtime providers and no failed tensor outputs read. It identifies
small-Gram expansion nonconvergence and rejection of an alternative PV route
by its Cayley-cancellation check. This diagnoses those refusals without supplying
accepted tensor answers or resolving physical accuracy. The
[first build failure](experiments/tirust-first-failure-capture/diagnostics-private-build-failed-1/README.md)
and [second build timeout](experiments/tirust-first-failure-capture/diagnostics-private-build-failed-2/README.md)
remain preserved. The installed libraries and production precision policy are
unchanged.

The [published 31-page preceding-source PDF](https://github.com/MadGraphTeam/MadGraph7/blob/913a746201eb6ef806ccb219c890a7f3486d6322/docs/benchmarkfolder/benchmark.pdf)
retains its exact source, ledger and plots at commit `913a746201eb6`. Its SHA-256 is
`91ceeae99968df742e7e141467e3bab321394a0512328be7df9be0f5b86d559a`.
The current view keeps that campaign's scientific evidence and adds a separately
bound `entry-latch-cache-guard-1` supplement: eleven closed cases and 66 lanes.
All lanes pass `3e-6`; 54 also pass `1e-8`. The QCD n3/n4 and EW n4 ordinary-target
misses, accepted DP/QP counts and interrupted EW attempt remain visible. The
guard-only cache experiment is descriptive shared-host evidence; precise local
acceptance is still TODO.
The [33-page milestone](https://github.com/MadGraphTeam/MadGraph7/blob/48b664a029d35acb2646b8820f9c654cfed6e852/docs/benchmarkfolder/benchmark.pdf)
and its progress-1 selection remain unchanged at that commit. The preserved ordinary-report PDF
has 35 pages: historical raw figures remain on pages 27–30, current-source
numerical progress is on pages 32–33, cache-guard scope and local TODOs are on
page 34, and the separately qualified current-source raw study is on page 35.
The progress-3 selection retains the preceding campaign's gates and adds the
closed eleven-case current-source snapshot. The independently reviewed report
proposal and a small layout amendment are retained with the release receipts.

A new old-source selection/dossier binds the preceding precision TODO through
its exact archived path; the published selection/dossier remain immutable.
Rebuild and verify the archived ordinary view into a custom output
(the canonical report now uses the append builder below):

```sh
source tools/tirust_env.sh
current_report_dossier=campaigns/invariant-symmetry-1/report-snapshots/closed11-entry-cache-supplement-1.json
current_report_supplement=campaigns/entry-latch-cache-guard-1/report-supplements/progress-3/selection.json
python3 -B docs/benchmarkfolder/build_report.py --output /tmp/ordinary-35-page-report.pdf --campaign invariant-symmetry-1 \
  --campaign-report "$current_report_dossier" --supplement "$current_report_supplement"
python3 -B docs/benchmarkfolder/build_report.py --output /tmp/ordinary-35-page-report.pdf --verify-only --campaign invariant-symmetry-1 \
  --campaign-report "$current_report_dossier" --supplement "$current_report_supplement"
```

The archived 39-page PDF preserves those exact 35 pages from
`history/report-before-cms-stress-supplement-1/` and appends four separately
bound CMS diagnostic pages. Pages 36–38 preserve the earlier failed/incomplete
trial and conditional pole audits. Page 39 records the later eligible private
matched-input result without reclassifying earlier failures. The exact combined
source/render packet is `experiments/cms-stress-report-supplement-2/rendered-1/`;
it retains both original Typst sources, data, plots, public joins and commands.

Reproduce the composite into a fresh custom output and verify it:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/build_stress_report.py --output /tmp/benchmark-with-cms-stress.pdf
python3 -B docs/benchmarkfolder/build_stress_report.py --output /tmp/benchmark-with-cms-stress.pdf --verify-only
python3 -B docs/benchmarkfolder/build_stress_report.py --output /tmp/benchmark-with-cms-stress.pdf --verify-only --verify-pages
```

The builder authenticates the archived ledger's 12,252 inputs and the sealed
four-page artifact. It verifies same-version paired 144-dpi RGB pixels, per-page
layout text, geometry and fonts for all 39 delivered pages. This is not PDF
object-graph equivalence or a new annotation/link proof. The composite ledger
retains source pins, tool versions, commands, warnings and the page proof.
File-only verification launches no programs; explicit page checks use bounded
read-only document tools. The optional `--rebuild-supplement` compiles the known
`earlier/supplement.typ` and `later/matched-page.typ` sources separately, preserving
the earlier 1/3–3/3 footers, then joins them before the same checks. The older
builder guards canonical composite output; explicit custom output remains
available. No report command launches native physics or qualifies local timing.

The canonical 41-page PDF now preserves that exact 39-page composite under
`history/report-before-five-backend-timing-1/`, including its old ledger and
owning builder, and appends the sealed two-page
[five-backend timing tail](experiments/current-source-five-backend-timing-report-1/rendered-1/README.md).
The first-three and fresh four-lepton raw timing capsules remain separate,
with complete portable replay, runtime/card/cohort pins and all ten balanced
cycles. Each lane uses all 100 saved points. The CSV and plot use unchanged
analyzer statistics: wall ms per physical evaluation, median ± unscaled MAD,
and median of ten within-cycle backend/NinjaRS8 ratios. READY initialization
is separate. All timed accepted output calls were DP; internal retries and
hardware FPE flags were not observed. These shared-host timings carry no
confidence interval, baseline/rescue-chain acceptance result or local ≤1.05
claim. The complete registered local campaign remains
[Hua-Sheng's precise timing TODO](campaigns/entry-latch-cache-guard-1/timing-handoff-1/LOCAL_TIMING_TODO.md).
The separate private checked-CMS20 note does not qualify production timings.

Rebuild the timing tail and reproduce the complete report into a fresh output:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/build_timing_report.py --output /tmp/benchmark-with-five-backend-timings.pdf --rebuild-supplement
python3 -B docs/benchmarkfolder/build_timing_report.py --output /tmp/benchmark-with-five-backend-timings.pdf --verify-only
```

The matching owner authenticates both scientific timing seals, the optional
CMS evidence seal, the unchanged 39-page archive and sealed source/data/plot
inputs. It checks all 41 pages using paired144dpi RGB pixels, per-page text,
geometry and fonts; this is not PDF object-graph or new annotation/link
proof. The ledger retains exact tool versions, commands, warnings and page
proof. File-only verification runs no programs; optional `--verify-pages`
uses bounded read-only document tools. The ordinary and stress builders
refuse to overwrite this canonical timing composite; explicit historical
custom outputs remain available. The main historical Typst source and
NinjaRS report PDF/source remain unchanged.

Dossier creation and report building never launch new numerical measurements.
Each retrospective and raw plot has a separate page. The original numerical
and delivery gates remain unchanged.

For a future refresh, close the selected runs, create a new selection and dossier
path, and preserve this report's ledger/PDF before rebuilding. Do not overwrite
a previous dossier or use partial numerical rows as completed evidence. The
closed10 PDF, ledger and Typst sources are preserved under
`history/report-before-closed11/`; its immutable selections and dossier remain
in the campaign report-snapshots directory.
The closed11 PDF, ledger, context and changed report sources from commit
`772b15ed39cc` are preserved under `history/report-before-closed-provisional/`.

The report builders do not run numerical measurements. The current timing
`report-build.json` binds the archived 39-page PDF and ledger, sealed two-page
timing tail, final PDF and page-preservation receipt. The archived composite
ledger separately retains its original35-page ledger and four-page CMS tail. The archived original
ledger retains its exact Typst version, creation timestamp and font search paths. Typst 0.15.1, Libertinus
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

For this combined view, use the exact admitted selection from the sealed
[source proposal](experiments/report-appendix-c-raw-pilot-integration-proposal-3/README.md).
Copy it to a fresh append-once path, then collect with the promoted canonical
helpers; an existing path is never overwritten:

```sh
source tools/tirust_env.sh
python3 -B - <<'PY_COPY_SELECTION'
from pathlib import Path
import hashlib
root = Path("docs/benchmarkfolder")
raw = (root / "experiments/report-appendix-c-raw-pilot-integration-proposal-3/selection.proposed.json").read_bytes()
if hashlib.sha256(raw).hexdigest() != "7970d67ddf68547612edf6c384d1c4928c84cc695a8b7c3949acdbd1426b178b":
    raise ValueError("Raw-pilot selection differs from its reviewed pin")
with (root / "campaigns/invariant-symmetry-1/report-snapshots/closed11-raw-dp-pilot-selection-1.json").open("xb") as stream:
    stream.write(raw)
PY_COPY_SELECTION
python3 -B docs/benchmarkfolder/campaign_report.py \
  --campaign invariant-symmetry-1 \
  --selection campaigns/invariant-symmetry-1/report-snapshots/closed11-raw-dp-pilot-selection-1.json \
  --output campaigns/invariant-symmetry-1/report-snapshots/closed11-raw-dp-pilot-1.json
```

The canonical collector command requires prior source promotion. The proposal's
private dossier is a reviewed preview and is not substituted for this fresh
collection. Both selections pin the same terminal full11 checkpoint, lifecycle,
failed partial stress capsule, incomplete CMS capsule and retrospective plots;
adding raw diagnostics changes none of those qualifications. Rendering and visual review remain separate steps.

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
resolution of retained stress failures. The separately frozen preceding-source
`2c13246ffa63` cache lifecycle proof is closed in its native-DP and tensor-key
scope. The successor's guard/reset tests, fixed-mode parity and ordinary
comparisons retain their narrower scopes; the preceding full-source lifecycle
proof has not been rerun for those changes. That scoped lifecycle pass does not
close the physical stress or performance gates. A passing eight-process runtime gate
alone cannot complete the integration goal.
