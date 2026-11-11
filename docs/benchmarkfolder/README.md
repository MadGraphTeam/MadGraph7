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
the changed Fortran source. The current 35-page PDF retains that evidence and
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

The [preceding-source cache lifecycle proof](experiments/subprocess-cache-lifecycle-current-qualification-2/evidence-index.json),
index SHA-256 `212b6046a48030f26e12facfbd85b4179df0036633f14bbbc38c2788df376964`,
passes exact parity across four workers, 76 checked native-DP physical calls
and 96 separate DP/QP tensor-key requests. Its registered source is
`2c13246ffa63` in `invariant-symmetry-1`; the native DSO is the unchanged
`eb3d67fd…` installation. It has not been rerun for the successor's entry-latch
reset and cache guard. The successor's promotion tests, fixed-mode parity and
guard-only lifecycle measurements retain their narrower scopes. Physical-QP
rescues, rank-six, EW, loop-induced, poles-only and runtime remain outside the
preceding study's lifecycle scope.
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
and its progress-1 selection remain unchanged at that commit. The current PDF
has 35 pages: historical raw figures remain on pages 27–30, current-source
numerical progress is on pages 32–33, cache-guard scope and local TODOs are on
page 34, and the separately qualified current-source raw study is on page 35.
The progress-3 selection retains the preceding campaign's gates and adds the
closed eleven-case current-source snapshot. The independently reviewed report
proposal and a small layout amendment are retained with the release receipts.

A new old-source selection/dossier binds the preceding precision TODO through
its exact archived path; the published selection/dossier remain immutable.
Rebuild and verify this explicitly selected two-source view:

```sh
source tools/tirust_env.sh
current_report_dossier=campaigns/invariant-symmetry-1/report-snapshots/closed11-entry-cache-supplement-1.json
current_report_supplement=campaigns/entry-latch-cache-guard-1/report-supplements/progress-3/selection.json
python3 -B docs/benchmarkfolder/build_report.py --campaign invariant-symmetry-1 \
  --campaign-report "$current_report_dossier" --supplement "$current_report_supplement"
python3 -B docs/benchmarkfolder/build_report.py --verify-only --campaign invariant-symmetry-1 \
  --campaign-report "$current_report_dossier" --supplement "$current_report_supplement"
```

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
