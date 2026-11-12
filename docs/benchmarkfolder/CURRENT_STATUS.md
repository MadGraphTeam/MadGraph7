# Current acceptance status — 2026-11-12

This status selects evidence for production revision
`0fcd2cd873d61d134837e1d9e87afb13a47824fd`, registered as
[`entry-latch-cache-guard-1`](campaigns/entry-latch-cache-guard-1/manifest.json).
Its exporter inventory is
`afb6e0bea64ea0932751ff4869c206bcea9477d40a39a54bb99d0f06d85b75a6`
and numerical identity is
`3c13ad88b825d9d19b1030b62b317c0f31759e002d8dfdb9050142301a7cae24`.
The baseline remains `9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f`.
Review: [draft TIRust PR #437](https://github.com/MadGraphTeam/MadGraph7/pull/437)
into `madloop_ninjars`, stacked on [NinjaRS PR #315](https://github.com/MadGraphTeam/MadGraph7/pull/315) into `main`.
The integration goal is incomplete.

The current signed source has a [fresh local timing handoff](campaigns/tirust-oem-signed-1/timing-handoff-1/LOCAL_TIMING_TODO.md).
Its source and installation are frozen separately; all 66 new numerical lanes
and eight main runtime gates await local execution. The older handoffs and
results below retain their recorded source bindings.

The [OEM signing successor](OEM_SIGNING.md) updates the current installation
pin to signed TIRust `10ce34ed8ad5b6f5f989a0e220eb3a244a71d1a1`. Its scoped activation and fresh
bridge qualification are recorded separately. Numerical acceptance evidence
below remains bound to production revision `0fcd2cd873d6`; signing does not
transfer that evidence to the new source or close the outstanding physics and
local timing gates. The canonical 44-page PDF and its frozen inputs are
unchanged.

| Evidence | Current result and scope |
|---|---|
| [Ordinary process matrix](campaigns/entry-latch-cache-guard-1/QUALIFICATION_PROGRESS.md) | Eleven CLI-generated processes, six lanes and 100 saved points per lane. All 66 lanes pass `3e-6`; 54 also pass the separate `1e-8` target. The remaining target misses and independent-reference limitations are retained. |
| [Frozen ordinary selection](campaigns/entry-latch-cache-guard-1/report-supplements/progress-3/selection.json) | Selects the closed checkpoints for this production source, including both four-particle processes, interruption history and source-specific evidence. It does not import predecessor results as new measurements. |
| [Subprocess/event cache lifecycle](experiments/subprocess-cache-lifecycle-successor-qualification-1/README.md) | Closed for real-SM `N≤3`, rank≤2 native-DP amplitudes and separate DP/QP tensor-key diagnostics. It covers prefix isolation, entry-latch reset, invalidation, independent checks and context recreation. Physical-QP rescues, EW, loop-induced, rank-six physical and poles-only amplitudes remain outside this proof. |
| [Published benchmark](benchmark.pdf) | The current 44-page PDF preserves all 43 preceding pages and adds the ordinary EW and threshold qualification page. Its [ledger](report-build.json) binds PDF SHA256 `f8454f71dcd3a8614d777a3a91a6a71aa86bb5e0cfb45062cb87d41daeb3b321`. The analytic Fortran repair timing study and five-backend timings remain preserved. |
| [Five-reducer timing appendix](experiments/current-source-five-backend-timing-report-1/rendered-1/README.md) | Twenty descriptive full-amplitude timing lanes cover four SM processes, including a fresh QCD `2→4` process. All twenty pass their retained `3e-6` and `1e-8` numerical gates. Shared-host timings do not satisfy precise performance acceptance. |
| [Stability comparisons](stability_appendix_c/current-source-qcd3-raw-delivery-2/README.md) | Current-source QCD Zgg comparisons supplement the preserved Appendix-C-style plots. Internal consistency estimates, unavailable diagnostics and independent numerical error retain distinct scopes. |
| [Generic CMS prototype](experiments/generic-owned-CMS-normal-qualification-1/README.md) | Three ordinary controls agree bit for bit. Two forward fixtures retain legacy `420` refusals and candidate-reported `323` QP acceptance. Both caches are disabled; these self-check observations establish neither independent finite accuracy nor generic production coverage. |
| [Copied-integrand physical pair](experiments/fortran-qp-mass-false-integrand-qualification-1/README.md) | With `UseQPIntegrandForTIRust=False`, original and fresh-CLI mass candidate agree exactly on 100 physical QCD Zgg points per arm, including 13 accepted QP rescues. Histories, accuracy and observed state match. Both caches are disabled; internal copied-coefficient/native-input observations and broader coverage remain open. |
| [Actual EW export and ordinary mass pair](experiments/fortran-qp-mass-ew-qualification-1/README.md) | Fresh actual CLI/export/build for Gmu `e+ e- > t t~ a [virt=QED]`; original and candidate agree exactly across 100 ordinary points and steady warmup. Pole mode 0 uses full sums and restores stored filter state; persistent filter learning is unqualified. All points use DP219. The pair passes `3e-6` and `1e-8` comparison gates; QP-mass-path coverage, independent accuracy, threshold behavior and promotion remain open. |
| [EW threshold follow-up](experiments/fortran-qp-mass-ew-threshold-qualification-1/README.md) | Backend9 production control and mass prototype both refuse the saved threshold/soft-photon point with `420`/ACC1. Ordinary control and initialization agree exactly. Four failed-buffer word differences are diagnostic only. The mass correction does not resolve EW stress; accepted QP, native-input and independent-accuracy qualification remain open. |
| [Analytic Fortran repair prototype](experiments/fortran-analytic-repair-qualification-1/README.md) | For QCD `u u~ > z` (2→1), 544 standalone DP/QP wrapper calls pass and 100 full-amplitude points agree exactly. Seven same-flag timing pairs give median candidate/control `0.95279`, with every batch >1 s. This is a shared-host prototype comparison; the patch remains unapplied and formal eight-process performance acceptance is open. |
| [Original CMS observer refusal](experiments/generic-owned-CMS-observer-refusal-1/README.md) | The original replay remains failed at its first candidate journal. Three legacy workers pass own-plain comparisons; the partial candidate exposes a DP-eligibility/actual-QP flag-semantics mismatch. Exact refusal and incomplete scope are replayable; the corrected run below is retained separately. |
| [Corrected CMS observer run](experiments/generic-owned-CMS-observer-qualification-1/README.md) | Six workers pass exact own-plain histories and journal checks: 106 calls, 16 recorded vetoes and eight consumed upper-matrix changes. Actual QP ownership is observed in modes 4/5. Only these recorded provenance branches are qualified; independent amplitude truth and general caller/numerator coverage remain open. Cache evidence is scoped separately below; legacy `420` failures remain. |
| [Limited CMS cache qualification](experiments/generic-owned-CMS-cache-qualification-1/README.md) | Twelve fresh workers across cache0/cacheall and plain/observed arms pass six parity and six policy comparisons on five saved point IDs. Recorded words and histories agree exactly; 701 visible DP master reuses and two QP tensor hits are observed. Independent checks, resets, failure censoring and accounted-budget bounds pass. Budget exhaustion, Fortran group caching for this NSQSO1 export, independent accuracy, broader caller/lifecycle coverage and performance remain outside the proof. |

The [QP-mass correction](experiments/fortran-qp-mass-current-qualification-1/README.md),
its [fresh-CLI follow-up](experiments/fortran-qp-mass-fresh-cli-qualification-1/README.md),
the [legacy-dispatch recording unit](experiments/fortran-qp-mass-non9-dispatch-qualification-1/README.md)
and the generic CMS prototype are experimental qualifications. Their source
patches remain unapplied to production. Standalone tensor, snapshot and integer
geometry tests do not supply missing full-amplitude qualifications. The
[phase-space-repair profile](experiments/fortran-host-repair-profile-qualification-1/README.md)
measures existing work; it is not an optimization gain.

Required EW threshold, QCD soft/collinear and complex-mass stress accuracy
remain unresolved. The [retained physical-stress result](experiments/physical-stress-current-portable-4-terminal-2/evidence-index.json)
belongs to preceding source `2c13246ffa63`; it cannot qualify current production.
Broader constructor/current-point/q/veto coverage, broader cache-enabled CMS
coverage, full physical integration and measured benefit must be established
before any corresponding experimental source promotion. A promotion requires
a new source identity and fresh validation, rather than reusing this source's
66 ordinary passes. Native-DP and rescued-QP timing cohorts also remain
unqualified.

Final precise timings remain
[Hua-Sheng's local TODO](campaigns/entry-latch-cache-guard-1/timing-handoff-1/LOCAL_TIMING_TODO.md):
a fresh supported installation, all 66 local numerical lanes, eight main
TIRust-first versus baseline runtime gates and separate existing-backend
regression checks. The upper 95% paired runtime-ratio bound must be ≤1.05 for
each main process. Historical qualified timing failures remain retained.
Separate diagnostic native-DP and rescued-cohort timings remain an explicit
[local follow-up TODO](PRECISION_COHORT_TIMING_TODO.md); the required local
cohort adapter is not yet qualified.

The root [qualification index](qualification.json) and [summary](summary.json)
are preserved historical snapshots, not this current source's acceptance
selection. Their references to DSO `095e`, candidate `c91cb85f` and pending
successor physics describe those earlier snapshots. Frozen checkpoint documents
may likewise describe the report page count at their creation. The current
PDF and its reproduction commands are selected by the [current report ledger](report-build.json)
and [matching report instructions](experiments/ew-qp-mass-report-1/rendered-1/README.md).
No historical failure or unmeasured result is converted into a pass here.
