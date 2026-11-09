# Integrate TIRust into MadLoop

## Goal and working protocol

**Goal:** Integrate TIRust with oneloopmaster into MadLoop as reduction backend 9, with a Fortran interface, automatic installation, UV/IR selection, accelerated poles-only evaluation, and integral caching. Validate agreement with CutTools, Ninja, COLLIER, and NinjaRS, achieve the agreed per-process runtime requirement, and publish a reproducible Typst benchmark report.

At implementation start:

- Write this plan to `/shared/external/madloop_tirust/references/PLAN.md` and register the goal above with `create_goal`.
- Run all work from `/shared/external/madloop_tirust/madgraph7_madloop_tirust`. Keep reference repositories unchanged; the requested plan file is the sole exception.
- Use branch `madloop_tirust` and repository-local identity `huashengshao <erdissshaw@gmail.com>`.
- Freeze starting commit `9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f` as the performance baseline. Keep baseline builds and generated artifacts inside ignored working directories.
- Commit and push validated intermediate milestones to `origin/madloop_tirust`, setting its upstream on the first push.
- Continue parallel agent research and independent numerical, performance, and integration audits.

TIRust remains explicitly selected through option 9; installation does not change automatic backend defaults.

## Interface and installation

**Fortran integration**

- Add backend 9 throughout card parsing, availability checks, dispatch, exports, makefiles, profiling, and statistics. Expand the reduction-library array to accommodate nine IDs plus its zero sentinel.
- Implement a Fortran `ISO_C_BINDING` adapter feeding the existing tensor-contraction path.
- Generate the tensor permutation using the existing IREGI mapping machinery. Use TIRust's `Convention::MadLoop` and component weights; apply odd-rank signs and normalization exactly once.
- Supply MadLoop's on-shell-aware invariants, complex squared masses, and `MU_R**2`. Map TIRust's `[UV1, IR1, IR2, finite]` into MadLoop's Laurent coefficients after pole selection.
- Preserve legacy return codes. Identify TIRust DP with `(H,U)=(2,9)` and QP with `(3,3)`, adding dedicated counters and updating all consumers to distinguish the existing aliases.
- Propagate numerical failures into MadLoop's stability and rescue machinery. Never cache failed reductions as successful results.

**Rust bridge and precision**

- Add a small repository-owned bridge with opaque contexts and uniquely prefixed C exports. Build it as an isolated shared library because TIRust and NinjaRS currently use different dependency revisions.
- Pin supplied TIRust `854b79dcb7fffcb35e92b4a8a3dd443369b798c4` and oneloopmaster `bd703d09a78f7849a6b86a07096e6f222fc01193`, recording complete revisions, lockfiles, and patch hashes.
- Retain reusable plans and workspaces. Keep numerator construction and contraction in Fortran.
- Support DP and genuine QP. Reuse NinjaRS's lossless decimal transport for 113-bit Fortran inputs, evaluate with 256-bit MPFR, and contract with QP numerator coefficients. DD must not be advertised as QP.
- Preserve MadLoop's independent stability tests; TIRust's success status alone is insufficient.

**`install tirust`**

- Follow the existing transactional NinjaRS installer pattern, including discovery, configuration persistence, help, and completion.
- Support pinned downloads and local `--source` / `--oneloop-source` overrides, plus installation prefix, job count, and update controls.
- Copy sources into isolated staging before patching. Build and smoke-test before atomic publication; failures preserve the previous installation and configuration.
- Package libraries, Fortran interface sources/modules, ABI metadata, and runtime dependencies. Support relocated internal-dependency exports.
- Initially qualify GNU Fortran on Linux using the pinned development environment. Test coexistence with all five backends without repinning NinjaRS.

## Pole modes, caching, and Fortran optimization

**Public controls**

- Extend existing `LoopPoleMode` to backend 9, preserving its finite-term-retaining semantics.
- Add `TIRustComputeFinite`, default `.TRUE.`, and `UseQPIntegrandForTIRust`, default `.TRUE.`.
- Add `TIRustGlobalCache`: `-1` enables all supported topologies, `0` disables caching, positive values limit propagator count.
- Add `TIRustCacheMaxBytes`, default 268435456 per worker. Reaching the budget stops insertion without changing numerical results.

**Accelerated poles-only evaluation**

- Maintain a minimal pinned TIRust patch series inside MadGraph and apply it only to installation copies.
- Generate scalar pole kernels through oneloopmaster's existing expression/code-generation API, requesting only Laurent tags `-1` and `-2`.
- Propagate a partial-evaluation mode through tensor recurrences, gathers, stability checks, and precision rescues. Skip finite arithmetic while preserving double-pole contributions to simple poles and separate scaleless UV/IR residues.
- Initially compute all three pole slots, then apply UV/IR selection. Reject 'finite disabled and no poles selected.'
- Expose partial results through standalone pole diagnostics and the benchmark runner. Mark finite output as unevaluated; ordinary amplitude and Monte Carlo entrypoints reject partial requests.
- Qualify undotted SM integrals through six propagators and rank six. Unsupported requests fail explicitly; no silent full-evaluation fallback may be counted as accelerated execution.

**Cache lifecycle and optimization**

- Register plans before evaluation, reuse numerical integrals across helicities and split orders, and clear event-dependent entries at MadLoop's existing cache boundary.
- Separate full/poles-only, precision, and stability-check stores. Stability rechecks must not reuse primary-pass tensor results.
- Profile before modifying Fortran. Evaluate early cache lookup, reusable kinematic preprocessing, linear prefix sums, generated coefficient counts, and contiguous contraction/cache layouts.
- Retain optimizations only with numerical validation and measured benefit. Preserve existing floating-point safeguards and report gains separately from earlier NinjaRS optimizations.

## Validation and performance acceptance

Generate actual processes through the MG CLI and compare explicit backends **1, 6, 7, 8, and 9**.

| Final particles | `loop_sm`, QCD | `loop_qcd_qed_sm_Gmu`, EW |
|---|---|---|
| 1 | `u u~ > z [virt=QCD]` | `u d~ > w+ [virt=QED]` |
| 2 | `g g > t t~ [virt=QCD]` | `e+ e- > mu+ mu- [virt=QED]` |
| 3 | `u u~ > z g g [virt=QCD]` | `e+ e- > t t~ a [virt=QED]` |
| 4 | `u u~ > z z g g [virt=QCD]` | `e+ e- > mu+ mu- a a [virt=QED]` |

Pin the downloaded EW model. Add mixed QCD/QED `u u~ > g a` and loop-induced `g g > h`, `g g > h h` controls.

- Use 100 saved deterministic physical points per main process. On-shell 2->1 samples vary boosts; higher multiplicities use 1 TeV. Add separate threshold, soft/collinear, small-Gram, and complex-mass stress fixtures.
- Compare Born, finite, single-pole, and double-pole results using MadLoop's existing `3e-6` regression criterion, targeting `1e-8` on ordinary points. Investigate failures using independent reducers and higher precision; retain every failure.
- Test tensor mapping through rank six, normalization, scale dependence, scaleless poles, cache invalidation, subprocess isolation, and failure/rescue behavior.
- Verify poles-only coefficients against full evaluation and instrument that finite kernels/arithmetic are skipped.
- Test installation rollback, missing/incompatible libraries, relocated exports, mixed-backend initialization, QP transport, and historical statistics compatibility.
- Freeze identical models, cards, compiler flags, point sets, helicity/color sums, pole settings, and rescue policy. Explicitly override profiling defaults that otherwise disable COLLIER poles.
- Measure warmed full-amplitude runtime using single-core, alternating paired repetitions: at least seven pairs, approximately one second per timed batch. Report initialization separately.
- **Acceptance:** the upper 95% paired confidence bound for candidate/baseline runtime must be <=1.05 for every main process. Compare the TIRust-first rescue chain against the starting revision's materialized default chain, and check existing-backend regressions separately.
- Report standalone timings for all five reducers, native-DP results separately from rescued results, and actual backend/precision usage. Label FULL-SKIP only as an estimated reduction time.

Unresolved accuracy failures or runtime regressions remain open acceptance items.

## Report and delivery milestones

Produce `/shared/external/madloop_tirust/madgraph7_madloop_tirust/docs/benchmarkfolder/benchmark.pdf`, its Typst source, numerical summaries, and reproduction commands.

Match the typography, margins, navy headings, table design, panels, and green-white-red timing palette from the [preserved pyAmpliCol Typst source](https://github.com/MadGraphTeam/pyamplicol/blob/7266c1600f6ec4e39be39986e6503a515c380c52/docs/performance_reports/loop/benchmarks.typ). Pin and attribute that revision because the supplied `loop` branch URL no longer resolves. Preserve the existing NinjaRS report.

Commit and push these milestones:

1. Recorded plan, baseline, environment, and frozen validation inputs.
2. Installer, ABI, Fortran integration, and DP/QP qualification.
3. Pole selection, accelerated poles-only mode, and cache validation.
4. Measured Fortran optimizations and complete five-backend campaign.
5. Independent audit fixes, reproducible report, and final documentation.

The report must distinguish measured results, failures, and unqualified cases. Mark the assigned goal complete only when the agreed implementation, validation, performance, and delivery criteria are satisfied.
