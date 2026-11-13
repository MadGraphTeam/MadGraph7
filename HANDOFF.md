# TIRust–MadLoop handoff

**Status: NOT COMPLETE.** The backend and its supporting features are implemented,
but the agreed current-source numerical, stress, precision-cohort and performance
acceptance criteria have not all been satisfied. Do not mark the integration goal
complete or make the draft PR ready on the basis of the existing report alone.

Audit date: **2026-11-13**. Audited implementation commit:
`b71c59086d409ff031544ca15dcaef077a28b38b`. The commit adding this handoff is a
documentation-only successor; it does not qualify or change the implementation.

## Working agreement and repositories

- Always work from `/shared/external/madloop_tirust/madgraph7_madloop_tirust`.
- Branch: `madloop_tirust`, tracking `origin/madloop_tirust`.
- Repository-local identity: `huashengshao <erdissshaw@gmail.com>`.
- Remote: `https://github.com/MadGraphTeam/MadGraph7.git`.
- Commit and push validated milestones. Do not reset unrelated work or overwrite
  immutable evidence. Keep builds and generated working outputs in ignored
  `.tirust-build/` directories.
- Original performance baseline: `9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f`.
- Full user plan: `/shared/external/madloop_tirust/references/PLAN.md`;
  repository plan: [docs/benchmarkfolder/PLAN.md](docs/benchmarkfolder/PLAN.md).
- Preserve reference checkouts. The user separately authorized upstream TIRust
  OEM/performance work; use an isolated checkout for further upstream changes.
  Do not modify COLLIER or fix its known cache bug.
- Final precise timings are Hua-Sheng's **local-machine TODO**. Shared-host work
  is best effort, without reserved CPUs. See the
  [timing amendment](docs/benchmarkfolder/PLAN-AMENDMENT-2026-11-10.md).

The existing assigned goal remains incomplete (the goal service last reported
`usageLimited`, not `complete`). This handoff does not waive any acceptance gate.

## Review state, verified during this audit

- [MadGraph draft PR #437](https://github.com/MadGraphTeam/MadGraph7/pull/437):
  open, head `madloop_tirust`, base `madloop_ninjars`; head matched audited commit.
- [Original NinjaRS PR #315](https://github.com/MadGraphTeam/MadGraph7/pull/315)
  targets `main`.
- [TIRust PR #35](https://github.com/huashengshao/TIRust/pull/35): open, non-draft,
  `hs/madloop-speed` into `main`; head is the current pinned `9a781998…`, base was
  `49dc836e4b3fb0785dba3646a8a9a9c9de19cc5f`. It is **not merged**.
- OEM activation was published on upstream main in `de60dc480fec…` and is
  retained in the new source. The OEM target is `iregi_rust`; oneloopmaster
  operations are covered within that core's scoped calls. Do not copy license
  keys or personal credentials into new handoff/evidence documents.

## Current source versus the installed runtime: resolve this first

| Item | Current implementation pin / identity |
|---|---|
| TIRust | `9a781998e54a39fb06d3621de3cd458da09d2a52` |
| oneloopmaster | `bd703d09a78f7849a6b86a07096e6f222fc01193` |
| Symbolica | `dfe6bd97b974b1028dc63071ea165136cd2b0ef5` |
| Poles-only patch SHA256 | `a310c5a943397e4bc244af56692322b30d9761055f6af74bec8e1c7945053cce` |
| Bridge `vendor/tirust/src/lib.rs` SHA256 | `e5bdb54a4c484c0e5390478d2e3f9dd8942b1d60746a669d77c7df7558378d37` |

Authoritative files: [installer](madgraph/various/tirust_install.py),
[Cargo.toml](vendor/tirust/Cargo.toml), [Cargo.lock](vendor/tirust/Cargo.lock),
[patch manifest](vendor/tirust/patches/manifest.json) and
[current bridge contract](vendor/tirust/README.md).

**The normal local CLI currently selects an older runtime.** At audit time:

- `input/mg7_configuration.txt` key `tirust` points to
  `.tirust-build/tirust-native-tiers-final-1/HEPTools/tirust/lib`.
  Its metadata pins `10ce34ed8ad5b6f5f989a0e220eb3a244a71d1a1`, with DSO SHA256
  `d427667a0f2f55d0bd96778c1259ffcf52e3925341890b81c38f5f94e1eb7052`.
  This is the runtime used for the published 48-page report, **not current HEAD**.
- Default `HEPTools/tirust` is older again: pin `854b79dc…`, DSO `49af12f0…`.
  `install tirust` without `--update` can reuse a valid old ABI-1 installation.
- `../references/TIRust` is clean `main` at `de60dc480fec6475d2373a533c64d1c40ce23b2e`.
  It is not current pin `9a781998…`; blindly passing it as `--source` is wrong.
- Previously generated exports keep their old absolute library paths. Updating
  root configuration alone does not upgrade them: regenerate, or qualify an
  isolated relink explicitly.

Preserve the old runtime for historical reproduction. Fresh-install the new pin
in a new prefix, smoke-test it, and record source/patch/lock/native hashes before
calling any new results current-source evidence.

## What is implemented, and what is still unqualified

| Requirement | Completion audit |
|---|---|
| Backend 9; card parsing, availability, dispatch, exports, makefiles, statistics | Implemented; explicit selection, automatic defaults preserved. |
| Fortran ISO_C_BINDING adapter; IREGI permutation, signs, normalization, invariants | Implemented; historical rank-six/interface evidence exists. Latest-source qualification is still required. |
| DP and genuine binary128-input QP | Implemented: 113-bit input transport, 256-bit evaluation and optional QP numerator contraction. DD is not QP. |
| Transactional installer, source overrides, ABI metadata and relocated exports | Implemented; historical tests exist. Recheck the current pin/patch, fresh installation, relocation/NLO and coexistence. |
| UV/IR selection, finite control, accelerated poles-only diagnostics | Implemented. Current regenerated patch/lazy evaluators need current-pin pole/full and skipped-finite regression evidence. |
| Numerical caching, budgets, event/subprocess boundaries, independent checks | Implemented and recently changed materially. Full current-source cache/failure/budget/precision qualification remains open. |
| CLI process matrix | Historical 66 lanes pass `3e-6` (54 meet `1e-8`) at source `0fcd2cd873d6`. Those passes do not qualify `b71c59086d`. |
| Threshold, soft/collinear, small-Gram, complex-mass accuracy | Unresolved failures/refusals and independent-accuracy gaps remain. |
| Runtime acceptance | All eight precise local gates and separate existing-backend regressions remain open. Shared-host timings do not establish them. |
| Native-DP versus rescued timing cohorts | Incomplete; backend-9 cohort runner deliberately refuses until a per-call provenance adapter is qualified. |
| Appendix-C-style stability plots | Delivered for older sources. Refresh QCD/EW, multiple-multiplicity raw-DP/rescued comparisons for the final source; retain explicit sample/tail limits. |
| Typst benchmark PDF/source/data | Delivered and verified at 48 pages, but it describes older code/runtime. Append new source-bound results; preserve historical pages and NinjaRS report. |

### Three changes after the published O3 measurements

1. `b350286280`: exact-input whole-tensor cache for primary full-DP requests;
   retained accepted rescued-route subintegrals; separate shared check store;
   shared plans and `tier_margin_bits=0`; pin `d774a952…`.
2. `ab31cdf840`: pooled auxiliary executors and reduced retained layouts;
   pin `6e4bf348…`.
3. `b71c59086d`: pin `9a781998…`, lazy master evaluators and regenerated poles
   patch; `double_double_headroom_bits=26`. The DD conditioning gate changes
   from `2^-66` to `2^-80`, with a claimed rounding amplification limit `2^-26`.

These are numerical/cache-policy changes, not documentation-only updates.
The looser DD gate needs independent high-precision and physical-stress
qualification. Do not infer it is safe for all SM amplitudes from agreement on
one process, or relax regression criteria to make it pass.

Commit messages describe 16 bridge tests, upstream suites of 711/713 passes,
464,000 replayed calls, 150 MadLoop points and 53,200 DD-versus-256-bit tensors.
These are useful leads; corresponding new committed reproducible evidence was
not found during this audit. Recover and seal the original receipts, or rerun
before treating the claims as full current-source qualifications.

## Verified historical evidence and timing interpretation

[benchmark.pdf](docs/benchmarkfolder/benchmark.pdf) and
[its owner](docs/benchmarkfolder/build_stock_o3_profile_report.py) verified during
this audit: **48 pages**, PDF SHA256
`a7cb2798ed7b3de1028d5fe3d8cf4bedb419a1b2ef14e13e58e8fdc6f6868279`.
The last page binds MadGraph source `cda704e28d634e893fabc3ac358d9a27789f93aa`
and the old `d427…` runtime. Its actual serial `check profile` FULL values are:

| Backend | Full helicity-summed ms / PS point |
|---|---:|
| CutTools | 1579.82 |
| Ninja | 1504.55 |
| COLLIER | 1771.92 |
| NinjaRS | 1450.70 |
| TIRust | 2260.84 |

Each run used 50 timed points, generated MODEL/DHELAS/process/driver objects at
`-O3`, normal learned filters (48 contributing of 96 helicities), both UV/IR
poles, and `CTModeRun=1`. The separate stability phase is not timed. All ten
pairwise comparisons of the single retained common endpoint pass `1e-8`;
this does not certify every timed point. TIRust was 1.55845 times NinjaRS in
this unpaired shared-host capture. These numbers must **not** be relabeled as
measurements of the latest pin.

The `b71…` commit quotes **concurrent** “loop evaluation” times
124.6/74.8/333.7/58.8/248.2 ms. Without raw logs their exact scope is unverified.
In MadGraph, “loop evaluation” can label an unpolarized or single-helicity
**FULL−SKIP estimate**. It is not the full helicity-summed PS-point time.
Do not compare those figures with the FULL table above or count them as the
agreed paired performance acceptance. Extract `1000 * run_unpolarized_total`
and the raw “Helicity sum time / PSpoint” line from actual profile runs.

For optimistic profiling keep normal `LoopPoleMode=-1`, `HelicityFilterLevel=2`
and learned loop filters. Explicit pole mode `0` bypasses learned filters and
has a different workload. `check profile` preserves COLLIER's default UV/IR
flags (both true); the separate `check timing` wrapper has different defaults.
Changing `FFLAGS` does not rebuild old objects: fresh/rebuilt MODEL, DHELAS,
process and driver objects are required. Installed reducer release builds must
be recorded separately.

## Recommended next steps

### 1. Freeze the next implementation and qualify a fresh installation

Inspect the tree before changing it; retain the current baseline and records.
Use GNU Fortran/Linux in the pinned environment. All commands below run from
the MadGraph repository. These are next-agent commands, **not actions performed
by this handoff audit**:

```bash
cd /shared/external/madloop_tirust/madgraph7_madloop_tirust
git status --short
git rev-parse HEAD
nix develop
source tools/tirust_env.sh
export FFLAGS='-w -fPIC -O3'
python3 -B bin/madgraph
```

At the MG prompt, choose a fresh unique prefix (example):

```text
install tirust --update --prefix=.tirust-build/handoff-9a781998/HEPTools --jobs=2
quit
```

Let the installer fetch the pinned source; do not substitute the stale reference
checkout. Verify actual metadata pin `9a781998…`, patch `a310c5…`, bridge source,
DSO/Fortran-module hashes, exports and DP/genuine-QP/poles/cache smoke including
`--require-poles`. Record runtime dependency resolution and coexistence with
1/6/7/8. Preserve previous installations and failures. Re-export processes.
Check the upstream PR status and final pin before another release/repin.

### 2. Register a new append-only campaign

The existing `tirust-native-tiers-1` registration and
[LOCAL_FINAL_VALIDATION_TODO.md](docs/benchmarkfolder/LOCAL_FINAL_VALIDATION_TODO.md)
select `77eb…`/`10ce…`/`d427…`; they are **templates/history for HEAD**, not an
already valid current-source campaign. The wrapper should reject incompatible
source/native identities. Do not edit their frozen manifests or import passes.

After installation qualification, use a fresh campaign ID, for example:

```bash
python3 -B tools/tirust_benchmark.py register-campaign \
  --campaign tirust-headroom26-1 \
  --tirust-prefix .tirust-build/handoff-9a781998/HEPTools/tirust \
  --predecessor entry-latch-cache-guard-1 \
  --inherit-physical-cards \
  --reason 'Current 9a781998 pin, repeated-DP cache/check store and DD headroom26'
```

Use the **measured physical-input predecessor** above. The pending
`tirust-native-tiers-1` has no physical observations to inherit; using an
unmeasured immediate predecessor previously failed registration. Inheriting
cards does not inherit numerical or timing results.

Then adapt the existing local TODO's supported
`tools/tirust_benchmark_local.py install/prepare/verify/forward` workflow to the
new campaign and fresh installation/measurement IDs. Follow with `init`, `model`,
`build`, `numerical`, `campaign`, `summarize` and `build_local_report.py` as shown
there. Keep exported O3 flags unchanged for both baseline and candidate; the
wrapper records/checks them. Use the
[comparison-backend recipe](docs/benchmarkfolder/qualification/comparison-backends/README.md)
without repinning NinjaRS. Avoid unscoped benchmark `init`, which can rewrite an
older root baseline record. Keep all frozen points, model pins and card ancestry.

### 3. Complete current-source correctness and stress qualification

Run all **11 processes × six lanes × 100 saved physical points**: baseline
CutTools and candidate 1/6/7/8/9. Compare Born, finite and both poles at `3e-6`,
retaining/investigating separate `1e-8` target misses.

| Final particles | QCD: `loop_sm` | EW: pinned `loop_qcd_qed_sm_Gmu` |
|---|---|---|
| 1 | `u u~ > z [virt=QCD]` | `u d~ > w+ [virt=QED]` |
| 2 | `g g > t t~ [virt=QCD]` | `e+ e- > mu+ mu- [virt=QED]` |
| 3 | `u u~ > z g g [virt=QCD]` | `e+ e- > t t~ a [virt=QED]` |
| 4 | `u u~ > z z g g [virt=QCD]` | `e+ e- > mu+ mu- a a [virt=QED]` |

Controls: mixed QCD/QED `u u~ > g a`, loop-induced `g g > h` and `g g > h h`.
Retain the extra Zggg timing process and its frozen inputs as an additional
control. Use the existing deterministic 2→1 boosts and higher-multiplicity
1 TeV fixtures; do not replace difficult points.

Investigate EW threshold, QCD soft/collinear/small-Gram and complex-mass failures
with independent reducers and genuine higher-precision inputs. Failed buffers,
self-consistency or same-engine higher precision alone are not independent
amplitude truth. Requalify the headroom26 gate over these domains; revise it if
needed. The older QP-mass/CMS/analytic-Fortran prototypes remain unpromoted unless
explicitly reviewed and qualified.

For the new cache policy test exact-key completeness, failure/nonfinite paths,
cache-off equivalence, event/subprocess/context isolation, primary/check
separation, budget exhaustion/insertion refusal and policy changes. Include
QCD/EW, loop-induced, genuine QP, finite/poles-only and supported rank-six cases.
Pooled executors must not leak precision, gate or kinematic state between leases.
Verify new lazy pole evaluators preserve scaleless UV/IR residues and skip finite
kernels/arithmetic; unsupported dotted/N>6/rank>6 requests must fail explicitly.

### 4. Finish provenance, stability plots and precise runtime acceptance

- Implement/qualify per-amplitude/per-tensor route plus cache provenance before
  native-DP versus rescued-cohort claims. `tools/tirust_benchmark_cohorts.py`
  currently deliberately refuses backend 9. Status `219` is DP transport, not
  native-only arithmetic; `323` identifies genuine QP transport, with numerator
  precision also controlled by `UseQPIntegrandForTIRust`. Batch route counters
  miss whole-tensor/Fortran-cache reuse and cannot classify individual points.
- Refresh final-source QCD/EW and multiple-multiplicity Appendix-C-style raw-DP
  cumulative stability plots and ratio-to-Ninja panels; show automatic rescues
  separately. Retain failures in the denominator, distinguish internal estimates
  from independent error, and state sample size/tail resolution and any remaining
  larger-sample TODO. See the
  [stability amendment](docs/benchmarkfolder/PLAN-AMENDMENT-STABILITY-2026-11-10.md).
- On Hua-Sheng's local machine, run at least seven alternating paired warmed
  repetitions for **each of the eight main processes**, using full amplitudes
  with natural stability/rescue, matched O3 flags, points/cards/poles/helicity
  and color sums. Compare the TIRust-first chain with the **materialized default
  chain of baseline 9228fc58…**; do not assume today's defaults are the baseline.
  The upper 95% paired candidate/baseline bound must be **≤1.05 in every case**.
  Measure existing-backend regressions separately, including if a main gate fails.
  Record initialization separately and actual batch lengths (at least about
  one second; a complete heavy stream can take longer). Native/rescued cohort
  diagnostics and standalone/profile timings do not replace these gates.

### 5. Publish final-source evidence and close review

Append, rather than relabel, numerical summaries, failures, actual backend and
precision usage, plots and measured optimizations to the Typst report. Preserve
its attributed pyAmpliCol style and the existing NinjaRS report. Refresh mutable
status/local instructions and PR descriptions to the final pin and native build.
The current status/README/local TODO/precision TODO are stale for `b71…`; this
handoff is the current-source warning, not a rewrite of their historical evidence.
Recover/seal or reproduce the latest optimization logs before citing them as
qualified. Finish independent review of both PRs. Mark the assigned goal complete
only after the remaining agreed implementation and acceptance requirements pass.

## Useful entry points and evidence

- User guide: [docs/tirust.md](docs/tirust.md).
- ABI/cache/gate contract: [vendor/tirust/README.md](vendor/tirust/README.md).
- Fortran adapter: [mtirust.f90](vendor/tirust/mtirust.f90) and
  [generated interface template](madgraph/iolibs/template_files/loop_optimized/TIRUST_interface.inc).
- Installer tests: `tests/unit_tests/various/test_tirust_install.py`;
  Fortran/transport tests: `tests/unit_tests/loop/test_tirust_fortran.py` and
  `test_tirust_transport_cache.py`; bridge tests: `vendor/tirust/src/lib.rs`.
- Historical ordinary outcomes:
  [QUALIFICATION_PROGRESS.md](docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/QUALIFICATION_PROGRESS.md).
- Preserved EW threshold refusal:
  [README](docs/benchmarkfolder/experiments/fortran-qp-mass-ew-threshold-qualification-1/README.md).
- Actual O3 profile packet:
  [index](docs/benchmarkfolder/experiments/zggg-o3-stock-profile-1/index.json);
  [report/reproduction](docs/benchmarkfolder/experiments/zggg-o3-stock-profile-report-1/rendered-1/README.md).
- Report verification (pure, no physics execution):

  ```bash
  python3 -B docs/benchmarkfolder/build_stock_o3_profile_report.py \
    --root docs/benchmarkfolder --verify-only
  ```

- Known COLLIER-cache bug note, user said **do not fix**:
  `/shared/external/madloop_ninjars/madgraph7_madloop_ninjars/.ninjars-build/upstream-poles-20261023/ninjars/COLLIER_CACHE_BUG.md`.

## Checks performed for this handoff

At implementation `b71…`, **67 focused installer/notices/statistics/Fortran-boundary
unit tests passed** with the pinned GNU environment, and the existing 48-page
report's pure verifier passed. Unit log on this host:
`.tirust-build/handoff-audit-20261113/unit-boundaries.log`.
Reproduce the focused boundary suite in the pinned environment:

```bash
python3 -B -m unittest \
  tests.unit_tests.various.test_tirust_install \
  tests.unit_tests.various.test_rust_dependency_notices \
  tests.unit_tests.various.test_ninjars_integration_boundaries \
  tests.unit_tests.various.test_ninjars_statistics \
  tests.unit_tests.loop.test_tirust_fortran \
  tests.unit_tests.loop.test_tirust_transport_cache
```

Those boundary tests do not build/qualify the current Rust dependency or run the
SM/stress campaign. No fresh installation, physics benchmark or performance
acceptance campaign was run by this audit. No production code or configuration
was changed; this handoff is documentation only.
