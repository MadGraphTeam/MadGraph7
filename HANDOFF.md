# TIRust–MadLoop handoff

**Status: NOT COMPLETE.** Backend 9 is implemented and requalified on the shared
host for TIRust `main` `a5d4afb` (campaign `tirust-a5d4afbf-1`). Two things are
open before PR #437 merges:

- **A TIRust accuracy defect.** Method-choice lanes can be accepted with a false
  accuracy claim (one stress point in this campaign). The TIRust session owns the
  fix; it needs a new pin and a requalification.
- **Precise local timing acceptance**, which is Hua-Sheng's local TODO.

Hua-Sheng took PR #437 out of draft and requested ValentinHirschi's review on
2026-11-16; the PR description lists both open items. Do not mark the
integration goal complete before they are done.

Updated 2026-11-16. The current results are in
[`docs/benchmarkfolder/CURRENT_STATUS.md`](docs/benchmarkfolder/CURRENT_STATUS.md).

## Working agreement

- **Repository.** Work in `/shared/external/madloop_tirust/madgraph7_madloop_tirust`
  on branch `madloop_tirust` (remote `MadGraphTeam/MadGraph7`), with local
  identity `huashengshao <erdissshaw@gmail.com>`.
- **Pushing.** Push with
  `git -c credential.helper= -c credential.helper='!gh auth git-credential' push origin madloop_tirust`.
  There are no stored HTTPS credentials; `gh` is logged in.
- **Ownership** (Hua-Sheng, 2026-11-13).
  - The "TIRust to MadLoop interface" session owns the MadGraph pin
    (`vendor/tirust`, `UPSTREAM_REVISION`) and every push to `madloop_tirust`.
  - The "Speed up TIRust in MadLoop and its initialization" session owns TIRust
    upstream (PR #35 and later) and decides which revision to pin. It sends
    MadGraph changes as patches.
- **Evidence is lean.** Commit compact summaries plus exact commands. Raw lane
  data stays in the ignored `.tirust-build/` and `.benchmark-work/`.
- **Toolchain.** Source `tools/tirust_env.sh` **outside** `nix develop` (gfortran
  15.2, Rust 1.97.1, Python 3.13). The flake currently resolves newer tools.
  Set `FFLAGS='-w -fPIC -O3'`.
- **Locks.** Only NinjaRS (unlicensed Symbolica) needs
  `.tirust-build/numerical-tests.lock`. TIRust is OEM-licensed.
- **Do not touch** COLLIER or its cache bug, or the reference checkouts.

## What exists

- **Backend 9:** card parsing, dispatch, exports, statistics, a Fortran
  `ISO_C_BINDING` adapter, DP and genuine binary128-input QP, UV/IR selection,
  poles-only diagnostics, caches, and the transactional `install tirust`.
  Details are in [`docs/tirust.md`](docs/tirust.md) and
  [`vendor/tirust/README.md`](vendor/tirust/README.md).
- **Campaign tooling** (`tools/`):
  - `tirust_benchmark*.py` runs the frozen 66-lane protocol and the local
    timing wrapper.
  - `tirust_capture/` captures and replays bridge calls; it is the
    double-double gate study.
  - `tirust_stress.py` handles physical and CMS stress (`tirust_cms_stress.py`).
    It resumes after a reducer STOPs or hangs. `tirust_stress_rerun.py` reruns
    a campaign's recorded stress lanes against another bridge library, for
    bridge-only changes.
  - `tirust_stability.py` produces the Appendix-C plots.
  - `tirust_provenance.py` classifies every point by arithmetic route and checks
    cache-off and budget equivalence.
  - `tirust_benchmark_cohorts.py --backend 9` times cohorts per call.
  - `tirust_current_report.py` builds the report pages.
  - `tirust_requalify.sh` runs everything for a new pin.
  - `tirust_pin/` holds the pin-bump procedure.

## Return codes (format 2)

The `madloop_ninjars` merge (2026-11-13) brought MadLoop return-code format 2,
which retires PJFry and gives `U=2` to NinjaRS. TIRust now reports `U=3`, shared
with IREGI, in both precisions, with `T` for precision: stable DP is `213` and a
QP rescue is `323` (format 1 used `219` and `323`).
- The MadEvent and aMC@NLO `TIRust DP/QP` counters tell TIRust from IREGI
  through `ML_GET_RET_CODE_LIBRARY()` in `MadLoopCommons.inc`.
- `tools/tirust_benchmark_worker.py` (`BACKEND_CODES`) and the benchmark
  drivers' warm-status checks decode both formats.
- The rehearsal campaign `tirust-9a781998-1` predates the merge, so its
  summaries show `219`; the final campaign shows `213`.

## Results for `a5d4afb` (campaign `tirust-a5d4afbf-1`)

The final pin reproduces the rehearsal on `9a781998` and confirms the two bridge
fixes made since. Details and links are in `CURRENT_STATUS.md`.

- **Ordinary matrix:** 66/66 lanes pass `3e-6`; 54 also meet `1e-8`, with the
  same misses as before. TIRust needs no genuine-QP rescue in QCD Zgg and ZZgg
  and one in EW 2→4; its internal DD/MPFR tiers cover the rest.
- **Double-double gate (headroom 26):** 1,289,214 captured calls replay bit for
  bit. All 6,419 DD-tier calls agree with 256-bit evaluation within 1.9e-20,
  including the 492 lanes in the new window. Qualified.
- **Caches:** cache-off and a 1 MiB budget reproduce every bit for all 1,100
  ordinary points. The cache diagnostic passes 61/61, including the QP
  check-pass cache.
- **Stress (38 points), corrected 2026-11-16:** TIRust accepts 28 and refuses
  10. Four lane outcomes block:
  - **TIRust, inaccurate:** at CMS `forward-0.001` TIRust returns 213 with a
    claimed accuracy of 3.1e-15 but is 2.0e-6 from the QP reference (4.1e-6 in
    MadLoop's measure, above `MLStabThres = 1e-6`), alone and as the chain's first
    backend. Method-choice expansions stop near 1e-9 per tensor, cancellations
    amplify that, and MadLoop's direction check cannot see it.
  - **Chain, wrong fallback value:** at EW μ⁺μ⁻ and CMS `forward-1e-06` small
    Gram, TIRust refuses correctly, and the chain returns CutTools QP (329) with
    exactly half of the universal double pole. MadLoop's `OSThres` projection
    zeroes a small invariant there, which changes the integral.
  - Every other accepted TIRust value agrees with the reference. Seven refusals
    are at points where every reducer fails. At CMS `forward-1e-08`, CutTools
    hangs.
  - The first summary missed these: its tool let CutTools serve as its own
    reference and compared only at `3e-6`. `tools/tirust_stress.py` now needs two
    independent reducers for a fallback reference, checks every value's IR
    double pole and flags false accuracy claims above `MLStabThres`.
  - **CutTools STOP:** a CutTools fallback can `STOP` the process (QCD ZZgg
    soft-collinear point).
  - **512-bit QP retry confirmed:** the H H small-Gram point now matches the
    reference in forced QP, and TIRust accepts the EW t t̄ γ threshold point. The
    checked reducers disagree there, but TIRust agrees with NinjaRS QP within
    1.1e-9.
- **Stability:** identical to the rehearsal; every reducer has 100/100 usable
  estimates, and TIRust's raw-DP availability is 100/100 for Zgg and ZZgg (the
  earlier pilot had 24 and 0).
- **Helicity filters:** since `madloop_ninjars`, MadLoop learns and applies
  helicity filters with `LoopPoleMode = 0`, which baseline `9228fc58` bypasses.
  This matters for the local paired timing (see its handoff).
- **Provisional profile:** not rerun on this pin (host overloaded); the
  rehearsal measured TIRust/NinjaRS 1.177 on stock O3 Zggg.
- **Report:** `docs/benchmarkfolder/benchmark.pdf` appends 6 current-source
  pages to the preserved 48 pages. Build it with
  `python3 -B tools/tirust_current_report.py render --campaign tirust-a5d4afbf-1 --publish`
  and verify it with `… verify`.
- **Rehearsal:** `tirust-9a781998-1` keeps the g g → g g g g and g g → z z z z
  checks, the QP 512-bit retry rerun and the
  [recovered optimization logs](docs/benchmarkfolder/campaigns/tirust-9a781998-1/recovered-optimization-evidence.md).

## Remaining work

1. **TIRust method-choice accuracy (blocking merge).** The TIRust session is to
   fix it upstream (accept a method-choice lane in binary64 only when its
   expansion converged to binary64 precision), after measuring the cost on
   Zggg. Then follow item 4 for the new pin; if that comes before the local
   timing, time the new campaign.
2. **Hua-Sheng's local timing.** Follow
   [`LOCAL_TIMING_TODO.md`](docs/benchmarkfolder/campaigns/tirust-a5d4afbf-1/timing-handoff-1/LOCAL_TIMING_TODO.md).
   Gate: the upper 95% paired candidate/baseline bound is ≤ 1.05 for every main
   process, plus existing-backend regressions.
3. **Shared-host profile** (optional): rerun the stock O3 Zggg profile on this
   pin when the host is quiet.
4. **A later pin**, if the upstream session proposes one:
   - follow [`tools/tirust_pin/README.md`](tools/tirust_pin/README.md): export
     the commit, hand-merge the poles-only patch if needed, then run
     `regen_poles_patch.sh` and `pinbump.py`; move `vendor/tirust/pole-kernels`
     and the bridge's Symbolica `[patch.crates-io]` to TIRust's revisions;
   - install, build and test the bridge;
   - commit, then run
     `REASON='…' tools/tirust_requalify.sh tirust-<pin8>-1` (several hours on
     the shared host);
   - copy `report/current-source.typ` and `timing-handoff-1/` from
     `tirust-a5d4afbf-1`, render and publish the report, and update
     `CURRENT_STATUS.md`, this file and the PR #437 description.
5. **Outside TIRust.**
   - MadLoop's `OSThres` projection should not zero physically nonzero
     invariants. TIRust's refusal of such inputs is correct; CutTools returns a
     wrong value there.
   - A CutTools STOP or hang is a MadLoop/CutTools limitation; it can abort a
     TIRust-first chain that falls back to CutTools.
6. **Review.** PR #437 (stacked on NinjaRS PR #315) is in review with
   ValentinHirschi.

Earlier campaigns, the audit-style evidence capsules and the older report
builders are preserved in git history and in the tag
`madloop_tirust-evidence-2026-11-13`.
