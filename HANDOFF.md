# TIRust–MadLoop handoff

**Status: NOT COMPLETE.** Backend 9 is implemented and requalified on the shared
host for TIRust `main` `2b9ba961`, with the bridge's method choice off (campaign
`tirust-2b9ba961-1`). Two things are open before PR #437 merges:

- **A TIRust accuracy defect.** At the EW t t̄ γ threshold stress point, TIRust
  accepts a double-precision value that is 1.1% off. Its tiers evaluate binary64
  inputs that fix the point only to about 1%, and MadLoop's stability test cannot see
  this. The fix belongs to the bridge (TIRust session), then a requalification.
  Turning the method choice off fixed the previous defect (CMS `forward-0.001`) and
  exposed this one.
- **Precise local timing acceptance**, which is Hua-Sheng's local TODO.

Hua-Sheng took PR #437 out of draft and requested ValentinHirschi's review on
2026-11-16; the PR description lists both open items. Do not mark the
integration goal complete before they are done.

Updated 2026-11-17. The current results are in
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

## Results for `2b9ba961`, method choice off (campaign `tirust-2b9ba961-1`)

Details and links are in `CURRENT_STATUS.md`. The previous campaign
`tirust-a5d4afbf-1` (TIRust `a5d4afb`, method choice on) keeps its summaries.

- **Ordinary matrix:** 66/66 lanes pass `3e-6`; 54 also meet `1e-8`, with the
  same misses as before. TIRust needs no genuine-QP rescue in any process; its
  internal DD/MPFR tiers cover the lanes its binary64 gates refuse.
- **Double-double gate (headroom 26):** 1,289,214 captured calls replay bit for
  bit. All 23,123 DD-tier calls agree with 256-bit evaluation within 1.7e-16,
  including the 1,121 lanes in the new window. Qualified.
- **Caches:** cache-off and a 1 MiB budget reproduce every bit for all 1,100
  ordinary points. The cache diagnostic passes 61/61, including the QP
  check-pass cache.
- **Stress (38 points):** TIRust accepts 28 and refuses 10. Four lane outcomes
  block:
  - **TIRust, wrong:** at the EW t t̄ γ threshold TIRust returns 213 with a
    claimed accuracy of 6.5e-9, but its finite part is 1.1% below the QP values of
    TIRust, NinjaRS and CutTools. The top quarks' kinetic energies (2e-12 and
    6e-13 GeV) are close to the binary64 spacing of their energy (2.8e-14 GeV). The
    tiers evaluate these rounded inputs accurately, so both loop directions
    agree. Boosted and rotated copies of the point are 0.3% to 1.4% off. With
    method choice on, the point was a correct QP rescue. This happens alone and
    as the chain's first backend.
  - **Chain, wrong fallback value:** at EW μ⁺μ⁻ and CMS `forward-1e-06` small
    Gram, TIRust refuses correctly, and the chain returns CutTools QP (329) with
    exactly half of the universal double pole. MadLoop's `OSThres` projection
    zeroes a small invariant there, which changes the integral.
  - CMS `forward-0.001` (the method-choice defect) now agrees. Every other
    accepted TIRust value agrees with the reference. Seven refusals are at points
    where every reducer fails. At CMS `forward-1e-08`, CutTools hangs.
  - `tools/tirust_stress.py` needs two independent reducers for a fallback
    reference, checks every value's IR double pole and flags false accuracy
    claims above `MLStabThres`. Since 2026-11-17 it also marks a value WRONG when,
    with no reference, it lies far outside the spread of all independent values.
    The t t̄ γ threshold has no reference, because Ninja's single pole is 112× too
    large.
  - **CutTools STOP:** a CutTools fallback can `STOP` the process (QCD ZZgg
    soft-collinear point).
- **Stability:** every reducer has 100/100 usable estimates. TIRust's fractions
  above 1e-9 drop to 0.02 for Zgg and 0.04 for ZZgg (previously 0.03 and 0.09).
- **Cost:** the TIRust session measured +5.5% (stock `check profile`) to +22%
  (default card) TIRust time per Zggg point with method choice off.
- **Helicity filters:** since `madloop_ninjars`, MadLoop learns and applies
  helicity filters with `LoopPoleMode = 0`, which baseline `9228fc58` bypasses.
  This matters for the local paired timing (see its handoff).
- **Provisional profile:** not rerun on the final pins; the rehearsal measured
  TIRust/NinjaRS 1.177 on stock O3 Zggg.
- **Report:** `docs/benchmarkfolder/benchmark.pdf` appends 6 current-source
  pages to the preserved 48 pages. Build it with
  `python3 -B tools/tirust_current_report.py render --campaign tirust-2b9ba961-1 --publish`
  and verify it with `… verify`.
- **Rehearsal:** `tirust-9a781998-1` keeps the g g → g g g g and g g → z z z z
  checks, the QP 512-bit retry rerun and the
  [recovered optimization logs](docs/benchmarkfolder/campaigns/tirust-9a781998-1/recovered-optimization-evidence.md).

## Remaining work

1. **Tiers on inaccurate binary64 inputs (blocking merge).** The TIRust session
   has the reproducer: the t t̄ γ threshold point, boosted and rotated copies, and
   bridge traces with exact DP and QP inputs. They are in the ignored
   `.tirust-build/ttxa-threshold-2b9ba961/`. The session proposes a bridge fix and
   its cost on the ordinary matrix; Hua-Sheng approves it. Apply it as a patch,
   then follow item 4 from "install". If that comes before the local timing, time
   the new campaign.
2. **Hua-Sheng's local timing.** Follow
   [`LOCAL_TIMING_TODO.md`](docs/benchmarkfolder/campaigns/tirust-2b9ba961-1/timing-handoff-1/LOCAL_TIMING_TODO.md).
   Gate: the upper 95% paired candidate/baseline bound is ≤ 1.05 for every main
   process, plus existing-backend regressions.
3. **Shared-host profile** (optional): rerun the stock O3 Zggg profile on the
   final pin when the host is quiet.
4. **A later pin or bridge change**, if the upstream session proposes one:
   - follow [`tools/tirust_pin/README.md`](tools/tirust_pin/README.md): export
     the commit, hand-merge the poles-only patch if needed, then run
     `regen_poles_patch.sh` and `pinbump.py`; move `vendor/tirust/pole-kernels`
     and the bridge's Symbolica `[patch.crates-io]` to TIRust's revisions;
   - install, build and test the bridge;
   - commit, then run
     `REASON='…' tools/tirust_requalify.sh tirust-<pin8>-1` (several hours on
     the shared host);
   - copy `report/current-source.typ` and `timing-handoff-1/` from
     `tirust-2b9ba961-1`, render and publish the report, and update
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
