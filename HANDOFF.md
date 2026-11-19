# TIRust–MadLoop handoff

**Status: in review.** Backend 9 is implemented and requalified on the shared
host for TIRust `main` `1c130a81` (huashengshao/TIRust#43), with the bridge's
method choice off and its shifted DP check pass (campaign `tirust-1c130a81-1`).
No accepted TIRust value is wrong in the stress set any more: the EW t t̄ γ
threshold defect of `tirust-2b9ba961-1` is fixed. Only the review of PR #437 is
open. A precise local timing is no longer a condition for merging (Hua-Sheng,
2026-11-19); the cost figures below are the shared-host measurements.

Hua-Sheng took PR #437 out of draft and requested ValentinHirschi's review on
2026-11-16. Olivier Mattelaer and Theo Heimel were added as reviewers on
2026-11-17. NinjaRS PR #315 merged into `main` on 2026-11-16, so #437 now
targets `main`.

Updated 2026-11-19. The benchmark workspace `docs/benchmarkfolder` (report,
campaigns, `CURRENT_STATUS.md`) is no longer in the branch. It is a release
bundle, restored with `python3 -B tools/tirust_benchmark_bundle.py restore`; see
[`docs/tirust-benchmarks.md`](docs/tirust-benchmarks.md). Paths below under
`docs/benchmarkfolder/` refer to the restored workspace.

## Working agreement

- **Repository.** Work in `/shared/external/madloop_tirust/madgraph7_madloop_tirust`
  on branch `madloop_tirust` (remote `MadGraphTeam/MadGraph7`), with local
  identity `huashengshao`.
- **Pushing.** Push with
  `git -c credential.helper= -c credential.helper='!gh auth git-credential' push origin madloop_tirust`.
  There are no stored HTTPS credentials; `gh` is logged in.
- **Ownership** (Hua-Sheng, 2026-11-13).
  - The "TIRust to MadLoop interface" session owns the MadGraph pin
    (`vendor/tirust`, `UPSTREAM_REVISION`) and every push to `madloop_tirust`.
  - The "Speed up TIRust in MadLoop and its initialization" session owns TIRust
    upstream (PR #35 and later) and decides which revision to pin. It sends
    MadGraph changes as patches.
- **Evidence stays out of the branch.** Campaigns are written into the ignored
  workspace `docs/benchmarkfolder` and published as a new release bundle
  version. Raw lane data stays in the ignored `.tirust-build/` and
  `.benchmark-work/`.
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

## Results for `1c130a81`, shifted check pass (campaign `tirust-1c130a81-1`)

Details and links are in `CURRENT_STATUS.md`. The previous campaigns
`tirust-2b9ba961-1` (TIRust `2b9ba961`, without the shifted check pass) and
`tirust-a5d4afbf-1` (TIRust `a5d4afb`, method choice on) keep their summaries.

- **Pin and bridge:** TIRust `main` `1c130a81` (PR #43: the MPFR tiers round at
  their working precision) in `ee8fe28123`. The bridge change `af0bfe92cd`
  (handover 0005) multiplies every invariant of MadLoop's DP check pass by
  1 ± 2^-52 and keeps exact zeros and on-shell masses. MadLoop's own check then
  measures how much the result depends on input rounding. The install passes
  `smoke --require-poles` and the bridge tests (22 default, 24 poles-only). The
  upstream fixtures are bit for bit those of `2b9ba96`.
- **Ordinary matrix:** 66/66 lanes pass `3e-6`; 54 also meet `1e-8`. Every
  status equals `tirust-2b9ba961-1`, and TIRust needs no genuine-QP rescue in any
  process. One comparison changed: ZZgg point 45 moved from 1.8e-9 to 1.2e-8
  (Ninja 3.0e-8) and now misses `1e-8`. MadLoop now returns the midpoint of two
  independently rounded DP evaluations.
- **Double-double gate (headroom 26):** 1,289,214 captured calls replay bit for
  bit. All 23,123 DD-tier calls agree with 256-bit evaluation within 1.7e-16,
  including the 1,121 lanes in the new window. Qualified.
- **Caches:** cache-off and a 1 MiB budget reproduce every bit for all 1,100
  ordinary points. The cache diagnostic passes 61/61, including the QP
  check-pass cache.
- **Stress (38 points):** TIRust accepts 28 and refuses 10. **No accepted TIRust
  value is wrong.** Two lane outcomes were flagged, both outside TIRust:
  - **t t̄ γ threshold (fixed):** `tirust-2b9ba961-1` accepted a DP value 1.1%
    off with a claimed accuracy of 6.5e-9. The binary64 kinematics fix that point
    only to about 1%. The two DP evaluations now disagree, and MadLoop rescues the
    point in QP (323), bit for bit TIRust's forced QP and 3.2e-9 from the nearest
    of five independent values. The 9|6|7|1 chain returns the same value.
  - **Chain, wrong fallback value:** at EW μ⁺μ⁻ and CMS `forward-1e-06` small
    Gram, TIRust refuses correctly, and the chain returns CutTools QP (329) with
    half of the universal double pole. Current `main` (`fabaad5172`) with its
    default chain returns the same bits; with the stock `MLStabThres = 1e-3` it
    returns COLLIER's DP value with the double pole 3.4% low, and so does the
    9|6|7|1 chain (MadGraphTeam/MadGraph7#494). Lowering `OSThres` to 1e-14
    changes neither value, so the `OSThres` projection is not the cause.
  - Every other accepted TIRust value agrees with the reference. Seven refusals
    are at points where every reducer fails. At CMS `forward-1e-08`, CutTools
    hangs.
  - `tools/tirust_stress.py` needs two independent reducers for a fallback
    reference, checks every value's IR double pole and flags false accuracy
    claims above `MLStabThres`. A value with no reference is also WRONG when it
    lies far outside the spread of all independent values. The t t̄ γ threshold
    has no reference, because Ninja's single pole is 112× too large.
  - **CutTools STOP:** a CutTools fallback can `STOP` the process (QCD ZZgg
    soft-collinear point).
- **Stability:** every reducer has 100/100 usable estimates. TIRust's fractions
  above 1e-9 are 0.00 for Zgg (previously 0.02) and 0.04 for ZZgg.
- **Cost:** the TIRust session measured +5.5% (stock `check profile`) to +22%
  (default card) TIRust time per Zggg point with method choice off. The shifted
  check pass adds no evaluation: 4.33 s → 4.40 s on the Zggg default-card trace
  replay, within the run-to-run spread.
- **Helicity filters:** since `madloop_ninjars`, MadLoop learns and applies
  helicity filters with `LoopPoleMode = 0`, which baseline `9228fc58` bypasses.
  A paired timing against that baseline must separate the two (see the
  campaign's `timing-handoff-1/`).
- **Provisional profile:** not rerun on the final pins; the rehearsal measured
  TIRust/NinjaRS 1.177 on stock O3 Zggg.
- **Report:** the workspace's `benchmark.pdf` appends 6 current-source
  pages to the preserved 48 pages. Build it with
  `python3 -B tools/tirust_current_report.py render --campaign tirust-1c130a81-1 --publish`
  and verify it with `… verify`.
- **Rehearsal:** `tirust-9a781998-1` keeps the g g → g g g g and g g → z z z z
  checks, the QP 512-bit retry rerun and the
  recovered optimization logs
  (`campaigns/tirust-9a781998-1/recovered-optimization-evidence.md` in the workspace).

## Remaining work

1. **Shared-host profile** (optional): rerun the stock O3 Zggg profile on the
   final pin when the host is quiet. A paired local timing
   (`campaigns/tirust-1c130a81-1/timing-handoff-1/LOCAL_TIMING_TODO.md` in the
   workspace) is optional too; it is not a merge condition.
2. **A later pin or bridge change**, if the upstream session proposes one:
   - follow [`tools/tirust_pin/README.md`](tools/tirust_pin/README.md): export
     the commit, hand-merge the poles-only patch if needed, then run
     `regen_poles_patch.sh` and `pinbump.py`; move `vendor/tirust/pole-kernels`
     and the bridge's Symbolica `[patch.crates-io]` to TIRust's revisions;
   - install, build and test the bridge;
   - commit, then run
     `REASON='…' tools/tirust_requalify.sh tirust-<pin8>-1` (several hours on
     the shared host);
   - copy `report/current-source.typ`, `timing-handoff-1/` and `.gitignore` from
     `tirust-1c130a81-1`, render and publish the report, and update
     `CURRENT_STATUS.md`, `README.md`, `RELEASE.md` and
     `LOCAL_FINAL_VALIDATION_TODO.md` in the workspace, this file and the PR #437
     description;
   - build the new bundle version from the workspace. Add the campaign's files
     except `experiments/`, `report/benchmark-preview.pdf` and the
     `inputs/**/compilation-*.log` files, as v1 and v2 do:

     ```bash
     C=campaigns/tirust-<pin8>-1; B=docs/benchmarkfolder; args=()
     for e in .gitignore cases.json commands manifest.json points provenance stability \
              stress-points summaries timing-handoff-1 report/current-source.typ \
              report/current-source-supplement.pdf report/report-build.json report/report-data.json
     do args+=(--add "$C/$e"); done
     while IFS= read -r f; do args+=(--add "${f#$B/}"); done \
       < <(find "$B/$C/inputs" -type f ! -name '*.log' | sort)
     python3 -B tools/tirust_benchmark_bundle.py build --workspace "${args[@]}" --output .tirust-build/bundle
     ```
   - publish it, then repin `tools/tirust_benchmark_bundle.json` and
     `docs/tirust-benchmarks.md`.
3. **Outside TIRust.**
   - At forward small-Gram points, CutTools' QP value has half the IR double
     pole, and COLLIER's DP value passes the stock 1e-3 check with a double pole
     about 3% low. `main`'s default chain returns one of them; TIRust refuses
     these points (MadGraphTeam/MadGraph7#494).
   - A CutTools STOP or hang is a MadLoop/CutTools limitation, on `main` too; it
     can abort a TIRust-first chain that falls back to CutTools.
4. **Review.** PR #437 targets `main` and is in review with Valentin Hirschi,
   Olivier Mattelaer and Theo Heimel. Olivier removed the campaigns from the
   branch (`27d068d990`); the rest of the benchmark folder moved to the release
   bundle, as the NinjaRS PR did.

Earlier campaigns, the audit-style evidence capsules and the older report
builders are preserved in git history and in the tags
`madloop_tirust-evidence-2026-11-13` and `madloop_tirust-evidence-2026-11-17`
(the full workspace at `5298fd3066`).
