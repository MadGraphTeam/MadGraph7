# TIRust–MadLoop handoff

**Status: NOT COMPLETE.** Backend 9 is implemented and requalified on the shared
host for TIRust `9a781998`, the head of TIRust PR #35. Two things are still open:
1. requalifying a **merged TIRust `main`** pin, once PR #35 is merged;
2. **precise local timing acceptance**, which is Hua-Sheng's local TODO.

Do not mark the integration goal complete, or take PR #437 out of draft, before
both are done.

Updated 2026-11-13. The current results are in
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
    It resumes after a reducer STOPs or hangs.
  - `tirust_stability.py` produces the Appendix-C plots.
  - `tirust_provenance.py` classifies every point by arithmetic route and checks
    cache-off and budget equivalence.
  - `tirust_benchmark_cohorts.py --backend 9` times cohorts per call.
  - `tirust_current_report.py` builds the report pages.
  - `tirust_requalify.sh` runs everything for a new pin.
  - `tirust_pin/` holds the pin-bump procedure.

## Results for `9a781998` (campaign `tirust-9a781998-1`)

- **Ordinary matrix:** 66/66 lanes pass `3e-6`; 54 also meet `1e-8`, with the
  same misses as before. TIRust's genuine-QP rescues fell to 0 in QCD Zgg and
  ZZgg and to 1 in EW 2→4; its internal DD/MPFR tiers cover them.
- **Double-double gate (headroom 26):** 482,376 captured calls replay bit for
  bit. All 6,131 DD-tier calls agree with 256-bit evaluation within 1.9e-20,
  including the 482 lanes in the new window. Qualified.
- **Caches:** cache-off and a 1 MiB budget reproduce every bit for all 1,100
  ordinary points.
- **Stress (38 points):** no accepted TIRust value is wrong. TIRust accepts 27
  and refuses 11.
  - Seven refusals are at points where every reducer fails.
  - Three are small-Gram or threshold points that the 9|6|7|1 chain rescues.
  - One is CMS `forward-1e-08`, where CutTools then hangs.
  - **CutTools STOP:** a CutTools fallback can `STOP` the process (QCD ZZgg
    soft-collinear point).
  - **Genuine-QP refusals:** TIRust's genuine-QP interface refuses 48 rank-4
    box tensors at the loop-induced H H small-Gram point, where the DP-input
    path with 512-bit tiers succeeds. Unchecked forced QP then returns an
    incomplete sum; automatic mode rejects it through the QP failure latch.
    Reported upstream with a trace of the exact inputs.
- **Stability:** every reducer has 100/100 usable estimates. TIRust's raw-DP
  availability is now 100/100 for Zgg and ZZgg; the earlier pilot had 24 and 0.
- **Provisional stock O3 Zggg profile:** TIRust/NinjaRS 1.177, initialization
  8 s. Not acceptance.
- **Report:** `docs/benchmarkfolder/benchmark.pdf` appends 6 current-source
  pages to the preserved 48 pages. Build it with
  `python3 -B tools/tirust_current_report.py render --campaign tirust-9a781998-1 --publish`
  and verify it with `… verify`.
- **Recovered logs:** the logs behind the b350286, ab31cdf and b71c590 figures
  are indexed in
  [`recovered-optimization-evidence.md`](docs/benchmarkfolder/campaigns/tirust-9a781998-1/recovered-optimization-evidence.md).

## Remaining work

1. **Final pin.** When the upstream session reports that TIRust PR #35 is merged:
   - follow [`tools/tirust_pin/README.md`](tools/tirust_pin/README.md): export
     the merge commit, hand-merge the poles-only patch if needed, then run
     `regen_poles_patch.sh` and `pinbump.py`;
   - install, build and test the bridge;
   - commit, then run
     `REASON='…' tools/tirust_requalify.sh tirust-<pin8>-1` (it takes several
     hours on the shared host);
   - copy `report/current-source.typ` and `timing-handoff-1/` from
     `tirust-9a781998-1`, render and publish the report, and update
     `CURRENT_STATUS.md`, this file and the PR #437 description.
2. **Hua-Sheng's local timing.** Follow
   `docs/benchmarkfolder/campaigns/<final>/timing-handoff-1/LOCAL_TIMING_TODO.md`.
   Gate: the upper 95% paired candidate/baseline bound is ≤ 1.05 for every main
   process, plus existing-backend regressions.
3. **Optional upstream improvements.**
   - TIRust's handling of exactly singular Gram matrices from `OSThres`-zeroed
     CMS invariants.
   - A 512-bit tier for TIRust's genuine-QP interface, which refuses the H H
     small-Gram boxes that the DP-input path evaluates.
   - A CutTools STOP or hang is a MadLoop/CutTools limitation; it can abort a
     TIRust-first chain that falls back to CutTools.
4. **Review.** Finish the review of PR #437 (stacked on NinjaRS PR #315) after
   the final pin.

Earlier campaigns, the audit-style evidence capsules and the older report
builders are preserved in git history and in the tag
`madloop_tirust-evidence-2026-11-13`.
