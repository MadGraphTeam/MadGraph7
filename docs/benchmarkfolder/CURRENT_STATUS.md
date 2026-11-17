# Current status — 2026-11-17

**Current campaign:** [`tirust-2b9ba961-1`](campaigns/tirust-2b9ba961-1/). TIRust
`main` `2b9ba961f640a6f0ee3b7f376cd23c5b869f41c1`: `a5d4afb` plus one pin-only commit
(Symbolica community `d0424f9e`, oneloopmaster `07fc754f`, no TIRust source change).
The MadGraph source is `35e3408e7a`, in which the bridge turns TIRust's method choice off.
It replaces [`tirust-a5d4afbf-1`](campaigns/tirust-a5d4afbf-1/) (TIRust `a5d4afb`, method
choice on) and the rehearsal [`tirust-9a781998-1`](campaigns/tirust-9a781998-1/). Their
summaries stay in their directories.

**Why it was rerun.** In `tirust-a5d4afbf-1`, TIRust accepted an inaccurate value from
its method choice (CMS `forward-0.001`). The bridge now sends lanes that its binary64
gates refuse to its 106/256/512-bit tiers instead (`35e3408e7a`, a patch from the TIRust
session). That fixes `forward-0.001`, but this campaign finds **a new blocking defect**:
at the EW t t̄ γ threshold, TIRust accepts a double-precision value that is 1.1% off
(below). PR #437 is in review; this defect must be fixed before it merges.

**Return codes.** MadLoop uses return-code format 2 since the `madloop_ninjars`
merge (`c667b8d569`). TIRust reports `U=3`, which it shares with IREGI: a stable
DP result is `213` and a QP rescue is `323`. The rehearsal ran under format 1
(`219`). The benchmark tools decode both.

| | |
|---|---|
| MadGraph exporter | `35e3408e7a`: the merged `madloop_ninjars`, the bridge's 512-bit genuine-QP retry (`0eb109bbb0`), its QP check-pass cache (`78a03130ec`) and method choice off (`35e3408e7a`) |
| TIRust runtime | fresh `install tirust` build, DSO `6d1271b574b3…`, poles-only patch `3dae2b93…`, bridge `lib.rs` `bc41688b5699…` |
| Toolchain | gfortran 15.2.0, Rust 1.97.1, Python 3.13.12 (`tools/tirust_env.sh` outside `nix develop`), `FFLAGS=-w -fPIC -O3` |
| Baseline | `9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f` (starting-revision CutTools) |

## Results (shared host)

| Check | Result |
|---|---|
| Installation | `smoke --require-poles` passes. All 27 DP and 54 QP lanes of the pinned method-choice fixtures pass. In the gate-passed fixtures, one synthetic rank-5 lane (LI-32) is 7e-6 off in DP, as it was with the previous library. That is within upstream's 1e-5 contract (see "Findings for TIRust upstream") |
| [Ordinary matrix](campaigns/tirust-2b9ba961-1/summaries/numerical-summary.md) | **66/66 lanes pass `3e-6`; 54 also meet `1e-8`**, with the same misses as before. In Zgg and ZZgg every candidate reducer misses at the same points. TIRust needs no QP rescue in any process: EW 2→4 is now 100/0 (previously 99/1; runtime `854b79dc`: 87/13 Zgg, 14/86 ZZgg, 40/60 EW 2→4). Of the 66 lanes only TIRust's changed: its largest EW t t̄ γ residual is now 6.8e-10 (previously 1.1e-9). Candidate CutTools differs from the baseline by up to 6.5e-10, because the candidate source includes `madloop_ninjars` and MadGraph `main` |
| [Double-double gate](campaigns/tirust-2b9ba961-1/summaries/dd-gate-summary.md) | All 1,289,214 recorded bridge calls (11 processes) replay bit for bit. All 23,123 DD-tier calls agree with 256-bit evaluation within 1.7e-16, including the 1,121 lanes admitted only by the looser 2^-80 gate (within 1.1e-17). **Qualified.** There are 3.6× more DD calls than in `tirust-a5d4afbf-1`, because the lanes that the method choice served now go to the tiers |
| [Per-point provenance](campaigns/tirust-2b9ba961-1/summaries/provenance-summary.md) | 1,100 points classified as binary64-only, DD, MPFR, genuine QP or failed. More points use the tiers: Zgg 66/34/0 (previously 87/13/0), ZZgg 0/83/17 (14/71/15), EW 2→4 12/87/1 with no genuine QP (40/58/1 and 1 QP), EW t t̄ γ 97/3/0 (100/0/0). Each status equals its numerical lane. Cache-off and a 1 MiB budget reproduce every bit |
| [Stress](campaigns/tirust-2b9ba961-1/summaries/stress-summary.md) | 38 points (17 real-mass, 21 complex-mass). TIRust accepts 28 and refuses 10. CMS `forward-0.001` now agrees with the QP reference. **One accepted TIRust value is wrong (blocking):** at the EW t t̄ γ threshold, TIRust returns 213 with a claimed accuracy of 6.5e-9, but its finite part is 1.1% below the QP values of TIRust, NinjaRS and CutTools (3.3e-2 in MadLoop's accuracy measure). On `a5d4afb` this point was a correct QP rescue (323). The 9\|6\|7\|1 chain returns the same value. **At EW μ⁺μ⁻ and CMS `forward-1e-06` small Gram the chain returns a wrong value:** CutTools QP (329) has exactly half of the universal double pole. At CMS `forward-1e-08` CutTools hangs. Every other accepted TIRust value agrees with the reference. Seven refusals are at points where every reducer fails: soft/collinear, the Zgg/ZZgg thresholds and the mixed small-Gram point |
| [Stability](campaigns/tirust-2b9ba961-1/stability/) | All five reducers have 100/100 usable estimates in every process, in raw DP and after rescue. No TIRust raw-DP estimate exceeds 1e-6. Fractions above 1e-9: Zgg TIRust 0.02 (previously 0.03; Ninja 0.00, NinjaRS 0.01); ZZgg TIRust 0.04 (previously 0.09; Ninja 0.20, NinjaRS 0.15, COLLIER 0.03). Other reducers are unchanged |
| Cache diagnostic | `tools/tirust_benchmark_cache.py --campaign tirust-2b9ba961-1` on `qcd_n1_uux_z` passes 61/61 records, including the QP check-pass cache (cold miss, then hit) |
| [Report](benchmark.pdf) | 48 preserved pages plus 6 current-source pages for this campaign |
| Cost of method choice off | The TIRust session measured, on paired replays, +5.5% TIRust time per point on the stock `check profile` Zggg trace and +22% with the default card. Not acceptance |
| Provisional profile | Not rerun on the final pins. The [rehearsal profile](campaigns/tirust-9a781998-1/summaries/profile-zggg-o3.json) (DSO `6c11cfd9`) measured stock O3 Zggg FULL at TIRust 1128 ms against NinjaRS 962 ms per point (ratio 1.177, previously 1.558). Not acceptance |
| [g g → g g g g profile](campaigns/tirust-9a781998-1/summaries/profile-gg4g-o3.md) | Rehearsal runtime. Stock O3 `check profile`: the five reducers are within 7% per helicity-summed point (about 30 s), because coefficient construction takes 91–97%. For one helicity, the TIRust loop evaluation (1.29 s) is 2.4× CutTools, 7× Ninja and 11× NinjaRS, and 0.71× COLLIER. Not acceptance |
| [g g → z z z z](campaigns/tirust-9a781998-1/summaries/ggzzzz-tirust.md) | Rehearsal runtime with the 512-bit QP retry. Check of pyAmpliCol's TIRust refusals, which are by design there. In MadLoop, TIRust returns 213 at all 60 ordinary points with no refusal; the double-double tier is needed at 2 of them, where TIRust agrees with COLLIER within 8.6e-8. **No accepted TIRust value disagrees with a reference.** At the small-Gram stress point, the DP value accepted at 1.1e-4 agrees with forced QP within 1.7e-5. The threshold point is a QP rescue at 1.5e-8, which no other reducer can check. The 9\|6\|7\|1 chain equals TIRust everywhere |

**Stress classification.** A reference is forced-QP CutTools and Ninja when they agree.
Otherwise it takes two agreeing stability-checked values from independent reducers. Every
value's IR double pole is checked, and so are values whose error exceeds `MLStabThres`
while their claimed accuracy does not. Since 2026-11-17, a value with no reference is also
WRONG when even the nearest independent value is farther than `MLStabThres` or ten times
the independent values' largest difference. This rule finds the t t̄ γ threshold defect,
where Ninja's wrong single pole leaves no reference. It changes no classification of
`tirust-a5d4afbf-1`.

## Findings outside TIRust

- On the ZZgg soft-collinear stress point, CutTools stops the process with a
  Fortran `STOP` (CT692), and exit code 0, as soon as a lane falls back to it.
- At the most forward complex-mass small-Gram point (`forward-1e-08`), CutTools never returns.
  TIRust and the chain handle the Z-resonance and WW-threshold points, agreeing with the QP references
  within 1e-14.
- Either way, a TIRust-first chain ending in CutTools can abort or hang a run.
  The stress tools resume after such points.
- **`OSThres` projection changes the integral.** At CMS `forward-1e-06` and the EW
  μ⁺μ⁻ small-Gram point, MadLoop's on-shell projection (`BUILD_KINEMATIC_MATRIX`)
  sets a small but nonzero invariant exactly to zero. CutTools QP (329), in the
  baseline too, then returns exactly half of the double pole that every regular
  point has (−2.403e-3 against −4.807e-3 × Born), and the 9\|6\|7\|1 chain returns
  that value after TIRust's refusal. With `OSThres = 1e-14`, TIRust and Ninja QP
  agree to 10 digits at `forward-1e-06`. The fix belongs in MadLoop's projection.
- **Other reducers' wrong accepted values.** CutTools DP accepts a finite part of
  2.4e18 (true 4.0e-5) at the loop-induced H H small-Gram point. COLLIER, Ninja and
  NinjaRS accept a wrong single pole at the QCD t t̄ threshold, and Ninja QP one 112×
  too large at the EW t t̄ γ threshold. TIRust is right at the first two points.
- **Helicity filters differ between baseline and candidate.** Since
  `madloop_ninjars` (`9a9b28d62e`), MadLoop learns and applies helicity filters
  with `LoopPoleMode = 0`, the campaign setting; baseline `9228fc58` bypasses
  them there. The candidate therefore skips helicities that the baseline
  computes, which matters for the local paired timing (see its handoff).

## Findings for TIRust upstream

- **The precision tiers hide inaccurate binary64 inputs (open, blocking merge).** At the
  EW t t̄ γ threshold stress point (√s = 2 m_t (1 + 1e-7), with a photon carrying 1e-7 of
  √s), the top quarks' kinetic energies are 2.1e-12 and 5.7e-13 GeV. The binary64 spacing
  of their energy is 2.8e-14 GeV, so the double-precision kinematics fix the point only to
  about 1%.
  - The bridge's tiers evaluate these rounded inputs accurately. Both loop directions
    therefore agree, and MadLoop accepts the value.
  - MadLoop's DP rotations (`NRotations_DP`) are exact axis permutations, which leave the
    rounding unchanged, so they would not catch it either.
  - Boosted and rotated binary64 copies of the same point give TIRust DP values 0.3% to
    1.4% off forced QP, with claimed accuracies of 5e-11 to 6.5e-9. Forced QP agrees across
    them to 2e-9.
  - 25 lanes use the 512-bit tier here. No ordinary point of the campaign needs it.
  - With method choice on, DP reduction noise made the point fail MadLoop's check, and
    MadLoop rebuilt it in QP, where `ImprovePSPoint` restores the kinematics.

  The reproducer, with bridge traces of the exact DP and QP inputs, has been sent to
  the TIRust session. The fix belongs to the bridge. It needs Hua-Sheng's approval, then
  a requalification.
- **Method-choice lanes with a false accuracy claim (fixed in the bridge,
  `35e3408e7a`).** TIRust's method-choice expansions stop near 1e-9 per tensor, and
  cancellations amplify that. The error is the same in both loop directions, so MadLoop's
  direction check cannot see it.
  - At CMS `forward-0.001` on `a5d4afb`, TIRust returned 213 2.0e-6 off with a claimed
    accuracy of 3.1e-15. The TIRust session's forward-angle scan found up to 2.8e-5 at
    θ = 5e-4.
  - With method choice off, both now agree with QP.
  - Upstream documents this caveat for embedders in TIRust PR huashengshao/TIRust#42
    (documentation only, no pin change).
- **Synthetic rank-5 fixture lane LI-32.** In the upstream gate-passed fixtures this
  lane is 7e-6 off in DP, the same with both libraries and within the 1e-5 contract.
  - It is a Direct PV lane, so its error is amplified rounding. MadLoop's direction check
    probably sees only part of it. That differs from the method choice, whose error is
    identical in both directions.
  - Lanes with rank R > N come only from effective vertices (HEFT ggH and other
    higher-dimension operators), not from renormalizable processes.
  - If a HEFT-type process is ever qualified, the bridge could tighten
    `EmbedderOptions::gate` for R > N lanes. That would be a measured follow-up and
    Hua-Sheng's call.
- **Exactly singular Gram matrices.** TIRust refuses (420) the invariants that
  MadLoop's `OSThres` projection sets to exact zero (CMS `forward-1e-06/1e-08`,
  EW μ⁺μ⁻ small Gram). Refusing is correct: those inputs describe a different
  integral (see "Findings outside TIRust").
- **Genuine-QP refusals at small Gram (fixed in the bridge, `0eb109bbb0`).** On
  the rehearsal runtime, the genuine-QP interface refused all 48 rank-4 box
  tensors at the loop-induced H H small-Gram point, and unchecked forced QP
  returned an incomplete sum with the QP failure flag set. The bridge now
  retries such lanes once at 512 bits. The H H point agrees with the reference
  in this campaign.
- **Uncached QP check pass (fixed in the bridge, `78a03130ec`).** The g g → z z z z
  threshold point took 13 h, almost all in MadLoop's QP check pass, which bypassed
  the bridge's QP cache. The check pass now has its own QP cache.

## Open items

1. **Tiers on inaccurate binary64 inputs (blocking merge).** The TIRust session proposes
   a bridge fix and Hua-Sheng approves it. Then rerun `tools/tirust_requalify.sh` under a
   new campaign ID.
2. **Precise timing acceptance.** This is Hua-Sheng's local TODO:
   [`LOCAL_TIMING_TODO.md`](campaigns/tirust-2b9ba961-1/timing-handoff-1/LOCAL_TIMING_TODO.md).
   Time the newest campaign, because the fix for item 1 may change TIRust's cost.
   Every main process needs an upper 95% paired candidate/baseline bound of at
   most 1.05. Read the TIRust effect against the candidate's explicit-backend
   controls too, because of the helicity-filter difference above.
3. **Shared-host profile on the final pin** (optional, provisional). Rerun the stock O3
   Zggg profile when the host is quiet.
4. **`OSThres` projection** (MadLoop, outside this PR). Do not zero physically
   nonzero invariants; until then a TIRust-first chain can return CutTools' wrong
   value at such points.

The earlier campaigns, reports and evidence are preserved in git history and in
the tag `madloop_tirust-evidence-2026-11-13`.
