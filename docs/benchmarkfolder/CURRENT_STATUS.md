# Current status — 2026-11-15

**Current campaign:** [`tirust-a5d4afbf-1`](campaigns/tirust-a5d4afbf-1/), the
final pin. TIRust `main` `a5d4afbf87a0afc155dc57d2ac633af77a932a16`, after TIRust
PR #35 was merged as `a2cfa4b` (Symbolica community `0e2ad04c`, oneloopmaster
`2e7afabb`). It replaces the rehearsal
[`tirust-9a781998-1`](campaigns/tirust-9a781998-1/) on the unmerged PR head, whose
summaries stay in its directory.

**Pin after the 2026-11-16 Symbolica `community` rebase.** `0e2ad04c` is no
longer on any Symbolica branch. The bridge now pins TIRust `main`
`2b9ba961f640a6f0ee3b7f376cd23c5b869f41c1`: `a5d4afb` plus one pin-only commit
(Symbolica community `d0424f9e`, oneloopmaster `07fc754f`, no TIRust source
change). The campaign above was measured on `a5d4afb`; it was not rerun.

**Return codes.** MadLoop uses return-code format 2 since the `madloop_ninjars`
merge (`c667b8d569`). TIRust reports `U=3`, which it shares with IREGI: a stable
DP result is `213` and a QP rescue is `323`. This campaign runs under format 2;
the rehearsal ran under format 1 (`219`). The benchmark tools decode both.

| | |
|---|---|
| MadGraph exporter | `78a03130ec`: the merged `madloop_ninjars`, the bridge's 512-bit genuine-QP retry (`0eb109bbb0`) and its QP check-pass cache (`78a03130ec`). Later commits change only `tools/`, `tests/` and `docs/` |
| TIRust runtime | fresh `install tirust` build, DSO `8124ce130df3…`, poles-only patch `3dae2b93…`, bridge `lib.rs` `b8148e6508c1…` |
| Toolchain | gfortran 15.2.0, Rust 1.97.1, Python 3.13.12 (`tools/tirust_env.sh` outside `nix develop`), `FFLAGS=-w -fPIC -O3` |
| Baseline | `9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f` (starting-revision CutTools) |

## Results (shared host)

| Check | Result |
|---|---|
| Installation | `smoke --require-poles` passes. All 27 DP and 54 QP pinned C-ABI fixture lanes pass and are bit-identical to the `9a781998` runtime |
| [Ordinary matrix](campaigns/tirust-a5d4afbf-1/summaries/numerical-summary.md) | **66/66 lanes pass `3e-6`; 54 also meet `1e-8`**, with the same misses as the rehearsal. In Zgg and ZZgg every candidate reducer misses at the same points. TIRust needs no QP rescue in QCD Zgg (100/0) or ZZgg (100/0), and 99/1 in EW 2→4 (runtime `854b79dc`: 87/13, 14/86 and 40/60). Candidate CutTools now differs from the baseline by up to 6.5e-10 (rehearsal: 0), because the candidate source includes `madloop_ninjars` and MadGraph `main` |
| [Double-double gate](campaigns/tirust-a5d4afbf-1/summaries/dd-gate-summary.md) | Every recorded bridge call (11 processes, 1,289,214 calls) replays bit for bit. All 6,419 DD-tier calls agree with 256-bit evaluation within 1.9e-20, including the 492 lanes admitted only by the looser 2^-80 gate. **Qualified.** There are 2.7× more calls than in the rehearsal because MadLoop now learns helicity filters with `LoopPoleMode = 0`, evaluating each helicity separately on the first three points. The bridge's exact-input cache answers those repeats; later points make the same calls as before |
| [Per-point provenance](campaigns/tirust-a5d4afbf-1/summaries/provenance-summary.md) | 1,100 points classified as binary64-only, DD, MPFR, genuine QP or failed, identical to the rehearsal. Each status equals its numerical lane. Cache-off and a 1 MiB budget reproduce every bit |
| [Stress](campaigns/tirust-a5d4afbf-1/summaries/stress-summary.md) | 38 points (17 real-mass, 21 complex-mass). **No accepted TIRust value disagrees with a reference.** TIRust accepts 28 and refuses 10. Seven refusals are at points where every reducer fails: soft/collinear, the Zgg/ZZgg thresholds and the mixed small-Gram point. At EW μ⁺μ⁻ and CMS forward-1e-6 small Gram, CutTools QP rescues through the 9\|6\|7\|1 chain. At CMS forward-1e-8, CutTools hangs. At the EW t t̄ γ threshold TIRust accepts a QP rescue (323). The checked reducers disagree with each other there (Ninja's single pole is 112× too large), but TIRust agrees with NinjaRS QP within 1.1e-9 and with CutTools QP within 3.7e-6. Forced QP at the H H small-Gram point now agrees with the reference (8.5e-8) |
| [Stability](campaigns/tirust-a5d4afbf-1/stability/) | Table identical to the rehearsal. All five reducers have 100/100 usable estimates in every process, raw DP and after rescue; the earlier pilot had TIRust at 24/100 for Zgg and 0/100 for ZZgg. No TIRust raw-DP estimate exceeds 1e-6. Fractions above 1e-9: Zgg TIRust 0.03 (Ninja 0.00, NinjaRS 0.01); ZZgg TIRust 0.09 (Ninja 0.20, NinjaRS 0.15, COLLIER 0.03) |
| Cache diagnostic | `tools/tirust_benchmark_cache.py --campaign tirust-a5d4afbf-1` on `qcd_n1_uux_z` passes 61/61 records, including the QP check-pass cache (cold miss, then hit) |
| [Report](benchmark.pdf) | 48 preserved pages plus 6 current-source pages for this campaign |
| Provisional profile | Not rerun on this pin: the shared host was overloaded (load average about 10,700 on 384 cores) when the campaign finished. The [rehearsal profile](campaigns/tirust-9a781998-1/summaries/profile-zggg-o3.json) (DSO `6c11cfd9`) measured stock O3 Zggg FULL at TIRust 1128 ms against NinjaRS 962 ms per point (ratio 1.177, previously 1.558). Not acceptance |
| [g g → g g g g profile](campaigns/tirust-9a781998-1/summaries/profile-gg4g-o3.md) | Rehearsal runtime. Stock O3 `check profile`: the five reducers are within 7% per helicity-summed point (about 30 s), because coefficient construction takes 91–97%. For one helicity, the TIRust loop evaluation (1.29 s) is 2.4× CutTools, 7× Ninja and 11× NinjaRS, and 0.71× COLLIER. Not acceptance |
| [g g → z z z z](campaigns/tirust-9a781998-1/summaries/ggzzzz-tirust.md) | Rehearsal runtime with the 512-bit QP retry. Check of pyAmpliCol's TIRust refusals, which are by design there. In MadLoop, TIRust returns 213 at all 60 ordinary points with no refusal; the double-double tier is needed at 2 of them, where TIRust agrees with COLLIER within 8.6e-8. **No accepted TIRust value disagrees with a reference.** At the small-Gram stress point, the DP value accepted at 1.1e-4 agrees with forced QP within 1.7e-5. The threshold point is a QP rescue at 1.5e-8, which no other reducer can check. The 9\|6\|7\|1 chain equals TIRust everywhere |

## Findings outside TIRust

- On the ZZgg soft-collinear stress point, CutTools stops the process with a
  Fortran `STOP` (CT692), and exit code 0, as soon as a lane falls back to it.
- At the most forward complex-mass small-Gram point (`forward-1e-08`), CutTools never returns.
  TIRust and the chain handle the Z-resonance and WW-threshold points, agreeing with the QP references
  within 1e-14.
- Either way, a TIRust-first chain ending in CutTools can abort or hang a run.
  The stress tools resume after such points.
- **Helicity filters differ between baseline and candidate.** Since
  `madloop_ninjars` (`9a9b28d62e`), MadLoop learns and applies helicity filters
  with `LoopPoleMode = 0`, the campaign setting; baseline `9228fc58` bypasses
  them there. The candidate therefore skips helicities that the baseline
  computes, which matters for the local paired timing (see its handoff).

## Findings for TIRust upstream

- **Exactly singular Gram matrices.** TIRust refuses (420) complex-mass
  invariants that MadLoop's `OSThres` projection sets to exact zero (CMS
  `forward-1e-06/1e-08`, EW μ⁺μ⁻ small Gram). Elsewhere the chain rescues;
  at `forward-1e-08` CutTools then hangs.
- **Genuine-QP refusals at small Gram (fixed in the bridge, `0eb109bbb0`).** On
  the rehearsal runtime, the genuine-QP interface refused all 48 rank-4 box
  tensors at the loop-induced H H small-Gram point, and unchecked forced QP
  returned an incomplete sum with the QP failure flag set. The bridge now
  retries such lanes once at 512 bits. This campaign confirms the fix: the H H
  point agrees with the reference, and TIRust accepts the EW t t̄ γ threshold
  point.
- **Uncached QP check pass (fixed in the bridge, `78a03130ec`).** The g g → z z z z
  threshold point took 13 h, almost all in MadLoop's QP check pass, which bypassed
  the bridge's QP cache. The check pass now has its own QP cache.

## Open items

1. **Precise timing acceptance.** This is Hua-Sheng's local TODO:
   [`LOCAL_TIMING_TODO.md`](campaigns/tirust-a5d4afbf-1/timing-handoff-1/LOCAL_TIMING_TODO.md).
   Every main process needs an upper 95% paired candidate/baseline bound of at
   most 1.05. Read the TIRust effect against the candidate's explicit-backend
   controls too, because of the helicity-filter difference above.
2. **Shared-host profile on this pin** (optional, provisional). Rerun the stock O3
   Zggg profile when the host is quiet.
3. **Exactly singular Gram matrices.** TIRust refuses complex-mass invariants
   that MadLoop's `OSThres` projection sets to exact zero. The chain rescues
   them; an upstream improvement is optional.

The earlier campaigns, reports and evidence are preserved in git history and in
the tag `madloop_tirust-evidence-2026-11-13`.
