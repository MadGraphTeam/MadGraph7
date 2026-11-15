# Current status — 2026-11-15

**Current campaign:** [`tirust-9a781998-1`](campaigns/tirust-9a781998-1/). This
is a rehearsal on TIRust `9a781998e54a39fb06d3621de3cd458da09d2a52`, the head of
TIRust PR #35 before its merge.

**Final pin (2026-11-14).** TIRust PR #35 is merged (`a2cfa4b`). MadGraph now pins
TIRust `main` `a5d4afbf87a0afc155dc57d2ac633af77a932a16` (`32a4a52242`), and the
bridge keeps a separate QP cache for MadLoop's check pass (`78a03130ec`). A fresh
install passes `smoke --require-poles`, and all 27 DP and 54 QP C-ABI fixture lanes
are bit-identical to the `9a781998` runtime. The final requalification
`tirust-a5d4afbf-1` (`tools/tirust_requalify.sh`) is running; its results will
replace the rehearsal figures below.

**Return codes.** Since the `madloop_ninjars` merge (2026-11-13), MadLoop uses
return-code format 2. TIRust reports `U=3`, which it shares with IREGI, so a
stable DP result is `213` and a QP rescue is `323`. This campaign ran before the
merge, under format 1, so its summaries show TIRust DP as `219`. The benchmark
tools decode both formats. The merge also changes MadLoop templates; the
final-pin campaign requalifies the merged source.

| | |
|---|---|
| MadGraph exporter | `624ef69f05` (generator inventory `be13df83…`); later commits change only `tools/`, `tests/` and `docs/` |
| TIRust runtime | fresh `install tirust` build, DSO `6c11cfd9eecf…`, poles-only patch `a310c5…`. The bridge has since gained the genuine-QP 512-bit retry (`lib.rs` `15be50e0…`); only the stress lanes were rerun with it |
| Toolchain | gfortran 15.2.0, Rust 1.97.1, Python 3.13.12 (`tools/tirust_env.sh` outside `nix develop`), `FFLAGS=-w -fPIC -O3` |
| Baseline | `9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f` (starting-revision CutTools) |

## Results (shared host)

| Check | Result |
|---|---|
| Installation | `smoke --require-poles` passes; pinned C-ABI fixtures pass 27 DP and 54 QP lanes |
| [Ordinary matrix](campaigns/tirust-9a781998-1/summaries/numerical-summary.md) | **66/66 lanes pass `3e-6`; 54 also meet `1e-8`.** The misses match the previous campaign; in Zgg and ZZgg every candidate reducer misses at the same points. TIRust needs no QP rescue in QCD Zgg (100/0) or ZZgg (100/0), and 99/1 in EW 2→4 (previously 87/13, 14/86 and 40/60). |
| [Double-double gate](campaigns/tirust-9a781998-1/summaries/dd-gate-summary.md) | Every recorded bridge call (11 processes, 482,376 calls) replays bit for bit. All 6,131 DD-tier calls agree with 256-bit evaluation within 1.9e-20, including the 482 lanes admitted only by the looser 2^-80 gate. **Qualified.** |
| [Per-point provenance](campaigns/tirust-9a781998-1/summaries/provenance-summary.md) | 1,100 points classified as binary64-only, DD, MPFR, genuine QP or failed. Each status equals its numerical lane. Cache-off and a 1 MiB budget reproduce every bit. |
| [Stress](campaigns/tirust-9a781998-1/summaries/stress-summary.md) ([QP 512-bit retry rerun](campaigns/tirust-9a781998-1/summaries/qp-rescue-512.md)) | 38 points (17 real-mass, 21 complex-mass). **No accepted TIRust value disagrees with a reference.** TIRust accepts 27 and refuses 11. Seven refusals are at points where every reducer fails: soft/collinear, the Zgg/ZZgg thresholds and the mixed small-Gram point. At the EW t t̄ γ threshold, Ninja rescues, but the reference reducers disagree with each other. At EW μ⁺μ⁻ and CMS forward-1e-6 small Gram, CutTools QP rescues through the 9\|6\|7\|1 chain. At CMS forward-1e-8, CutTools hangs. |
| [Stability](campaigns/tirust-9a781998-1/stability/) | All five reducers have 100/100 usable estimates in every process, raw DP and after rescue; the earlier pilot had TIRust at 24/100 for Zgg and 0/100 for ZZgg. No TIRust raw-DP estimate exceeds 1e-6. Fractions above 1e-9: Zgg TIRust 0.03 (Ninja 0.00, NinjaRS 0.01); ZZgg TIRust 0.09 (Ninja 0.20, NinjaRS 0.15, COLLIER 0.03). |
| [Provisional profile](campaigns/tirust-9a781998-1/summaries/profile-zggg-o3.json) | Stock O3 Zggg FULL: TIRust 1128 ms against NinjaRS 962 ms per point (ratio 1.177, previously 1.558). TIRust initializes fastest (8 s). Not acceptance. |
| [g g → g g g g profile](campaigns/tirust-9a781998-1/summaries/profile-gg4g-o3.md) | Stock O3 `check profile`: the five reducers are within 7% per helicity-summed point (about 30 s), because coefficient construction takes 91–97%. For one helicity, the TIRust loop evaluation (1.29 s) is 2.4× CutTools, 7× Ninja and 11× NinjaRS, and 0.71× COLLIER. Not acceptance. |
| [g g → z z z z](campaigns/tirust-9a781998-1/summaries/ggzzzz-tirust.md) | Check of pyAmpliCol's TIRust refusals, which are by design there. In MadLoop, TIRust returns 213 at all 60 ordinary points with no refusal; the double-double tier is needed at 2 of them, where TIRust agrees with COLLIER within 8.6e-8. **No accepted TIRust value disagrees with a reference.** At the small-Gram stress point, the DP value accepted at 1.1e-4 agrees with forced QP within 1.7e-5. The threshold point is a QP rescue at 1.5e-8, which no other reducer can check. The 9\|6\|7\|1 chain equals TIRust everywhere. |

## Findings outside TIRust

- On the ZZgg soft-collinear stress point, CutTools stops the process with a
  Fortran `STOP` (CT692), and exit code 0, as soon as a lane falls back to it.
- At the most forward complex-mass small-Gram point (`forward-1e-08`), CutTools never returns.
  TIRust and the chain handle the Z-resonance and WW-threshold points, agreeing with the QP references
  within 1e-14.
- Either way, a TIRust-first chain ending in CutTools can abort or hang a run.
  The stress tools resume after such points.

## Findings for TIRust upstream

- **Exactly singular Gram matrices.** TIRust refuses (420) complex-mass
  invariants that MadLoop's `OSThres` projection sets to exact zero (CMS
  `forward-1e-06/1e-08`, EW μ⁺μ⁻ small Gram). Elsewhere the chain rescues;
  at `forward-1e-08` CutTools then hangs.
- **Genuine-QP refusals at small Gram (fixed in the bridge after this
  campaign).** At the loop-induced H H small-Gram point, the genuine-QP
  interface (113-bit inputs, 256-bit MPFR) refused all 48 rank-4 box tensors.
  Unchecked forced-QP mode then returned an incomplete sum, about 11× too small,
  with the QP failure latch set, which automatic mode rejects. The bridge now
  evaluates such a lane once more at 512 bits. A rerun of this campaign's stress
  lanes on the patched bridge
  ([`qp-rescue-512.md`](campaigns/tirust-9a781998-1/summaries/qp-rescue-512.md))
  gives the H H point identical to NinjaRS QP. It also lets TIRust accept the
  EW t t̄ γ threshold point, where the reference reducers disagree with each
  other. Every other stress result is unchanged.
- **Uncached QP check pass (fixed in the bridge, `78a03130ec`).** The g g → z z z z
  threshold point took 13 h, almost all in MadLoop's QP check pass, which bypassed
  the bridge's QP cache. The check pass now has its own QP cache.

## Open items

1. **Final-pin requalification.** `tirust-a5d4afbf-1` on TIRust `main` `a5d4afb`
   is running. When it completes, its summaries, report pages and the timing
   handoff replace the rehearsal ones.
2. **Precise timing acceptance.** This is Hua-Sheng's local TODO:
   [`LOCAL_TIMING_TODO.md`](campaigns/tirust-9a781998-1/timing-handoff-1/LOCAL_TIMING_TODO.md).
   Every main process needs an upper 95% paired candidate/baseline bound of at
   most 1.05.
3. **Exactly singular Gram matrices.** TIRust refuses complex-mass invariants
   that MadLoop's `OSThres` projection sets to exact zero. The chain rescues
   them; an upstream improvement is optional.

The earlier campaigns, reports and evidence are preserved in git history and in
the tag `madloop_tirust-evidence-2026-11-13`.
