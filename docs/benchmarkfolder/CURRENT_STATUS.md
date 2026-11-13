# Current status — 2026-11-13

**Current campaign:** [`tirust-9a781998-1`](campaigns/tirust-9a781998-1/). This
is a rehearsal on TIRust `9a781998e54a39fb06d3621de3cd458da09d2a52`, the head of
TIRust PR #35, which is not yet merged into TIRust `main`. The final MadGraph pin
must be a merged `main` commit. Once that exists, rerun the same pipeline with
`tools/tirust_requalify.sh tirust-<pin8>-1`.

| | |
|---|---|
| MadGraph exporter | `624ef69f05` (generator inventory `be13df83…`); later commits change only `tools/`, `tests/` and `docs/` |
| TIRust runtime | fresh `install tirust` build, DSO `6c11cfd9eecf…`, poles-only patch `a310c5…` |
| Toolchain | gfortran 15.2.0, Rust 1.97.1, Python 3.13.12 (`tools/tirust_env.sh` outside `nix develop`), `FFLAGS=-w -fPIC -O3` |
| Baseline | `9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f` (starting-revision CutTools) |

## Results (shared host)

| Check | Result |
|---|---|
| Installation | `smoke --require-poles` passes; pinned C-ABI fixtures pass 27 DP and 54 QP lanes |
| [Ordinary matrix](campaigns/tirust-9a781998-1/summaries/numerical-summary.md) | **66/66 lanes pass `3e-6`; 54 also meet `1e-8`.** The misses match the previous campaign; in Zgg and ZZgg every candidate reducer misses at the same points. TIRust needs no QP rescue in QCD Zgg (100/0) or ZZgg (100/0), and 99/1 in EW 2→4 (previously 87/13, 14/86 and 40/60). |
| [Double-double gate](campaigns/tirust-9a781998-1/summaries/dd-gate-summary.md) | Every recorded bridge call (11 processes, 482,376 calls) replays bit for bit. All 6,131 DD-tier calls agree with 256-bit evaluation within 1.9e-20, including the 482 lanes admitted only by the looser 2^-80 gate. **Qualified.** |
| [Per-point provenance](campaigns/tirust-9a781998-1/summaries/provenance-summary.md) | 1,100 points classified as binary64-only, DD, MPFR, genuine QP or failed. Each status equals its numerical lane. Cache-off and a 1 MiB budget reproduce every bit. |
| [Stress](campaigns/tirust-9a781998-1/summaries/stress-summary.md) | 38 points (17 real-mass, 21 complex-mass). **No accepted TIRust value disagrees with a reference.** TIRust accepts 27 and refuses 11. Seven refusals are at points where every reducer fails: soft/collinear, the Zgg/ZZgg thresholds and the mixed small-Gram point. At the EW t t̄ γ threshold, Ninja rescues, but the reference reducers disagree with each other. At EW μ⁺μ⁻ and CMS forward-1e-6 small Gram, CutTools QP rescues through the 9\|6\|7\|1 chain. At CMS forward-1e-8, CutTools hangs. |
| [Stability](campaigns/tirust-9a781998-1/stability/) | All five reducers have 100/100 usable estimates in every process, raw DP and after rescue; the earlier pilot had TIRust at 24/100 for Zgg and 0/100 for ZZgg. No TIRust raw-DP estimate exceeds 1e-6. Fractions above 1e-9: Zgg TIRust 0.03 (Ninja 0.00, NinjaRS 0.01); ZZgg TIRust 0.09 (Ninja 0.20, NinjaRS 0.15, COLLIER 0.03). |
| [Provisional profile](campaigns/tirust-9a781998-1/summaries/profile-zggg-o3.json) | Stock O3 Zggg FULL: TIRust 1128 ms against NinjaRS 962 ms per point (ratio 1.177, previously 1.558). TIRust initializes fastest (8 s). Not acceptance. |

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
- **Genuine-QP refusals at small Gram.** At the loop-induced H H small-Gram
  point, the genuine-QP interface (113-bit inputs, 256-bit MPFR) refuses all 48
  rank-4 box tensors (status 1). The DP-input path, with internal tiers up to
  512 bits, evaluates the same point correctly. Unchecked forced-QP mode
  (`CTModeRun=4`) then returns an incomplete sum, about 11× too small, with the
  QP failure latch set. Automatic mode rejects that result through the latch, so
  no wrong value is accepted. A trace with the exact QP inputs is kept at
  `.tirust-build/q-9a781998/qp-reproducer/li_gg_hh/` on the shared host.

## Open items

1. **Final pin.** Once TIRust PR #35 is merged, pin the `main` merge commit
   ([`tools/tirust_pin`](../../tools/tirust_pin/README.md)) and rerun
   `tools/tirust_requalify.sh`.
2. **Precise timing acceptance.** This is Hua-Sheng's local TODO:
   [`LOCAL_TIMING_TODO.md`](campaigns/tirust-9a781998-1/timing-handoff-1/LOCAL_TIMING_TODO.md).
   Every main process needs an upper 95% paired candidate/baseline bound of at
   most 1.05.
3. **Exactly singular Gram matrices.** TIRust refuses complex-mass invariants
   that MadLoop's `OSThres` projection sets to exact zero. The chain rescues
   them; an upstream improvement is optional.

The earlier campaigns, reports and evidence are preserved in git history and in
the tag `madloop_tirust-evidence-2026-11-13`.
