# Two P-wave bound states: TIRust fixtures (Phase 1)

One-loop QCD virtuals with two P-wave states, first target e⁺e⁻ → χ_c0 + h_c (`e+ e- > chic0(1|3P01) hc(1|1P11) [virt=QCD]`, model `loop_sm_onia-c_mass`, m_c = 1.55 GeV). The plan is in `references/PLAN.md` (two-P-wave plan, phases 0–5).

## State (Phase 1)

- The output generates, builds and runs (commit 417f251782).
- The Born of the loop output equals the tree `standalone_fortran` output (5.9411540544807900·10⁻¹⁰ at the test point).
- The virtual is reported unstable (return code 410): TIRust refuses part of the raised-power integrals (below).
- The quadruple-precision path stops with a message (Phase 3).

## Raised-power integrals

The second-order dual term (∂²/∂ε₁∂ε₂) needs these integrals:
- one line doubled (rank r+1, and the rank-r D₁₂ term);
- one line tripled (rank r+2);
- two lines doubled (rank r+2).

`fixtures.py` dumps every such call with the inputs MadLoop passes: powers, masses², invariants, shifts and μ², as hex doubles. It records whether TIRust's capability check accepts the call, its status, and the binary64 array when the call is served. It covers both passes (CTModeRun 1, and 2 with the reversed loop).

```
fixtures.py OUTDIR OUT.json.gz 10.58:2 20:2
```

TIRust 5025e12 + gate + relation tiers, √s = 10.58 and 20 GeV, two points each, both passes: 225 raised-power calls per point and pass, of which 53 are refused.

| Family | Lines, rank | Served | Refused |
|---|---|---|---|
| one line doubled | N = 2–5, rank 2–4 | 1104 | 56 with status 1 (N = 4 rank 3: 16; N = 5 rank 4: 40) |
| one line tripled | N = 2–4, rank 4 | 272 | 0 |
| two lines doubled | N = 3, 4, rank 4 | 0 | 368, by the capability check (`max_dotted_propagators = 1`) |

Every one of the 8 (point, pass) records has the same pattern (46 capability refusals and 7 status refusals each).

## Files (sent to TIRust on 2026-11-23)

- `data/fixtures_ee_chic0_hc.json.gz` (1.9 MB): e⁺e⁻ → χ_c0 h_c, every raised-power call, as above.
- `data/fixtures_gg_chic0_chic0_refused.json.gz` (23 kB): `g g > chic0(1|3P01) chic0(1|3P01) [virt=QCD]` at √s = 20 GeV, one point, both passes. Only the refused calls are kept, with `served_counts` per (n, rank, powers) for the others. That is 277 distinct inputs, from `fixtures.py OUTDIR OUT.json.gz 20:1` on the same TIRust.

| g g → χ_c0 χ_c0, per pass | Served | Refused |
|---|---|---|
| one line doubled | 1608 | 32 with status 1 |
| one line tripled | 519 | 1 with status 1 |
| two lines doubled (N = 3–5, rank 3–5) | 0 | 634, by the capability check |
