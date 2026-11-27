# The estimate gap at the softest u u~ > eta_c g point

MadLoop's double-precision accuracy estimate at the softest point of the 2.5.5 comparison, `u u~ > etac(1|1S01) g / t t~ b b~ [virt=QCD]` at `PS.input` of `V0_uux_ccxg` (√s = 3.1015 GeV, a 1.5 MeV gluon), is 4.3e-12. The true error of the finite part is 2.2e-9 against forced quadruple precision. The error is the same with b, t at 10⁴ GeV (2.4e-9, estimate 1.6e-10). These runs use TIRust da7902b + bridge v16 (the pinned TIRust refuses the point), on 2026-11-27, with outputs generated at `25a823e463`.

The scripts patch a scratch output in place and run in this order:
1. `modes.py`
2. `split.py`
3. `qcmp.py`
4. `lcsub.py`
5. `lccmp.py`

The output path is set at the top of each script.

## Findings

1. **Not the input momenta** (`modes.py`).
   - Shifting each input component by one binary64 unit moves quadruple precision by at most 6e-13, so the amplitude is well conditioned in its inputs.
   - The same shifts move double precision by 7.8e-9.
   - CTModeRun 1 and 2 agree to 1.4e-12.
   - One rotation (`NRotations_DP = 1`) sees the error (estimate 5.1e-9; 2 rotations: 7.1e-9).
   - `ImprovePSPoint` makes no difference.
2. **Two loop groups** (`split.py`). The counterterm part agrees with quadruple precision to 1e-11. The error sits in loop groups 44 and 49, at 1.3e-9 and 1.5e-9 out of −0.443 each, identical in both loop directions.
   - Both groups are pentagons: u, ū, the gluon and the two constituents (each P/2), with masses (0, 0, 0, m_c, 0) and rank 3.
   - The pentagons are nearly degenerate: two legs have the same momentum, and the gluon is soft.
3. **Not TIRust** (`qcmp.py`).
   - At the 8 pentagon calls, TIRust's double-precision tensor integrals agree with its quadruple-precision ones, computed from the same inputs, to 1.7e-12 of the largest coefficient.
   - Substituting the quadruple-precision integrals leaves groups 44 and 49 unchanged.
   - (The comparison is meaningless for the box calls, where the quadruple-precision routine called this way does not return MadLoop's ordering. It is not used.)
4. **Not the contraction, but the loop coefficients** (`lcsub.py`):

   | Variant for groups 44 and 49 | Error vs QP |
   |---|---|
   | QP loop coefficients rounded to binary64, contracted in double precision with the double-precision integrals | 5e-13 (total 2.3e-11) |
   | Double-precision loop coefficients, contracted in quadruple precision | 1.3e-9 (unchanged) |

5. **Mechanism** (`lccmp.py`).
   - MadLoop builds the 35 loop coefficients of each pentagon in double precision. Their errors reach 5.4e-14 of the largest coefficient (13), about 250 units of binary64, accumulated in the loop-vertex updates.
   - The tensor integrals are large: up to 8e7 at rank 2 and 2.5e8 at rank 3.
   - The contraction Σ c_j I_j cancels by 5.3e4 (Σ|c_j I_j| / |Σ c_j I_j|).
   - The coefficient errors therefore shift each group by 3.0e-9 and 4.0e-9 relative, which is the observed error.
6. **Why the estimate misses it.** With a tensor-integral library, MadLoop's loop-direction test (CTModeRun 2) recomputes only the integrals (`SWITCH_ORDER`); both passes contract the same `LOOPCOEFS`. TIRust's check pass shifts only the invariants. The counterterm estimate of the bound-state outputs covers only the counterterm sum. A rotation rebuilds the coefficients from rotated momenta and sees the error.

## Consequences

- **Size.** The error is 2e-9 at a 1.5 MeV gluon, and about 1e-11 at the next points (√s = 3.11 GeV). It is far below `MLStabThres`, and below every target of the report.
- **Scope.** It is a property of MadLoop's TIR path (double-precision numerator coefficients times an ill-conditioned tensor basis), not of TIRust or of the heavy quarks.
- **A fix for the estimate.** MadLoop's estimate could cover it with a contraction term, like the counterterm-rounding term: c · ε · Σ|c_j I_j| / |Σ c_j I_j|, where c is the coefficient precision in units of ε. Here c ≈ 500 would give 3e-9. The alternative is one rotation (`NRotations_DP = 1`, about +50% per point). Neither is applied.
