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
- **A fix for the estimate.** A contraction term like the counterterm-rounding term covers it. It is now applied with a calibrated constant (below, "The contraction-rounding term"). One rotation (`NRotations_DP = 1`) also covers it, at 1.4 to 2 times the time per point.

## A second gap: threshold bubbles of the ³S₁⁽⁸⁾ channels

In `g g`, `u g` and `u u~ > Jpsi(1|3S18) + jet` at √s = 203 GeV (`PS2.input`), MadLoop's estimate (2.7e-9, 3.8e-9, 1.2e-9) is 14 to 90 times below the true error (3.8e-8, 1.5e-7, 1.1e-7).

- `instr.py` and `probe.py` show the error sits in three loop groups, each off by the same 3.5e-7. All three are charm bubbles (masses m_c, m_c) on the gluon that carries the octet's momentum P.
- Substituting TIRust's quadruple-precision integrals does not remove the error, and neither does substituting the quadruple-precision coefficients.
- Shifting the inputs by one binary64 unit moves quadruple precision by 4e-12 but double precision by 3.8e-8.
- **Cause.** The bubble sits at p² = M² = 4 m_c², the threshold, where it behaves like √(p² − 4 m_c²). At 203 GeV, P² computed in double precision from the boosted momenta is 2.2e-13 (relative) below the threshold. A rounding of the components of order 100 GeV gives about 1000 units of P². The quadruple-precision pass rebuilds the momenta with P² exact to 1e-32.
- **Why the test misses it.**
  - The bubble is the same in both loop directions.
  - TIRust's check-pass shift (one unit) is far below the 1000-unit offset.
  - A rotation (a permutation of the axes) rounds P² the same way.
- **Possible fix.** Snap a loop invariant to (m_i + m_j)² when it lies within the rounding of the momentum components. This belongs to TIRust's decision on degenerate kinematics or to MadLoop's bound-state interface. Not done here.

## The contraction-rounding term

Bound-state outputs bound the accuracy from below by ONIA_TIRK ε max_pass Σ_loops Σ_j |c_j I_j| / |c₀|.

**Where it is computed:**
- `ONIA_TIRABS` is accumulated in `TIR_interface.f`, per double-precision pass.
- The ratio is taken in `loop_matrix.f` against the finite part of the same pass, both in the normalisation of the loop groups, before the bound-state factors (`ONIA_TIRREL`).
- The term enters next to the counterterm term (`ONIA_CTACC`).

A first version divided by the final, normalised finite part. That is off by a process-dependent factor (256 in g g → ³S₁⁽⁸⁾ g), which is why its first calibration with K = 10 inflated the S-wave estimates at 1 TeV by up to 800.

**Calibration.** `sets.py` generates the outputs with `ONIA_TIRK = 0`, and `calib2.py` records MadLoop's estimate without the term, ε·ONIA_TIRREL and the true error. The records are in `data/cal_<set>.json`. The sets:
- the ten 2.5.5 cases (b, t excluded);
- the three ³S₁⁽⁸⁾ channels;
- the 24 P-wave points of u u~ / g g → χ_c0 g at 1 TeV;
- 200 random g g → J/ψ g points, 60 g g → c c̄[¹S₀⁽⁸⁾] g points and 60 e⁺e⁻ → c c̄[³P₀⁽⁸⁾] g points at 1 TeV (setup A of the profiles).

That is 550 points of 18 processes.

**Results:**
- MadLoop's estimate is below the error at 13 S-wave points where the contraction is the cause. They need K = 1 to 236; the largest is the traced u u~ → η_c g point.
- With K = 300 all of them are covered.
- The S-wave estimates rise by a median factor 1 to 130 (largest single factor 7400, at an estimate of 2e-14), to at most 2e-8. No point reaches `MLStabThres`.
- Not covered, at any K:
  - the ³S₁⁽⁸⁾ threshold bubbles (they would need K ≈ 4e4);
  - the P-wave singlets at 1 TeV (K ≈ 5e4);
  - 19 of the 60 P-wave octet points, low by factors 1 to 47, whose errors are in the raised-power integrals.

**One rotation instead** (`rot.py`, `NRotations_DP = 1`, same points):
- It catches the contraction misses, all P-wave singlet misses and 17 of the 19 octet misses.
- It misses the threshold bubbles (worst factor 800).
- It costs 1.4 to 2.0 times the time per point.

It stays an option for users who need P-wave estimates at the 1e-8 level.
