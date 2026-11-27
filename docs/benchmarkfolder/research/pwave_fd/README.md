# P-wave check by finite differences

These scripts check the P-wave virtual of MadLoop's onia layer (`onia_loop.inc`, P-wave block) at a relative momentum q0 ≠ 0 of the constituents. There, every loop integral is an ordinary one, so the check does not depend on TIRust's threshold (Coulomb) lanes.

The check compares two numbers with the same weights:
- **mode 1, the dual derivative:** MadLoop's dual numerators and the doubled-propagator integrals give Σ_h Σ_d conj(W2(h,d)) dM(h; e_d) at q0.
- **mode 2, the value:** the same weights W2 are frozen and used as the value weight. MadLoop then gives Σ_h conj(W2(h,d)) M(h; q0 ± h e_d), and central differences are taken with Richardson steps h and h/2.

The directions e_d are two unit vectors perpendicular to q0. The bound state is at rest, with mass 2 sqrt(m² + q²) at each point, so every external leg stays on its mass shell.

This setup avoids two problems:
- **IR non-analyticity:** a shift off the mass shell changes the IR poles of the on-shell integrals into logarithms of the off-shellness, which are not analytic at the base point.
- **Point improvement:** for the same reason, MadLoop's own improvement of off-shell points (`ImprovePSPoint`) changes the function being differenced.

| File | Role |
|---|---|
| `build.sh`, `fix_config.py` | scratch TIRust build (TIRust revision, 0001 and follow-up patches, `vendor/tirust` bridge, optionally patched with `BRIDGE_PATCH`, its pole kernels optionally on the `oneloop` revision `ONELOOP_REV`) and its installation metadata |
| `pw_check.py` | generates the outputs and evaluates them at points (`onia_loop_check.py`) |
| `fd_patch.py`, `fd_patch2.py`, `fd_patch3.py` | add the test modes to a generated `onia_loop.f`, in double and quadruple precision |
| `fd_setup.sh`, `pw_fd_driver.f`, `pw_fd.mk` | patch, build and run the check |
| `pw_modes.py` | the full virtual of an output in the given `CTModeRun` modes (`1,4,-1:1.0d-40`), optionally at a relative momentum with on-shell constituents |
| `psi_chic.py`, `psi_chic_ref.py` | e+ e- > J/psi chi_cJ (an S-wave and a P-wave state) against the closed form of Wang, Ma and Chao and the helicity tables of Dong, Feng and Jia (below) |
| `qp_floor.py` | the floor of quadruple precision: the loop groups summed in quadruple precision against MadLoop's sum, for several b and t masses (below) |
| `unequal_mass.py` | P waves with unequal constituent masses (e+ e- > B_c b c~): the computed 1/ε against `ONIA_IR_POLES` and against it without the self term and LDME counterterm (app. B of the two-P-wave manuscript); see benchmark.pdf, sect. 5.2.1 |

The paths are those of the scratch workspace in which the scripts ran.

Results at q0 = (0.3, 0.2, 0.4) GeV, m_c = 1.55, μ = 2 m_c. The relative differences are in the finite part and in the 1/ε and 1/ε² coefficients:

| Process | Dual derivative against finite differences | Share of the doubled-line integrals |
|---|---|---|
| u u~ > Ps(2\|3P1), m_e = 10 GeV | 8e-13, 1e-12, 2e-11 | none: no loop line carries q |
| a a > chic0(1\|3P01) | 5e-11, 7e-9 (3e-16 absolute) | 80% |
| g g > chic0(1\|3P01) | 1e-10, 9e-11, 2e-12 | 55% |

For a a > chic0, the same evaluation with MadLoop's stability checks (loop reversal) has an estimated accuracy of 1.8e-13 and return code 213.

The patches add the test modes to the double- and quadruple-precision routines alike. With `CTModeRun=4`, the same check runs in quadruple precision. Use `ImprovePSPoint=-1` and the Richardson step h = 3e-4 GeV (1e-3 in double precision):

| Process | QP dual derivative against QP finite differences | QP against DP dual derivative |
|---|---|---|
| u u~ > Ps(2\|3P1), m_e = 10 GeV | 3e-13, 4e-14, 3e-12 | 4e-13, 2e-13, 2e-12 |
| a a > chic0(1\|3P01) | 2e-14, 5e-11 (1/ε, 1/ε² = 0) | 2e-14, 9e-12 |
| g g > chic0(1\|3P01) | 6e-13, 7e-13, 9e-14 | 2e-14, 4e-13, 7e-14 |

At the same q0 with on-shell constituents, the full virtual (`pw_modes.py`) agrees between double and quadruple precision to 4e-13. The rescue with a forced `MLStabThres` returns code 323.

## Two P-wave states: the mixed derivative

`fd2_patch.py` and `pw2_fd_driver.f` do the same check for two P-wave states (`e+ e- > chic0(1|3P01) hc(1|1P11) [virt=QCD]`, double precision only):
- **Relative momenta.** Each state gets its own relative momentum (`ML5_0_PW2_SET_RELMOM`), base values q0₁ = (0.3, 0.2, 0.4) and q0₂ = (−0.25, 0.35, 0.15) GeV.
- **Mode 1.** It keeps the frozen mixed weights W3 of the four pairs of directions perpendicular to q0₁ and q0₂, and sets W0 = W1 = W2 = 0. MadLoop then returns the ε₁ε₂ component.
- **Mode 2.** It uses each frozen W3 as the value weight. The mixed central difference [f(+,+) − f(+,−) − f(−,+) + f(−,−)]/4h², Richardson with h = 4e-3 and 2e-3 GeV, is summed over the four pairs.
- **Kinematics.** √s = 10.58 GeV, the bound states of mass 2 sqrt(m² + q_i²), the direction of the first one fixed.

At q0 ≠ 0, TIRust (59aee45) refuses 23 single-doubled pentagons (N = 5, rank 4) per evaluation with status 1, so the full comparison has missing pieces. The driver therefore runs with the environment variable `ONIA_NRANGE='nmin nmax'` (a patch of the output's `TIRUST_interface.f`). It drops the loop groups with fewer or more lines from both modes alike, and the comparison stays exact for the groups that are left.

| Loop groups | Before the rank fix (finite, 1/ε) | After |
|---|---|---|
| bubbles | 2e-9, 2e-8 | — |
| triangles | 1e-7, 2e-9 | — |
| boxes | 2.5e-3, 0.24 | 1.2e-8, 2.4e-7 |
| N = 2–4 | 0.64, 0.75 | 8e-6, 6e-7 |
| N = 2–4, the states in the other order | 0.50, 6.7 | 2.5e-6, 4.5e-6 |

**What the check found.** The polynomial routines of two-P-wave outputs (`NCOEF_R`, `COEFTORANK_MAP`) were written for rank R+1, while `LOOPMAXCOEFS` was sized for R+2. In a rank-3 loop group, `NCOEF_R(RANK+2)` then read past the array. The tripled and two-doubled terms of every box were dropped without a message, and the virtual depended on the order of the two states. The exporter now writes both at R + npwave (`loop_onia.loop_rank_increase`).

**Precision.** The N = 2–4 totals are about 700 times smaller than the box and counterterm parts, so the 1e-6 level there is the cancellation of the finite-difference error of the parts (1e-8).

## Threshold values against the literature (TIRust 48958d1)

With TIRust 48958d1 (part of PR #56, built by `build.sh` with `REV=48958d1` and `FOLLOWUP=0001-followup-pwave-collinear.patch`), every integral of the 2 → 1 processes is served at zero relative momentum, the Coulomb box with a doubled gluon included. `pw_check.py` gives these values at μ = M = 2 m_c = 3.1 GeV, n_l = 3, in units of Born·α_s/2π. Every point is stable in double precision (code 213):

| Process | 1/ε² | 1/ε | Finite | Reference | Relative difference |
|---|---|---|---|---|---|
| a a > chic0(1\|3P01) | 0 | 0 | 0.357514045171 | 2 C_F(π²/4 − 7/3) (Barbieri et al.) | 9e-13 |
| a a > chic2(1\|3P21) | 0 | 0 | −10.666666666667 | −8 C_F | 3e-14 |
| g g > chic0(1\|3P01) | −6 | −9 | 17.161920646805 | Petrelli B^[1](3P0) | 2e-14 |
| g g > chic2(1\|3P21) | −6 | −9 | −1.735194861067 | Petrelli B^[1](3P2) | 1e-13 |
| g g > Jpsi(1\|3P08) | −6 | −12 | 11.759717345988 | corrected B^[8](3P0) (`onia_fks` RECOMPUTED_GG) | 1e-13 |
| g g > Jpsi(1\|3P28) | −6 | −12 | 15.182917379447 | corrected B^[8](3P2) | 1e-13 |
| a g > Jpsi(1\|3P08) | −3 | −7.5 | 3.357514045171 | Ozcelik H^(1) | all 8 printed digits |
| a g > Jpsi(1\|3P28) | −3 | −7.5 | 10.717181476647 | Ozcelik H^(1) | all 8 printed digits |
| g g > etac(1\|1P18) | −6 | −18 | 2.591452358839 | `onia_fks` ONIA_GG_1P18 (CDR, Larin γ5) | 1.5e-13 |

- **Conversions.** The references are those of `research/literature.md`, converted to MadLoop's normalisation by its section B.5.
- **Octet colour factor.** The octets confirm the corrected colour factor of the four-gluon-vertex diagram. Petrelli et al. print 15.0397510 and 11.2057601.
- **¹P₁[8] pole.** For `g g > etac(1|1P18)`, −18 includes the P → S pole −2C_A of arXiv:2402.19221 (appendix B). `ONIA_IR_POLES` includes it (`ONIA_PW_SOFT_POLE`) and predicts −18; without it, the prediction would be −12.
- **Earlier TIRust revisions.** They refuse the doubled-gluon Coulomb box. The point is then reported unstable, and the 1/ε coefficient misses −2C_F: γγ → χ_c0 gave −8.6838 and −8/3.

## The P → S pole of the prediction

`ONIA_IR_POLES` adds the P → S single pole of the soft current (arXiv:2402.19221, appendix B; `ONIA_PW_SOFT_POLE`). `pw_check.py` compares the predicted 1/ε coefficient with the computed one, with TIRust 48958d1 and four points per process. Every point is stable (code 213). The size of the term is that of a print of `PSPOLE` in the generated code, in units of Born·α_s/2π:

| Process | √s (GeV) | P → S term | Prediction − computed (1/ε) |
|---|---|---|---|
| g g > etac(1\|1P18) | M | −6 (−2 C_A) | < 1e-12 |
| u u~ > chic0(1\|3P01) g | 20 | −11.1 to −39.3 | ≤ 7e-9 |
| g u > chic0(1\|3P01) u | 20 | +10.9 to +42.7 | ≤ 2e-9 |
| e+ e- > Jpsi(1\|3PJ8) g, J = 0, 1, 2; etac(1\|1P18) g | 10 | 0 (≤ 5e-20) | ≤ 7e-11 |
| the 2 → 1 processes above but g g > etac(1\|1P18) | M | 0 | < 1e-12 |

- **Why it vanishes.** For equal masses the self term and the LDME counterterm cancel for every S-wave state S'. What is left are the cross vectors, which vanish by colour conservation when the bound state has a single coloured partner, as in the e+e- cases. In the 2 → 1 cases other than ¹P₁[8], every S' Born vanishes.
- **2 → 2 processes.** TIRust 48958d1 serves every integral of `u u~ > chic0 g` and `g u > chic0 u` at these points, but not those of `g g > chic0 g` (doubled-gluon pentagons). TIRust 370b632 (in PR #56) serves them: all four points stable (213), poles as predicted (P → S term +6.9 to +9.9) to 1.3e-9, finite parts equal to forced QP within 4e-12. The finite parts of these processes have no reference.

## An S-wave and a P-wave state: e+ e- > J/psi chi_cJ (TIRust 08cca00)

`psi_chic.py` generates `e+ e- > Jpsi(1|3S11) chicJ(1|3PJ1) / z t t~ b b~ QED=2 [virt=QCD]` for J = 0, 1, 2:
- **Setup.** m_c = 1.5 GeV, √s = 10.58 GeV, μ = 2 m_c. The b and t quarks are excluded from the loops, which is the limit m_b, m_t → ∞ of the n_l = 3 reference.
- **Points.** The virtual is evaluated at the four Gauss–Legendre nodes in cos θ.
- **Comparison.** The cross-section average of the finite part is compared with `psi_chic_ref.py`. That is the NLO cross section of arXiv:1107.2646 (appendix), converted from its LaTeX source and from n_f = 4 to MadLoop's n_l = 3 scheme (`research/literature.md`, C.6). The exclusive process has no real emission, and the virtual is UV and IR finite.

The pinned TIRust installation has no raised powers, and MadGraph refuses the output. With TIRust 08cca00 (`build.sh`, bridge v8 of the two-P-wave work), relative to the reference:

| J | Reference ⟨c0⟩ | DP (CTModeRun −1) | QP (CTModeRun 4) | Poles DP, QP |
|---|---|---|---|---|
| 0 | 14.8174900328 | 1.4e-9 | 1.2e-16 | 3.5e-13, 7.7e-15 |
| 1 | −1.2552345934 | 3.4e-7 | 2.3e-15 | 2.0e-13, 1.9e-15 |
| 2 | −4.3105820930 | 2.6e-8 | 1.9e-15 | 5.4e-13, 2.4e-15 |

- **Speed and codes.** Every node returns 213 in double precision, at 0.2 s per point. Quadruple precision takes 12 s per point.
- **Exact momenta.** `--exact` evaluates the virtual at √s = 18.5 GeV, where the momenta are exact in binary64. There quadruple precision agrees to 5.6e-15, 6.8e-15 and 1.3e-14.
- **Double precision against quadruple, per node.** The difference is at most 4.5e-7 (χ_c1). MadLoop's loop-reversal estimate is within a factor 5 of it, sometimes below.
- **The two quadruple-precision directions.** CTModeRun 4 and 5 are identical bit for bit, because MadLoop rounds each loop group to double precision before the sum.
- **Helicities.** The LO shares of λ = 0 and |λ| = 1 equal those of the helicity amplitudes of arXiv:1107.4351 (eq. (3.3)) to 2e-15. Its table 2 agrees within its two to three digits.
- **J/ψ η_c with the same exclusion.** Quadruple precision gives 24.53642097 at every point, Gong–Wang's 24.53642 to its seven digits.

### Heavy quarks instead of their exclusion

The first runs decoupled b and t by giving them masses of 10 and 100 TeV (`/ z` only, as the e+ e- cases of `tools/onia_loop_validate.py` did before 2026-11-24). Quadruple precision then stopped at 1e-8 to 1e-7, also with exact momenta.

`qp_floor.py` patches such an output to sum the loop groups in quadruple precision, and prints both sums. The cause:
- The counterterms and loop groups of the heavy quarks grow like M² and cancel. |counterterms| + Σ|loop group| is 8.8e8 times the total for χ_c2, and 5.9e7 for J/ψ η_c.
- MadLoop stores every loop group as a double-precision number (`LOOPRES`; `MP_TIRUST_LOOP` rounds the quadruple-precision value). The result therefore keeps 1e-16 times the parts.

χ_c2 at √s = 10.58 GeV, against the reference:

| m_b, m_t (GeV) | Parts / total | MadLoop (rounded) | Quadruple-precision sum |
|---|---|---|---|
| 1e3, 1e3 | 1.8e5 | 3.0e-7 | 3.0e-7 |
| 1e4, 1e4 | 1.7e7 | 3.7e-9 | 3.0e-9 |
| 1e4, 1e5 | 8.8e8 | 1.2e-7 | 1.5e-9 |
| 1e5, 1e5 | 1.7e9 | 3.2e-8 | 3.0e-11 |

- **Summed in quadruple precision.** What is left is the decoupling remainder of the heavy quarks, which falls like 1/M².
- **Physical masses.** With m_b = 4.75 and m_t = 173 GeV, the parts are 2.7e3 times the total, and the rounding is 1e-12.
- **J/ψ η_c.** The rounding is 6e-9 at 10 and 100 TeV.
