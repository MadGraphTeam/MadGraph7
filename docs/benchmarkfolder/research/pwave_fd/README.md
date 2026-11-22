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
| `build.sh`, `fix_config.py` | scratch TIRust build (TIRust revision, 0001 and follow-up patches, `vendor/tirust` bridge) and its installation metadata |
| `pw_check.py` | generates the outputs and evaluates them at points (`onia_loop_check.py`) |
| `fd_patch.py`, `fd_patch2.py`, `fd_patch3.py` | add the test modes to a generated `onia_loop.f`, in double and quadruple precision |
| `fd_setup.sh`, `pw_fd_driver.f`, `pw_fd.mk` | patch, build and run the check |
| `pw_modes.py` | the full virtual of an output in the given `CTModeRun` modes (`1,4,-1:1.0d-40`), optionally at a relative momentum with on-shell constituents |

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

## Threshold values against the literature (TIRust 48958d1)

With TIRust 48958d1 (PR #50, built by `build.sh` with `REV=48958d1` and `FOLLOWUP=0001-followup-pwave-collinear.patch`), every integral of the 2 → 1 processes is served at zero relative momentum, the Coulomb box with a doubled gluon included. `pw_check.py` gives these values at μ = M = 2 m_c = 3.1 GeV, n_l = 3, in units of Born·α_s/2π. Every point is stable in double precision (code 213):

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
- **¹P₁[8] pole.** For `g g > etac(1|1P18)`, −18 includes the P → S pole −2C_A of arXiv:2402.19221 (appendix B). `ONIA_IR_POLES` does not include that term yet and predicts −12.
- **Earlier TIRust revisions.** They refuse the doubled-gluon Coulomb box. The point is then reported unstable, and the 1/ε coefficient misses −2C_F: γγ → χ_c0 gave −8.6838 and −8/3.
