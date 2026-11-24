# Literature benchmarks for one-loop QCD virtuals of quarkonium processes (MadLoop-Onia validation)

Compiled 2026-11-21. Sources were read from the arXiv LaTeX sources (`arxiv.org/e-print/...`) and the ar5iv HTML renderings (for equation numbers), plus INSPIRE metadata. Equation numbers are the **arXiv-version** numbers as rendered by ar5iv; journal numbering can differ.

**The numbers in this report.** Every numerical value labelled "eval." was computed by me (plain double precision, Perl) from the quoted analytic formula. Each such evaluation was checked against at least one number printed in the source (see "validation").

**Status flags.**
- `[VERIFIED]`: read directly in the primary source and cross-checked.
- `[DERIVED]`: my own derivation or conversion from verified inputs.
- `[UNVERIFIED]`: not checked against a primary source.

---

## 0. Conventions and conversion recipes (read first)

### 0.1 Target normalisation (MadLoop/BLHA)

    2 Re(M_B^* M_1loop)  =  (alpha_s/2pi) * (4pi)^eps/Gamma(1-eps) * (mu_R^2/Q^2)^eps * [c2/eps^2 + c1/eps + c0] * |M_B|^2

- `M_B` is the 4-dimensional (HV) Born.
- For a UV- and IR-finite process, define the interference ratio in units of alpha_s/pi: `R = 2Re(B^*V)/|B|^2 = (alpha_s/pi) * a`. Then `c2 = c1 = 0` and **c0 = 2a**.
- For a 1->2 or 2->1 process whose Born contains no alpha_s, `a` does not depend on mu_R or Q.

### 0.2 Converting Petrelli et al. (hep-ph/9707223) to BLHA `[DERIVED]`

Petrelli et al. write `sigma_V = sigma_Born^(D) * (alpha_s/pi) * f_eps(M^2) * {P2/eps^2 + P1/eps + B + b0/eps_UV + Coulomb}`, where:

- `f_eps(Q^2) = (4 pi mu^2/Q^2)^eps * Gamma(1+eps)` [eq. (198)].
- `b0 = 11/6 C_A - 2/3 T_F n_f` [eq. (210)], with n_f = number of flavours lighter than the bound quark.
- The heavy-quark mass is on-shell and already renormalised [eq. (122)].
- alpha_s is left bare. The MSbar replacement is eq. (123): `alpha_s^bare = mu^{2eps} alpha_s [1 - (alpha_s/2pi) b0 (1/eps - gamma_E + ln 4pi)]`.

Use `Gamma(1+eps) Gamma(1-eps) = 1 + zeta_2 eps^2 + O(eps^4)`. For a Born of order alpha_s^2, with Q = M (the bound-state mass) and L_R = ln(mu_R^2/M^2):

    c2 = 2 P2 ,   c1 = 2 P1 ,   c0 = 2 B + 2 P2 * pi^2/6 + beta0 * L_R          (beta0 = 2 b0 = 11/3 C_A - 4/3 T_F n_l)

For general Q, with L_Q = ln(Q^2/M^2):

    c1(Q) = c1 + c2 L_Q
    c0(Q) = c0 + c1 L_Q + c2 L_Q^2/2

For a Born of order alpha alpha_s, replace beta0 L_R by (beta0/2) L_R. Drop the Coulomb term `C pi^2/(2v)` (see 0.5).

### 0.3 Converting the SCET-type finite remainders (Abreu et al. 2211.08838, Ozcelik 2501.04188) `[DERIVED, checked numerically]`

These papers give the "hard function" `H^(1) = 2 Re F^{fin,(1)}` in units of alpha_s^(n_l)/pi, at mu_R = mu_F = mu, with `l = ln(mu^2/m_Q^2)`. The IR poles are removed with an MSbar Z_IR, i.e. poles carry `S_eps = (4pi)^eps e^{-eps gamma_E}`. Converting `e^{-eps gamma_E}` to `1/Gamma(1-eps)` gives `c0 = 2 H^(1) + c2 * pi^2/12`. Therefore, at mu_R = mu_F = mu = Q:

    gg -> colour singlet or octet :  c0 = 2 H^(1)(l) - C_A pi^2/6
    gamma g -> colour octet       :  c0 = 2 H^(1)(l) - C_A pi^2/12

Validation: this reproduces Petrelli's B for 1S0[1], 1S0[8], 3P0[1] and 3P2[1] to better than 1e-7 (Section B.4).

### 0.4 Renormalisation scheme (same as the implementation)

- On-shell heavy-quark mass and wave function.
- MSbar alpha_s with n_l light flavours; heavy quarks decoupled (zero-momentum subtraction).

Consequences:

- **gg-initiated 2->1.** Heavy-quark loops appear at one loop only as external-gluon self-energies. They cancel exactly against the decoupling counterterm, so there is no heavy-quark constant. This is consistent with Abreu et al., who work in exactly this scheme and agree with Petrelli.
- **q qbar -> 3S1[8].** The s-channel gluon carries q^2 = M^2 = 4m^2. The heavy-quark vacuum polarisation gives a finite, mu-independent term:

      Delta(2Re B^*V/|B|^2) = (alpha_s/pi) * 4 T_F * Re I(M^2/m'^2)  per heavy flavour of mass m',
      I(z) = Int_0^1 dx x(1-x) ln|1 - z x(1-x)|

  For m' = m (the bound quark): I(4) = -4/9, so **Delta = -(16/9) T_F (alpha_s/pi) = -(8/9)(alpha_s/pi)**. Petrelli et al. omit this term (see B.3, erratum 1).

  Values of I(z) (eval.): b-loop in charmonium (m_b = 4.75 GeV) I = -0.0139; c-loop in bottomonium (M = 9.46 GeV, m_c = 1.5 GeV) Re I = +0.308.
- **Literature using alpha_s^(n_l+1) MSbar with an active heavy quark** (Zhang-Gao-Chao, Gong-Wang, Dong-Feng-Jia). Convert with `alpha_s^(nl+1)(mu) = alpha_s^(nl)(mu) [1 + (alpha_s/pi)(T_F/3) ln(mu^2/m^2)]`. For a Born of order alpha_s^2: `a^(nl) = a^(nl+1) + (2T_F/3) ln(mu^2/m^2)`.

### 0.5 Coulomb singularity

With strict threshold expansion (v = 0 before loop integration) in dimensional regularisation, the potential region is scaleless and the Coulomb `1/v` term does not appear. All "Coulomb-subtracted" literature values compare directly. Petrelli et al. show that the `pi^2/(2v)` term (colour factor C_F for singlets, C_F - C_A/2 for octets) cancels against the NLO LDME and leaves no finite remainder [Sec. 6, eqs. (134)-(139)].

### 0.6 CDR vs HV, 4-d vs D-d projectors, gamma5

- **CDR vs HV.** Catani-Dittmaier-Trocsanyi (hep-ph/0011222, Sec. 2.1) state that choosing d- or 4-dimensional external states gives O(eps) differences at one loop. The finite remainder after subtracting `I(eps) x Born(scheme)` is scheme independent; `gamma~_j^CDR = 0` [eq. (20)], so HV = CDR. Petrelli normalise to the **D-dimensional** Born (with D-dim polarisation sums (17)-(19), N_J eq. (22), Borns (251)-(260)). Their bracket is therefore directly the HV result normalised to the **4-dimensional** Born. This is confirmed numerically by agreement with the independent results of 2211.08838 and 2501.04188.
- **FDH/DR.** Results in DR would need `gamma~_q^DR = C_F/2`, `gamma~_g^DR = C_A/6` and `beta~0^DR = C_A/3` [CDT eqs. (11), (20)]. Campbell-Maltoni-Tramontano used FDH.
- **gamma5 (1S0).**
  - Petrelli et al. (Sec. 2.1) computed the 1S0[1,8] virtuals in both HV and NDR and found identical results diagram by diagram.
  - Abreu et al. (2211.08838, Sec. 2) note no difference between HV and NDR, since there is a single gamma5 in the trace.
  - Jia-Yang-Sang-Xu (1104.1418) used HV with West's trace formula. Their O(v^0) coefficient is the same whether the relative momentum q is 4- or D-dimensional (only O(v^2) differs).
  - One-loop 1S0 results are therefore gamma5-scheme independent. No finite Larin-type counterterm is needed at this order.
- **P-wave projectors.** Ozcelik (2501.04188, Secs. 5-6) notes that using the D-dimensional helicity decomposition matters only at **two loops**: the chi_2 J_z = 0 (A2) tree is O(eps) in d dimensions. At one loop, 4-d projectors give the same finite part.

---

## A. Colourless partners, UV and IR finite (1->2 or 2->1)

For all entries in this section: no poles, c0 = 2a, independent of mu_R and Q. Production and decay are related by crossing and have the same ratio.

| process | amplitude factor | rate factor (a = coefficient of alpha_s/pi) | c0 = 2a | source |
|---|---|---|---|---|
| eta_c <-> gamma gamma (1S0[1]) | 1 + C_F(alpha_s/pi)(pi^2/8 - 5/2) | C_F(pi^2/4 - 5) = **-3.37679853** | **-6.75359706** | `[VERIFIED]` 1104.1418, Table I (sum of diagrams); 2211.08838 eq. (91) `a_1^(1)` and H^(1)_{gamma gamma,[1]} = -3.376798533 (Sec. 5); 0910.4071 eq. (35) (r -> inf limit) |
| J/psi <-> l+ l- (vector current via gamma*; also the vector part of Z*) | 1 - 2 C_F alpha_s/pi | -4 C_F = **-16/3 = -5.3333333** | **-10.6666667** | `[VERIFIED]` Gong-Wang 0712.4220 eq. (10); Sang-Chen 0910.4071 Sec. V (F_2(3S1)) (Barbieri et al. 1975, Celmaster 1979) |
| chi_c0 <-> gamma gamma | 1 + C_F(alpha_s/pi)(pi^2/8 - 7/6) | C_F(pi^2/4 - 7/3) = pi^2/3 - 28/9 = **+0.178757023** | **+0.35751405** | `[VERIFIED]` 1511.06288 eq. (7); 2501.04188 H^(1)_{gamma gamma,A3} = 0.17875702258534 (eq. (90): C_F a_3); 0910.4071 eq. (59) |
| chi_c2 <-> gamma gamma, helicity +-2 (photon helicities opposite) | 1 - 2 C_F alpha_s/pi | -4 C_F = **-5.3333333** | **-10.6666667** | `[VERIFIED]` 1511.06288 eq. (8b); 2501.04188 H^(1)_{gamma gamma,A1} = -5.3333 |
| chi_c2 -> gamma gamma, helicity 0 (equal photon helicities) | Born = 0; A^(1)(lambda=0)/A^(0)(lambda=2) = -(sqrt6/6) C_F (3pi^2/8 - 6 ln2 + 1)(alpha_s/pi), i.e. magnitude **0.2951464 alpha_s/pi** | interference = 0 at this order | 0 | `[VERIFIED]` 1511.06288 eq. (8a); 2501.04188 a^(1)_{gamma gamma,A2;F} = a_2^(1) (eq. (89)). f_{0/2} = (alpha_s/pi)^2 (8 + 3pi^2 - 48 ln2)^2/216 (1511.06288 eq. (17)). Testable only through \|M_loop\|^2 (loop-induced helicity). |
| chi_c1 -> gamma gamma | 0 (Landau-Yang), at tree level and one loop | 0 | 0 | zero test |
| J/psi -> gamma gamma gamma | (angle dependent) | **integrated only**: Gamma = Gamma_0 [1 - 12.630 alpha_s/pi] | -- | `[VERIFIED]` Feng-Jia-Sang 1210.6337 eq. (1); Mackenzie-Lepage PRL 47 (1981) 1244 (-12.6, not opened); 2603.26199 eq. (1) (-12.63). No pointwise values published. 2603.26199 ships unlabelled grid files (`jpsinlo`, `NLOdata`) that may be phase-space samples. `[UNVERIFIED, unused]` |
| B_c -> l nu (axial current, 1S0[1], unequal mass; needs massive lepton) | 1 + (alpha_s/pi) C_F[(3/4)((r-1)/(r+1)) ln r - 3/2], r = m_b/m_c; i.e. (alpha_s/pi)[((m_b-m_c)/(m_b+m_c)) ln(m_b/m_c) - 2] | 2x amplitude coeff.; m_b = 4.75, m_c = 1.5: **-2.801213** | -5.602427 | `[VERIFIED]` Braaten-Fleming hep-ph/9501296 eq. (C0) (sec. 3); eval. agrees with their C_0 = 0.85 at alpha_s = 0.34 |
| B_c^* -> l nu / u dbar -> B_c^* (vector current, unequal mass) | 1 + (alpha_s/pi) C_F[(3/4)((r-1)/(r+1)) ln r - 2] | m_b = 4.75, m_c = 1.5: **-4.134547** | -8.269094 | `[VERIFIED]` Sang-Zhang-Zhou 2210.02979 eq. (9) (C^(1)(r); after Hwang-Kim); equal-mass limit gives 1 - 2C_F alpha_s/pi |

Notes:
- **Z couplings.** The axial current does not couple to 3S1 at O(v^0). For 1S0, the axial current gives f P^mu, which vanishes against a massless lepton current. Z*/gamma* -> J/psi therefore has the same factor as the photon.
- **h_c.** 1^{+-} has no simple colourless 1->2 or 2->1 channel at LO: h_c -> gamma gamma is forbidden by C, and e+e- -> gamma* -> h_c by P. The chi_cJ and h_c "electromagnetic" checks are therefore the e+e- -> chi_cJ gamma helicity amplitudes (Section C.2).

---

## B. 2->1 processes with IR poles (gg -> 1S0, 3P0, 3P2 [1,8]; q qbar -> 3S1[8])

### B.1 Primary source

Petrelli, Cacciari, Greco, Maltoni, Mangano, "NLO production and decay of quarkonium", hep-ph/9707223 v2, Nucl. Phys. B514 (1998) 245.

- **Born processes.** The non-vanishing 2->1 Borns are q qbar -> 3S1[8] and gg -> 1S0, 3P0, 3P2 [1,8] (Sec. 7, "Inclusion of virtual corrections").
- **Also non-zero: gg -> 1P1[8].** This channel has an f^{abc} (C-odd) Born and is not in Petrelli et al. Per 2402.19221 (Sec. 5, footnote), its NLO is implemented analytically in HELAC-Onia but **no published virtual** was found `[UNVERIFIED]`.
- **Vanishing 2->1 Borns.** gg -> 3S1[1,8] and gg -> 3P1[1,8] (Landau-Yang; for 3S1[8] the tree vanishes on shell), and q qbar -> 3PJ[8] (a vector current does not produce P = + states). Their 2->1 virtual interference is zero.
- **Diagram tables** (D_k, colour-stripped, with colour factors f_k^[1], f_k^[8]): Table 1 (1S0), Table 2 (3P0), Table 3 (3P2), Table 4 (q qbar 3S1[8]).
- **Virtual decomposition.** `Gamma_V = Gamma_Born (alpha_s/pi) f_eps(M^2) Sum_k D_k f_k` [eq. (121)]. Production: `sigma_V = (alpha_s mu^2eps/pi) sigma_0 f_eps(s) Sum_k D_k f_k delta(1-x)` [eq. (195)].
- **D-dimensional Borns.** Eqs. (240)-(249) (decay) and (251)-(260) (production). For example:
  - `sigma_B(gg -> 1S0[1]) = (C_F/D_A^2)(2 alpha_s^2 mu^{4eps} pi^3/m^5) (1-2eps)/(1-eps) delta(1-x) <O_1>`
  - 3P0 carries `1/((1-eps)(3-2eps))`; 3P2 carries `(6-13eps+4eps^2)/((1-eps)(3-2eps))/((1-eps)(5-2eps))`; q qbar 3S1[8] carries `(1-eps)/(3-2eps)`.
  - Polarisation counts: N_0 = 1, N_1 = (3-2eps)(1-eps), N_2 = (5-2eps)(1-eps) [eq. (22)].
- **Projectors** [spin eqs. (7)-(8), colour (9)-(10), L-projection (11)-(14), J polarisation sums (17)-(19)]:
  - `Pi_0 = (1/sqrt(8m^3)) (P/2 - q - m) gamma5 (P/2 + q + m)`
  - `Pi_1^alpha = (1/sqrt(8m^3)) (P/2 - q - m) gamma^alpha (P/2 + q + m)`
  - colour: `C_1 = delta_ij/sqrt(Nc)`, `C_8 = sqrt2 T^c_ij`
  - LDME normalisation: `O_1 = O_1^BBL/(2Nc)`, `O_8 = O_8^BBL`.

### B.2 Formulas as printed in the source `[VERIFIED: transcribed from LaTeX]`

Colour-singlet gg [eq. (124)]:

    Gamma_V = Gamma_B (alpha_s/pi) f_eps(M^2) { b0/eps_UV + C_F pi^2/(2v) - C_A (1/eps^2 + 11/(6 eps)) + (2/(3 eps)) n_f T_F + B^[1] }
    B^[1](1S0) = C_F(-5 + pi^2/4) + C_A(1 + 5 pi^2/12)                          [eq. (125)] = 11.9602070
    B^[1](3P0) = C_F(-7/3 + pi^2/4) + C_A(1/3 + 5 pi^2/12)                      [eq. (126)] = 13.5157625
    B^[1](3P2) = -4 C_F + C_A(1/3 + (5/3) ln2 + pi^2/6)                         [eq. (127)] =  4.0672048

Colour-octet gg [eq. (128)]:

    { b0/eps_UV + (C_F - C_A/2) pi^2/(2v) - C_A (1/eps^2 + 7/(3 eps)) + (2/(3 eps)) n_f T_F + B^[8] }
    B^[8](1S0) = C_F(-5 + pi^2/4) + C_A(2 + 7 pi^2/24)                          [eq. (129)] = 11.2591053
    B^[8](3P0) = C_F(-7/3 + pi^2/4) + C_A(17/54 + (35/27) ln2 + 7 pi^2/24)      [eq. (130)] = 12.4546777  (see erratum 2)
    B^[8](3P2) = -4 C_F + C_A(23/36 + (7/9) ln2 + 5 pi^2/12)                    [eq. (131)] = 10.5376823  (see erratum 2)

q qbar <-> 3S1[8] [eq. (132)]:

    { (C_F - C_A/2) pi^2/(2v) + b0/eps_UV + C_F(-1/eps^2 - 3/(2 eps)) - C_A/(2 eps) + B[3S1(8)] }
    B[3S1(8)] = C_F(-8 + 2pi^2/3) + C_A(50/9 + (2/3) ln2 - pi^2/4) - (10/9) n_f T_F           [eq. (133)]

In eq. (132) the constant is written "A[3S1(8)]", but eq. (133) defines "B[...]". This is a notation typo only. The same constant appears as "A" in Petrelli, hep-ph/9603439, Table 1 and eq. (11).

**Light-quark bubble.**
- gg channels: diagram j = `(-2/(3 eps_UV) + 2/(3 eps_IR)) n_f T_F` is scaleless. It feeds only the poles (gamma_g) and leaves no constant in B.
- q qbar channel: the s-channel bubble gives `(-2/(3 eps_UV) - 10/9) n_f T_F` (Table 4, d_8). Its constant `-(10/9) n_f T_F` is in B.

### B.3 Errata and corrections

**Erratum 1: missing heavy-quark bubble in q qbar -> 3S1[8]** `[VERIFIED]`.
- Hagiwara, Sumino, Yokoya, arXiv:0804.1014 (PLB 666 (2008) 71), text below eq. (13): "the term -8/9 in h_qq^(8) is missing in Ref. [Petrelli]". The source is the top-quark loop in the s-channel gluon: `2[Pi_t(4m_t^2) - Pi_t(0)] = -8/9`. The Petrelli et al. authors confirmed the error (footnote).
- Czakon-Mitov 0812.0353, eq. (qqbar-8-beta): the exact q qbar -> t tbar octet threshold expansion contains the n_l-independent "-8/9" plus a "-(n_l+1)/3 ln(mu^2/m^2)" term, and agrees with Hagiwara et al.
- In the decoupling scheme the correction is mu-independent:

      B'[3S1(8)] = B[3S1(8)] - (16/9) T_F

**Erratum 2 (new; found in this survey): octet P-wave colour factor of the four-gluon-vertex diagram (h)** `[DERIVED + numerically supported]`.

- **What is wrong.** Tables 2 and 3 give f_h^[8] = C_A for diagram h, the 4-gluon vertex attached through two gluons to the heavy-quark line (figure ggcc-prod, panel h).
- **Colour algebra.** The relevant colour terms of diagram h are f^{ace}f^{bde} T^c T^d = (F^a F^b)_{cd} T^c T^d. This is identical to the colour of diagram g, for which Petrelli correctly give f_g^[1] = C_A and f_g^[8] = C_A/2. Projecting onto the d^{abc} (C-even) octet Born gives:
  - singlet: Tr(F^a F^b)/2 = C_A x Born
  - octet: (1/4) Tr(D^e F^b F^a) = (C_A/2) x Born, since Tr(F^a F^b D^c) = (N/2) d^{abc}

  The f^{abe}f^{cde} term projects to zero on both singlet and d-type octet states. Hence **f_h^[8] = C_A/2**.
- **Where it matters.** D_h = 0 for 1S0, so only 3P0[8] and 3P2[8] are affected. D_h(3P0) = -19/27 + (70/27) ln2 and D_h(3P2) = -5/9 - (10/9) ln2.
- **Corrected constants** (B_corr = B_printed - (C_A/2) D_h):

      B^[8](3P0)_corr = C_F(-7/3 + pi^2/4) + C_A(2/3 + 7 pi^2/24)             = 10.8146609   (printed 12.4546777; Delta = -1.6400168)
      B^[8](3P2)_corr = -4 C_F + C_A(11/12 + (4/3) ln2 + 5 pi^2/12)           = 12.5262609   (printed 10.5376823; Delta = +1.9885786)

  The ln 2 drops out of B^[8](3P0), matching the singlet/octet 1S0 pattern.
- **Independent support.** Ozcelik, arXiv:2501.04188 (JHEP 06 (2025)), gives one-loop finite remainders for gg -> 3PJ[8]:
  - helicity amplitudes A3 (chi_0) and A1 (chi_2, J_z = +-2): H^(1) = 7.54398323143682 and 9.25558324816643 at mu = m_Q (Sec. 6, coefficients from eqs. (88)-(93)).
  - With the conversion of Sec. 0.3 (octet: B = H^(1)(mu=m) + 2C_A ln^2 2 - C_A ln2 + C_A pi^2/12), these give exactly the corrected values above.
  - The same conversion reproduces Petrelli's 1S0[1], 1S0[8], 3P0[1] and 3P2[1] constants to <1e-7 (B.4). The octet-minus-octet difference B^[8](3P0) - B^[8](1S0) is IR-scheme independent: Petrelli 1.1956, Ozcelik/Abreu -0.4444.
- **Caveat.** Feng, Lansberg, Wang (1504.00317, Sec. 2.2) state they "reproduce [Petrelli/Maltoni et al.] results for all relevant channels" with an independent code. That was a hadronic cross-section comparison of unspecified precision. No erratum is listed on INSPIRE. **Treat this as a strongly supported but unpublished correction**; the MadLoop result should be compared with both values.

### B.4 Independent confirmations of the other constants `[VERIFIED]`

- **Abreu, Becchetti, Duhr, Ozcelik, arXiv:2211.08838** (JHEP 02 (2023) 250). Scheme: HV gamma5, OS mass/WF, alpha_s^(n_l) with decoupling. One-loop finite remainders, eqs. (91)-(93), (116), (118), (122), (127):
  - gg -> 1S0[1]: F_reg = C_F a1 + C_A a2, with a1 = pi^2/8 - 5/2 and a2 = pi^2/6 + 1/2 - ln^2 2 + i pi ln2.

        H^(1)_{gg,[1]} = 6.6100877846099362771 + 5.5 l_muR - 1.3411169166403281435 l_muF - 1.5 l_muF^2 - (1/3) n_l (l_muR - l_muF)

  - gg -> 1S0[8]: F_reg = C_F a1 + C_A(a2/2 + a3), with a3 = pi^2/48 + 3/4 + ln2/2 - ln^2 2/2.

        H^(1)_{gg,[8]} = 7.9884276758812627233 + 5.5 l_muR - 2.8411169166403281435 l_muF - 1.5 l_muF^2 - (1/3) n_l (l_muR - l_muF)

  - gamma g -> 1S0[8]: F_reg = C_F a1 + C_A a3.

        H^(1)_{gamma g,[8]} = 2.9949845170911876879 + 2.75 l_muR - 2.1705584583201640717 l_muF - 0.75 l_muF^2 - (1/6) n_l (l_muR - l_muF)

  - Converted (Sec. 0.3) these give exactly Petrelli's B^[1](1S0) = 11.9602070 and B^[8](1S0) = 11.2591053.
- **Ozcelik, arXiv:2501.04188.** One-loop bases a_1..a_6 in eqs. (88)-(93). Helicity amplitudes: A1 = chi_2 with J_z = +-2 (gluons of opposite helicity); A2 = chi_2 with J_z = 0 (Born = 0 in 4d); A3 = chi_0.
  - gg -> 3P0[1] (A3): `H = 8.16564334016549183 + 5.5 l_R - 1.34111691664 l_F - 1.5 l_F^2 - (n_l/3)(l_R - l_F)` gives B = 13.5157625 = Petrelli.
  - gg -> 3P2[1] (A1): `H = -1.28291441377047568 + (same scale terms)` gives B = 4.0672048 = Petrelli.
  - gamma g -> 3P0[8] (A3): H = 3.55054007264674324. gamma g -> 3P2[8] (A1): H = 7.23037378838500508. Scale terms as for gamma g -> 1S0[8].
  - These octet results also pass the ratio check against 2211.08838 (C_F parts identical; colour structure consistent).
- **Indirect.** Hagiwara-Sumino-Yokoya 0804.1014 eq. (13) and Czakon-Mitov 0812.0353 (exact t tbar threshold expansions) agree with Petrelli's gg 1S0[1,8] and (after the -8/9) q qbar 3S1[8] constants. Hagiwara et al. also state that Kuhn-Mirkes (hep-ph/9301204) and Petrelli agree for gg -> 1S0[1].
- **Not checked** `[UNVERIFIED]`: Maltoni's PhD thesis (Pisa 1999), the HELAC-Onia NLO 2->1 code, Ma/Wang/Chao, Butenschoen-Kniehl. None of them was found to publish 2->1 virtual constants. "ResumOnia": no INSPIRE record found under that name.

### B.5 Ready-to-compare BLHA coefficients `[DERIVED]`

Set Q = mu_R and L = ln(mu_R^2/M^2), with M = 2m. Use C_A = 3, C_F = 4/3, T_F = 1/2, and beta0 = 11 - (2/3) n_l.

**gg -> n[1]** (n = 1S0, 3P0, 3P2):

    c2 = -2 C_A
    c1 = -beta0 - 2 C_A L
    c0 = 2 B^[1]_n - C_A pi^2/3 - C_A L^2

**gg -> n[8]:**

    c2 = -2 C_A
    c1 = -beta0 - C_A - 2 C_A L
    c0 = 2 B^[8]_n - C_A pi^2/3 - C_A L - C_A L^2

**q qbar -> 3S1[8]:**

    c2 = -2 C_F
    c1 = -3 C_F - C_A - 2 C_F L
    c0 = 2 B' - C_F pi^2/3 + (beta0 - 3 C_F - C_A) L - C_F L^2,   with B' = B - (16/9) T_F

**gamma g -> n[8]** (Born ~ alpha alpha_s):

    c2 = -C_A
    c1 = -beta0/2 - C_A - C_A L
    c0 = 2 H^(1)(l = 2 ln2) - C_A pi^2/12 + (beta0/2) L + (-beta0/2 - C_A) L - (C_A/2) L^2

Numbers at mu_R = Q = M (L = 0), charmonium (n_l = 3, beta0 = 9). All gg values (and the gamma g c0) are independent of n_l at L = 0. Values are eval.:

| process | c2 | c1 | c0 |
|---|---|---|---|
| gg -> 1S0[1] | -6 | -9 | **14.0508095** |
| gg -> 3P0[1] | -6 | -9 | **17.1619206** |
| gg -> 3P2[1] (only helicity +-2 contributes) | -6 | -9 | **-1.7351949** |
| gg -> 1S0[8] | -6 | -12 | **12.6486062** |
| gg -> 3P0[8] | -6 | -12 | **11.7597173** corrected (Petrelli-printed: 15.0397510) |
| gg -> 3P2[8] | -6 | -12 | **15.1829174** corrected (Petrelli-printed: 11.2057601) |
| q qbar -> 3S1[8] | -8/3 | -7 | **8.0165435** with heavy bubble (without: 9.7943213). General: c0 = 11.3498769 - (10/9) n_l |
| gamma g -> 1S0[8] | -3 | -7.5 | **2.2464029** |
| gamma g -> 3P0[8] | -3 | -7.5 | **3.3575140** |
| gamma g -> 3P2[8] (helicity +-2) | -3 | -7.5 | **10.7171815** |

**Poles cross-check.** Catani-Dittmaier-Trocsanyi (Section E) with the onium treated as a colourless particle (singlet) or as a massive colour-octet particle of mass M (octet) reproduces all c2 and c1 above. For example, gg -> octet gives c1 = -2gamma_g - C_A. Petrelli's P-wave poles equal the S-wave ones because all P->S transition terms are proportional to Born(gg -> 3S1) = 0.

### B.6 Loop-induced 2->1 test: gg -> 3S1[8] `[VERIFIED]`

Source: Ozcelik 2501.04188, App. "3S1[8] form factor". The tree amplitude vanishes; the one-loop amplitude is finite and non-zero, with f^{abc} colour and only J_z = 0 with lambda1 = lambda2:

    F^(1)_{c4} = C_A(pi^2/8 - 5/12 - ln2/2) + C_F(1/2 - pi^2/4 + 2 ln2) - T_F n_l/6 + T_F n_h (pi^2/8 - 7/6)
    H^(2) = |A4/N|^2 = 1.7960761912769237 - 0.4467258905111629 n_l + 0.0277777777777778 n_l^2   (= 4|F^(1)_{c4}|^2)

- The OS mass counterterm inserted into the vanishing tree contributes ("F^(0,1 mass CT) = 1").
- The result agrees with Ma-Wang-Zhao (PLB 737 (2014) 103) and Beenakker-Kleiss-Lustermans (1508.07115).
- **Normalisation-free check** `[DERIVED]`: \|M(n_l)\|^2/\|M(n_l=0)\|^2 = (1 - 0.1243619 n_l)^2. For n_l = 3 this is 0.393020.
- Absolute normalisation: N = -i (4 sqrt2 pi^2/sqrt(m_Q)) sqrt2 T_F f^{abc}/2, in Petrelli's projector convention. `[UNVERIFIED in absolute terms]`

### B.7 P-wave states with a non-vanishing P->S pole at 2->1

For gg -> 1P1[8], Born(gg -> 1S0[8]) is non-zero. Its one-loop virtual therefore carries the extra P-wave poles of 2402.19221 eq. (220) (Section E.3). This is a non-trivial pole test; no finite value is published.

---

## C. 2->2 with colourless initial state (finite; pointwise usable)

### C.1 e+ e- -> eta_c gamma (single form factor, K angle independent) `[VERIFIED + eval.]`

Source: Sang & Chen, arXiv:0910.4071 (PRD 81 (2010) 034028). The amplitude is A eps_{mu nu a b} P^a k^b; LO dsigma/dx ∝ (1 + x^2), and NLO keeps the same shape [eqs. (10)-(12)]. With r = 4m^2/s and b = sqrt(1 - r), eqs. (32) and (64), C_F = 4/3 inserted:

    2Re(B^*V)/|B|^2 = (alpha_s/pi) C(r)
    C(r) = -2[30r^2 - (84+pi^2) r + 2pi^2 + 54]/(9(2-r)(1-r)) + 8(2r-3)/(3(2-r)^2) ln(2/r - 2) - (4/b) ln((1-b)/(1+b))
           + 2/(3(r-1)) [ (1 + r/2) ln^2((1-b)/(1+b)) - ln^2(2/r - 1) ] + 4/(3(1-r)) Li2( r/(2-r) )

Limits:
- r -> 0, eq. (33): C -> -(2/9)[(9 - 6 ln2) ln r + 9(3 + ln^2 2 - 3 ln2) + pi^2]
- r -> inf, eq. (35): C' -> C_F(-5 + pi^2/4), i.e. eta_c -> gamma gamma. C'(r) for r > 1 is in App. A.

Values (eval.):

| sqrt(s) | m_c | r | C (alpha_s/pi) | c0 |
|---|---|---|---|---|
| 10.58 | 1.5 | 0.0804025 | **-2.573030** | -5.146061 |
| 10.6 | 1.5 | 0.0801000 | -2.568533 | -5.137066 |
| 10.6 | 1.4 | 0.0697757 | -2.404183 | -4.808366 |

Validation:
- Sang-Chen Table I: -18.4% at alpha_s = 0.24 (m = 1.4) implies C = -2.409.
- Li-He-Chao 0910.4155 Table II (m = 1.5, alpha_s = 0.26): -12.5/59.1 implies C = -2.56.

Use a QED-only (photon) model. Z exchange adds an axial-coupling structure with a different QCD factor.

### C.2 e+ e- -> chi_cJ gamma (helicity-resolved; K angle dependent for J = 1, 2) `[VERIFIED + correction DERIVED]`

Source: Sang-Chen 0910.4071, eqs. (40), (50)-(52), App. B eqs. (67)-(69). Per chi helicity lambda_H the LO distributions are:

    F~_0(0) = (1-3r)^2 (1+x^2)/6
    F~_1(0) = 1+x^2        F~_1(+-1) = r(1-x^2)
    F~_2(0) = (1+x^2)/3    F~_2(+-1) = r(1-x^2)    F~_2(+-2) = r^2 (1+x^2)

Each helicity has a constant K = 1 + (alpha_s/pi) C_J^{|lambda|}(r). The pointwise ratio is therefore:

    R_J(x) = Sum_lambda F~_J(lambda,x) C_J^{|lambda|}(r) / Sum_lambda F~_J(lambda,x)

**Correction.** The printed C_2^{+-2}(r) has a spurious overall r^2: every term carries r^2. With the printed expression, eq. (52) gives C_2 = -8.80 at m = 1.4, whereas their own Table I gives -69.0%/0.24·pi = -9.032. Using C_2^{+-2}(true) = C_2^{+-2}(printed)/r^2 reproduces -9.0320 exactly. It also reproduces the independent Li-He-Chao NLO angular distribution of chi_c2: (2.72 + cos^2) -> (1.61 + cos^2), eval. 2.723 -> 1.613. The printed version gives 1.54.

Values at sqrt(s) = 10.58 GeV, m = 1.5 GeV, r = 0.0804025 (eval.; units alpha_s/pi):

| coefficient | value |
|---|---|
| C_0^0 | 2.192370 |
| C_1^0 | -3.393660 |
| C_1^{+-1} | -2.769328 |
| C_2^0 | -8.645933 |
| C_2^{+-1} | -10.354289 |
| C_2^{+-2} (corrected) | -9.615221 |

| x = cos theta | R(chi_c0) | R(chi_c1) | R(chi_c2) |
|---|---|---|---|
| 0 | 2.192370 | -3.307172 | -9.212407 |
| 0.5 | 2.192370 | -3.338718 | -9.046520 |
| 0.9 | 2.192370 | -3.383298 | -8.759852 |

c0 = 2R.

Integrated values: C_1 = -3.347198 and C_2 = -8.997234.

At sqrt(s) = 10.6, m = 1.4 the integrated values reproduce Sang-Chen Table I: C0 = 1.9268 (table 1.924), C1 = -3.1546 (table -3.155), C2 = -9.0320 (table -9.032).

Open discrepancy: Li-He-Chao 0910.4155 Table II implies C_0 ≈ 2.04 at m = 1.5, while Sang-Chen give 2.18 (C_1 and C_2 agree). The Sang-Chen expression is recommended because it matches its own table at m = 1.4. `[UNRESOLVED]`

### C.3 gamma* gamma -> eta_c transition form factor (e.g. e- gamma -> e- eta_c, spacelike Q^2 = -t) `[VERIFIED + eval.]`

Source: Feng, Jia, Sang, arXiv:1505.02665 (PRL 115 (2015) 222001), eqs. (4)-(6), translated from Sang-Chen (A3) (Braaten 1983 and Shifman-Vysotsky 1981 are the originals, not opened). There is a single form factor, so K is pointwise exact. With tau = Q^2/m^2:

    amplitude: 1 + C_F (alpha_s/pi) f1(tau)
    f1(tau) = pi^2 (3-tau)/(6(4+tau)) - (20+9tau)/(4(2+tau)) - tau(8+3tau)/(4(2+tau)^2) ln((4+tau)/2)
              + 3 sqrt(tau/(4+tau)) atanh(sqrt(tau/(4+tau))) + (2-tau)/(4+tau) atanh^2(sqrt(tau/(4+tau)))
              - tau/(2(4+tau)) Li2(-(2+tau)/2)
    2Re(B^*V)/|B|^2 = 2 C_F f1 (alpha_s/pi);  c0 = 4 C_F f1

Evaluations of 2C_F f1 (eval.):

| tau | 2C_F f1 |
|---|---|
| 0 | -3.376799 (= eta_c -> gamma gamma) |
| 0.5 | -3.329628 |
| 1 | -3.285350 |
| 2 | -3.204028 |
| 4 | -3.063464 |
| 10 | -2.747144 |
| 20 | -2.390878 |
| 50 | -1.769679 |
| 100 | -1.205566 |

The tau = 0 value matches the eta_c -> gamma gamma check, which validates the transcription.

### C.4 e+ e- -> J/psi eta_c (single form factor eps_{mu nu rho sigma}, K angle independent) `[VERIFIED + eval.]`

Source: Gong & Wang, arXiv:0712.4220 (PRD 77 (2008) 054028), full analytic result. With s^ = s/(4 m_c^2):

    eq. (7): M_NLO + M_LO = M_LO {1 + (alpha_s/2pi)[ (8/3) pi^2/v - beta0 ln(2m_c/mu) + K1(s^)/6 + i pi K2(s^)/6 ]}
    eq. (9): sigma_NLO = sigma0 {1 + (alpha_s/pi)[ -beta0 ln(2m_c/mu) + K1(s^)/6 ]}

- K1 and K2 are given in App. A, eqs. (11)-(12): about 30 logs and Li2's.
- Scheme: OS mass/WF, **MSbar alpha_s with n_f = 4 including the charm loop** (charm not decoupled), beta0 = 25/3.
- My implementation of K1 reproduces their Table (all six (m_c, mu) rows) to <=1e-3, e.g. K = 1.7100 / 1.9745 / 2.1125 vs 1.710 / 1.975 / 2.114.

Values (eval.):

| sqrt(s) | m_c | K1/6 |
|---|---|---|
| 10.58 | 1.5 | 11.806107 |
| 10.6 | 1.5 | 11.820910 |
| 10.6 | 1.4 | 12.373160 |

Conversion to the implementation's scheme `[DERIVED]` (n_l = 3, decoupled charm, see 0.4):

    a^(3) = K1/6 + (2/3) ln2 + beta0^(3) ln(mu/(2m_c)),   beta0^(3) = 9

At sqrt(s) = 10.58, m = 1.5, mu = 2m: a = **12.268210**, c0 = **24.53642**. In the original n_f = 4 scheme: a = 11.806107.

In MadLoop the b and t quarks are excluded from the loops (`/ t t~ b b~`), the limit m_b, m_t -> infinity of the decoupling scheme. Masses of 10 and 100 TeV instead leave a remainder of order 1e-9 and, through the cancellation of their counterterms against their loop groups, a rounding floor of about 1e-8 in quadruple precision (`research/pwave_fd/README.md`).

Consistency:
- Zhang-Gao-Chao hep-ph/0506076: K = 15.7/8.0 = 1.96 at m = 1.5, mu = 2m, alpha_s = 0.259, sqrt(s) = 10.6 (n_f = 4, same scheme).
- Dong-Feng-Jia 1204.4128 (Sec. "Asymptotic expressions of NLO short-distance coefficients", n_f = 4, HV gamma5) give only the asymptotic (s >> m^2) form. It is about 5% off at B-factory energies, consistent with power corrections.

NNLO: Feng-Jia-Mo-Sang 1901.08447.

### C.5 Others (no pointwise values)

- **e+ e- -> J/psi J/psi** (QED Born via two photons; many form factors, angle dependent). Gong-Wang arXiv:0801.0648 (PRL 100 (2008) 181803), eq. (6) and the text below it: **integrated** K(s^ = 12.484) = **-11.190** (alpha_s/pi), m = 1.5 GeV, sqrt(s) = 10.6. It is mu-independent (no gluon at LO); the differential K is only plotted.
- **gamma gamma -> J/psi J/psi.** Yang, Chen, Qiao arXiv:2006.05351: only WW/LBS-convoluted cross sections, with different |R(0)|^2 at LO/NLO and pT cuts. Not usable pointwise.
- **e+ e- -> J/psi chi_cJ.** See C.6.

### C.6 e+ e- -> J/psi chi_cJ, J = 0, 1, 2 (an S-wave and a P-wave state; integrated and per helicity group) `[VERIFIED + eval.]`

Sources:
- **Wang, Ma, Chao, arXiv:1107.2646** (PRD 84 (2011) 034022), appendix: the full analytic NLO cross sections. Only their real parts are claimed.

      sigma_NLO = sigma_LO (1 + alpha_s Re M_J^NLO / (72 a^3 b d^3 e pi M_J^LO)),   a = sqrt(s)/(4 m_c)

  `M_J^LO`, `M_J^NLO` and the dilogarithm combinations `l_1 ... l_7` are in the appendix.
- **Dong, Feng, Jia, arXiv:1107.4351 v3** (JHEP 10 (2011) 141): the helicity decomposition, eq. (2.6), and the LO helicity amplitudes, eq. (3.3). Table 2 has the single-helicity LO and NLO cross sections at sqrt(s) = 10.58, m_c = 1.5, mu = 2m_c, alpha_s = 0.259. Eqs. (4.5)-(4.7) give the asymptotic forms only.
- **Zhang, Ma, Chao, 0802.3655** (chi_c0 only): K = 2.8. Both later papers find K = 1.57 with the same inputs; Wang et al. note a numerical error in it. Not used.

Scheme of both: on-shell charm mass and wave function, **MSbar alpha_s with n_f = 4 (charm active, beta_0 = 25/3)**, photon exchange only, no QED fragmentation. The Coulomb terms are absorbed into the wave functions (Wang et al.) or absent through the threshold expansion (Dong et al.). The virtual is **UV and IR finite** for every helicity amplitude, since the S-wave J/psi removes the non-factorisable soft term. The exclusive final state has no real emission, so the NLO correction is the virtual alone. Born and virtual have the angular form alpha + beta cos^2(theta): lambda = lambda_psi - lambda_chi = 0 gives sin^2(theta), and |lambda| = 1 gives (1 + cos^2(theta))/2 per amplitude.

Implementation (`research/pwave_fd/psi_chic_ref.py`): the appendix expressions of M_J^NLO converted mechanically (sympy) from the arXiv LaTeX source, and evaluated with mpmath (real parts of Li_2).

Checks of the transcription:
- K factors: 1.573 / 0.911 / 0.784 at mu = 2m_c and 1.785 / 1.246 / 1.142 at mu = sqrt(s)/2 (sqrt(s) = 10.6), against Wang et al.'s table 1: 1.57 / 0.91 / 0.78 and 1.79 / 1.25 / 1.14.
- At 10.58: 1.573 / 0.910 / 0.784 against Dong et al.'s table 2 (11.43/7.26, 1.11/1.22, 1.38/1.76).
- The LO ratios M_1^LO/M_0^LO = 0.1681467735 and M_2^LO/M_0^LO = 0.2421104202 equal those of Dong et al.'s eq. (3.3) to all digits.

Conversion to the implementation's scheme `[DERIVED]` (n_l = 3, heavy quarks decoupled; 0.4; Born of order alpha_s^2):

    c0 = 2 (A_J + (2 T_F/3) ln(mu^2/m_c^2)),   sigma_NLO = sigma_LO (1 + (alpha_s/pi) A_J)  (n_f = 4)

c0 is the cross-section average of MadLoop's finite part, int B c0 dcos / int B dcos. In MadLoop the b and t quarks are excluded from the loops (`/ t t~ b b~`), which is the limit m_b, m_t -> infinity of the decoupling scheme; heavy masses (10, 100 TeV) instead leave a rounding floor of 1e-7 (`research/pwave_fd/README.md`). Values (eval.) at sqrt(s) = 10.58, m_c = 1.5, mu = 2m_c:

| J | A_J (n_f = 4) | c0 (n_l = 3, decoupled) |
|---|---|---|
| 0 | 6.9466468960 | **14.8174900328** |
| 1 | -1.0897154170 | **-1.2552345934** |
| 2 | -2.6173891669 | **-4.3105820930** |

The LO shares of lambda = 0, from Dong et al.'s eq. (3.3), are 0.2301389 (chi_c0), 0.0820391 (chi_c1) and 0.4269518 (chi_c2). Table 2 gives, per group lambda = 0 / |lambda| = 1, A (n_f = 4) = 6.03 / 7.19 (chi_c0), -7.3 / -0.60 (chi_c1) and -4.33 / -1.43 (chi_c2), to the two or three digits of the table.

---

## D. Hadroproduction 2->2 (status only)

| process | references | pointwise one-loop values published? |
|---|---|---|
| g g -> J/psi g (3S1[1]) | Campbell, Maltoni, Tramontano hep-ph/0703113 (PRL 98 (2007) 252002): virtual in the **four-dimensional helicity scheme**; Artoisenet, Campbell, Lansberg, Maltoni, Tramontano 0806.3282 (Upsilon); Gong & Wang 0802.3727 (PRL 100 (2008) 232001) | No (checked LaTeX sources) |
| J/psi + Q Qbar | Artoisenet, Lansberg, Maltoni hep-ph/0703129 (PLB 653 (2007) 60) | No |
| Colour-octet 2->2 (1S0[8], 3S1[8], 3PJ[8] + jet) | Butenschoen & Kniehl 1009.5662, 1105.0820; Ma, Wang, Chao 1009.3655, 1012.1030; chi_cJ: Ma-Wang-Chao 1002.3987 | No |
| q qbar -> c cbar[n] g, 4 S-wave + 8 P-wave states | A H, Shao, Simon 2402.19221 Sec. 5: IR poles of the analytic one-loop matrix elements checked against App. B formulas | Poles only, no finite values |

**Recommendation.** For 2->2 hadroproduction, rely on:
- the universal pole checks of Section E;
- crossing and limit checks: soft and collinear limits of the real emission (2402.19221 Sec. 3), and the gg -> 3S1[8] loop-induced amplitude;
- internal consistency: gauge invariance, mu_R dependence c0(mu) - c0(mu0) predicted by c1 and beta0.

---

## E. Universal IR pole structure (formulas for code)

### E.1 Catani-Dittmaier-Trocsanyi, hep-ph/0011222 (PLB 500 (2001) 149)

Mass-renormalised, charge-unrenormalised amplitudes are normalised as `M = (g_s mu^eps/4pi)^q [M^(0) + (g_s/4pi)^2 M^(1)]` [eq. (7)]. Then `|M^(1)> = I_m |M^(0)> + finite` [eq. (8)], with [eq. (9)]:

    I_m = (4pi)^eps/Gamma(1-eps) { q/2 (beta0/eps - beta~0)
          + Sum_{j != k} T_j.T_k (mu^2/|s_jk|)^eps [ V_jk(s_jk; m_j, m_k; eps) + (1/v_jk)(i pi/eps - pi^2/2) Theta(s_jk) ]
          - Sum_j Gamma_j }

- v_jk = sqrt(1 - m_j^2 m_k^2/(p_j.p_k)^2) [eq. (12)].
- V for two massive partons [eq. (13)]: `V = (1/(2eps))(1/v_jk) ln((1-v_jk)/(1+v_jk)) - (1/4)(ln^2(m_j^2/|s_jk|) + ln^2(m_k^2/|s_jk|)) - pi^2/6`
- V for one massive, one massless [eq. (14)]: `V = 1/(2eps^2) + (1/(2eps)) ln(m_j^2/|s_jk|) - (1/4) ln^2(m_j^2/|s_jk|) - pi^2/12`
- V for two massless [eq. (15)]: `V = 1/eps^2`
- Gamma_g [eq. (16)]: `gamma_g/eps - gamma~_g - (2/3) T_R Sum_F ln(m_F^2/mu^2)`
- Gamma_q, massless [eq. (17)]: `gamma_q/eps - gamma~_q`
- Gamma_Q, massive [eq. (18)]: `C_F [1/eps + (1/2) ln(m_Q^2/mu^2) - 2]`
- gamma_q = 3/2 C_F and gamma_g = 11/6 C_A - 2/3 T_R N_f, with N_f massless [eq. (19)]. gamma~^CDR = 0 [eq. (20)].
- All momenta are outgoing. Theta(s_jk) = 1 when both partons are incoming or both outgoing.

In BLHA units (alpha_s/2pi): `2Re<M0|M1>(alpha_s/4pi) = (alpha_s/2pi) Re<M0|I|M0>`. After MSbar renormalisation in the decoupling scheme, the q/2 beta0 term is removed. The heavy-flavour logs (q/2)(4/3)T_R ln(mu^2/m^2) and -(2/3)T_R ln(m^2/mu^2) per external gluon are finite, so they do not affect c2 or c1. The `iπ/eps` term drops out of the real part. The `-pi^2/2` (time-like pairs) term enters only the finite part.

**Quarkonium:**
- An S-wave colour singlet is a colourless particle.
- An S-wave colour octet is a massive particle of mass M = m_Q + m_Qbar with T^2 = C_A. Its Gamma_I contributes C_A/eps.

This reproduces all Petrelli, Abreu and Ozcelik poles in Section B; for example gg -> octet gives c1 = -2gamma_g - C_A at Q = M.

### E.2 Same structure in FKS/MadGraph notation

Source: A H, Shao, Simon, arXiv:2402.19221 (JHEP 07 (2024) 050), App. B, eqs. (215)-(217):

    M^(n-1,1) = (alpha_s/2pi) (4pi)^eps/Gamma(1-eps) (mu^2/Q_ES^2)^eps V

**S-wave colour singlet** [eq. (216)], equivalent to Frederix et al. 0908.4272 eq. (B.2):

    V = -( (1/eps^2) Sum_{k massless} C_k + (1/eps) Sum_{k massless} gamma_k + (1/eps) Sum_{k massive} C_k ) M
        + (1/eps) Sum_{k massless} Sum_{l>k} ln(2 k_k.k_l/Q^2) M_kl
        + (1/(2eps)) Sum_{k<l massive} (1/v_kl) ln((1+v_kl)/(1-v_kl)) M_kl
        - (1/(2eps)) Sum_{k massive} ln(m_k^2/Q^2) Sum_{l massless} M_kl + V_FIN

Here `M_kl = -(2 - delta_kl)/(2s omega omega) Sum M T_k.T_l M^*`, i.e. M_kl = -2<T_k.T_l> M for k != l [eq. (MlinkedOnium2)]. In the gluon averaging omega(g) = 2(1-eps)(N_c^2-1), keep eps = 0 numerically.

**S-wave colour octet** [eq. (217)]: the same, with the onium included among the massive coloured particles (C = C_A, mass m_k = M, colour operator in the adjoint).

### E.3 P-wave extra poles (2402.19221 App. B) `[VERIFIED transcription]`

Notation:
- r1, r2 are the reduced Born processes with the P-wave onium replaced by an S-wave state of the same spin S:
  - colour singlet: r1 = [8]
  - colour octet: r1 = [8], r2 = [1]
- K is the onium momentum.
- `v_k = sqrt(1 - k_k^2 K^2/(k_k.K)^2)` [eq. (219)].
- Effective colour Casimirs:

      C_eff[18] = (2 + mQ/mQbar + mQbar/mQ)^2/(2N_c)
      C_eff[81] = (2 + mQ/mQbar + mQbar/mQ)^2 C_F
      C_eff[88] = (2 + ...)(-2/N_c + (N_c^2-2)/(2N_c)(mQbar/mQ + mQ/mQbar))

- Correlated Borns: `M_{k[C1C2]}(r1,r2)^mu = -(1/2s)(1/omega omega) Sum 2Re{ eps^{*mu}_{lambda_l}(K) M(r1)^* T_k.T_eff[C1C2] M(r2) }`. The J-projected analogues `M_{J,k[C1C2]}^mu` and `M_J(r)^{mu nu}` contract the orbital polarisation with Clebsch-Gordan weights [eqs. (MlinkedOnium3), (MlinkedOnium4J), (MtreenpoOniumJ)].

Extra terms relative to E.2:

**1P1[1]** [eq. (218)]:

    + (1/eps) Sum_{k massless} (k_{k,mu}/(K.k_k)) M_{k[18]}(r, r1)^mu
    + (1/(2eps)) Sum_{k massive} (k_{k,mu}/(K.k_k)) (m_k^2 K^2/(v_k^3 (K.k_k)^2)) [2v_k/(1-v_k^2) - ln((1+v_k)/(1-v_k))] M_{k[18]}^mu
    - (1/eps) ( (1/(2N_c)) 8/(mQ mQbar) - (2/K^2) C_eff[18] ) M(r1)

**1P1[8]** [eq. (220)]: the same structure, with [18] replaced by the sum of [88] (with r1) and [81] (with r2), and the last term replaced by:

    -(1/eps)(B_F 8/(mQ mQbar) - (2/K^2) C_eff[88]) M(r1) - (1/eps)(C_F 8/(mQ mQbar) - (2/K^2) C_eff[81]) M(r2),   B_F = (N_c^2-4)/(4N_c)

**3PJ[1]** [eq. (221)]: the k-terms use M_{J,k[18]}^mu, and the last term is:

    -(1/eps)(2/(3K^2)) g_{mu nu} C_eff[18] M_J(r1)^{mu nu} - (1/eps)((2J+1)/(2N_c)) 8/(9 mQ mQbar) M(r1)

**3PJ[8]** [eq. (222)]: analogous with [88]/[81] and B_F, C_F weights `8(2J+1)/(9 mQ mQbar)`.

Observation `[DERIVED]`: for equal masses (K^2 = 4m^2) the self terms in 1P1[1,8] cancel identically. Check: B_F 8/m^2 = (2/K^2) C_eff[88] = 2(N^2-4)/(N m^2), and C_F 8/m^2 = (2/K^2) C_eff[81]. Only the interference terms with external coloured partons survive. Consequences for 2->1:
- gg -> 3PJ[1,8]: no extra pole, because Born(gg -> 3S1[1,8]) = 0.
- gg -> 1P1[8]: a genuine extra 1/eps pole proportional to the interference with gg -> 1S0[8] and 1S0[1] Borns may appear.

The paper verified these formulas analytically for q qbar -> c cbar[n] g (12 Fock states; Sec. 5).

### E.4 Coulomb

`C pi^2/(2v)` with C = C_F (singlet) or C_F - C_A/2 (octet), Petrelli eq. (134)-(139). It is absent (zero) in dimensional regularisation at v = 0.

---

## F. Summary table

| check | formula / value | conventions | source |
|---|---|---|---|
| eta_c <-> gamma gamma | a = C_F(pi^2/4 - 5) = -3.37679853; c0 = -6.75359706 | finite; any scheme (HV = NDR) | 1104.1418 Tab. I; 2211.08838 eq. (91); 0910.4071 eq. (35) |
| J/psi <-> l+l- (gamma*, Z-vector) | a = -16/3; c0 = -32/3 | finite | 0712.4220 eq. (10); 0910.4071 Sec. V |
| chi_c0 <-> gamma gamma | a = pi^2/3 - 28/9 = 0.17875702; c0 = 0.35751405 | finite | 1511.06288 eq. (7); 2501.04188 eq. (90) |
| chi_c2 <-> gamma gamma, hel +-2 | a = -16/3; c0 = -32/3 | finite | 1511.06288 eq. (8b) |
| chi_c2 -> gamma gamma, hel 0 | Born 0; \|A1/A0(hel 2)\| = 0.2951464 alpha_s/pi | loop-induced helicity | 1511.06288 eq. (8a) |
| chi_c1 -> gamma gamma | 0 | Landau-Yang | -- |
| J/psi -> 3 gamma | integrated: 1 - 12.630 alpha_s/pi | integrated only | 1210.6337 eq. (1) |
| B_c -> l nu (axial) | a = 2[((mb-mc)/(mb+mc)) ln(mb/mc) - 2]; mb = 4.75, mc = 1.5: -2.801213 | finite; massive lepton | hep-ph/9501296 |
| B_c^* -> l nu (vector) | a = 2C_F[(3/4)((r-1)/(r+1)) ln r - 2]; -4.134547 | finite | 2210.02979 eq. (9) |
| gg -> 1S0[1] | c2 = -6, c1 = -beta0, c0 = 2B - pi^2 = 14.0508095 (L = 0); B = C_F(-5 + pi^2/4) + C_A(1 + 5pi^2/12) | Q = mu_R = M; nl-MSbar decoupled; HV | hep-ph/9707223 eqs. (124)-(125); confirmed 2211.08838 |
| gg -> 1S0[8] | c1 = -beta0 - 3; c0 = 12.6486062; B = C_F(-5 + pi^2/4) + C_A(2 + 7pi^2/24) | as above | eqs. (128)-(129); confirmed 2211.08838 |
| gg -> 3P0[1] | c1 = -beta0; c0 = 17.1619206; B = C_F(-7/3 + pi^2/4) + C_A(1/3 + 5pi^2/12) | as above | eq. (126); confirmed 2501.04188 |
| gg -> 3P2[1] | c1 = -beta0; c0 = -1.7351949; B = -4C_F + C_A(1/3 + 5/3 ln2 + pi^2/6) | only helicity +-2 | eq. (127); confirmed 2501.04188 |
| gg -> 3P0[8] | c1 = -beta0 - 3; c0 = 11.7597173 (corrected) vs 15.0397510 (Petrelli printed) | diagram-h colour factor C_A/2 | eq. (130) + this report; 2501.04188 |
| gg -> 3P2[8] | c1 = -beta0 - 3; c0 = 15.1829174 (corrected) vs 11.2057601 (printed) | as above | eq. (131) + this report; 2501.04188 |
| q qbar -> 3S1[8] | c2 = -8/3, c1 = -7, c0 = 2(B - 8/9) - 4pi^2/9 = 11.3498769 - (10/9) n_l (8.0165435 at n_l = 3) | heavy loop included (decoupling) | eqs. (132)-(133) + 0804.1014 erratum; 0812.0353 |
| gamma g -> 1S0/3P0/3P2(hel 2) [8] | c2 = -3, c1 = -beta0/2 - 3, c0 = 2.2464029 / 3.3575140 / 10.7171815 | Q = mu = M | 2211.08838 eq. (122); 2501.04188 Sec. 6 |
| gg -> 3S1[8] (loop-induced) | F^(1) = C_A(pi^2/8 - 5/12 - ln2/2) + C_F(1/2 - pi^2/4 + 2ln2) - T_F n_l/6 + T_F n_h(pi^2/8 - 7/6); \|M\|^2 ∝ (1 - 0.1243619 n_l)^2 | J_z = 0, lambda1 = lambda2 only | 2501.04188 App. |
| e+e- -> eta_c gamma | a = C(r); sqrt(s) = 10.58, m = 1.5: -2.573030 (c0 = -5.146061) | photon only; angle independent | 0910.4071 eqs. (32), (64) |
| e+e- -> chi_cJ gamma | R_J(x) from C_J^lambda; 10.58/1.5: chi_c0 2.192370; chi_c1(x = 0) -3.307172; chi_c2(x = 0) -9.212407 | per helicity; C_2^{+-2} printed/r^2 | 0910.4071 eqs. (50), (67)-(69) |
| e gamma -> e eta_c (gamma* gamma TFF) | a = 2C_F f1(Q^2/m^2); tau = 4: -3.063464; tau = 10: -2.747144 | spacelike, finite | 1505.02665 eqs. (4)-(6) |
| e+e- -> J/psi eta_c | a = K1(s^)/6 + (2/3)ln2 + 9 ln(mu/2m) (n_l = 3, decoupled); 10.58/1.5, mu = 2m: 12.268210 (c0 = 24.53642); n_f = 4 MSbar: 11.806107 | angle independent | 0712.4220 eqs. (9), (11) |
| e+e- -> J/psi J/psi | integrated K = -11.190 at s^ = 12.484 | integrated only | 0801.0648 eq. (6) + text |
| IR poles (any process) | CDT I-operator; onium octet = massive adjoint; P-wave extra terms | HV = CDR | hep-ph/0011222 eqs. (8)-(20); 2402.19221 eqs. (215)-(222) |

---

## G. Items flagged as not verified or open

1. **Petrelli octet P-wave constants B^[8](3P0), B^[8](3P2).** I infer a colour-factor error, supported by Ozcelik 2025 and colour algebra. There is no published erratum, and Feng-Lansberg-Wang (2015) claim cross-section-level agreement with Petrelli.
2. **gg -> 1P1[8] 2->1 virtual.** Not published; HELAC-Onia internal only.
3. **Li-He-Chao chi_c0 gamma K-factor** (~2.04) vs Sang-Chen (2.18) at m = 1.5. Unresolved.
4. **Absolute normalisation of the loop-induced gg -> 3S1[8] amplitude.** Only ratio checks are given.
5. **Not opened:** Mackenzie-Lepage 1981, Braaten 1983 / Shifman-Vysotsky 1981, Kuhn-Mirkes 1993 (only via Hagiwara et al.), Maltoni thesis, "ResumOnia" (no INSPIRE record found under that name).
6. **Equation numbers** are arXiv-version numbers (ar5iv); journal numbering may differ.
