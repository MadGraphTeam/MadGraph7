# One-loop virtuals of onium decays: H -> l+ l- a against He and Huang

He and Huang, arXiv:2608.23439, give NLO QCD widths of eta_c, chi_c0,1,2 (and eta_b, chi_bJ) -> l+ l- a for l = e, mu, with photon-energy cuts of 100, 300 and 500 MeV. Their Tables 1 and 2 list NLO (LO) widths in eV. At this order the NLO correction is the virtual alone: a colour singlet cannot emit a single real gluon. Here MadLoop's virtual of the decays (`[virt=QCD]` with the bound state in the initial state, `docs/onia_loop.md`, "Decays") is integrated over the three-body phase space and compared with their charmonium columns. The script is `hh.py`; the results are in `data/`.

## Setup

These are their inputs:
- alpha = 1/137;
- m_Q = m_H/2, with the PDG hadron masses: 2.984 GeV (eta_c); 3.415, 3.511, 3.556 GeV (chi_c0,1,2);
- |R_S(0)|^2 = 0.810 GeV^3 and |R'_P(0)|^2 = 0.075 GeV^5;
- m_e = 0.511 MeV and m_mu = 0.106 GeV;
- mu_R = m_H;
- on-shell mass and field renormalisation;
- photon exchange only, generated as `... > l+ l- a / z h`.

The decay LDME (`Block ldme_decay`) is 2 Nc |R(0)|^2/(4 pi) for the S wave and 2 Nc 3 |R'(0)|^2/(4 pi) for the P waves.

The integral is Gamma = 1/(2M) int |M|^2 dPhi_3, written as an integral over the lepton-pair mass m12 and the lepton angle in the pair rest frame:
- dPhi_3 = p* |p_a|/(64 pi^3 M m12) dm12^2 dcos(theta*);
- Gauss-Legendre in u = ln(m12^2 - 4 m_l^2), 24 nodes on each of 4 panels, with the cut boundaries as panel edges;
- 4 nodes in cos(theta*), exact for |M|^2, which is quadratic in cos(theta*) at fixed m12.

There are 384 points per channel, and every one is stable in double precision (code 213). The poles vanish, at most 2e-7 of the Born.

The virtual is linear in alpha_s, so NLO = LO + alpha_s C, with C = int Born * finite/(2 pi).

The points are evaluated in a boosted frame (velocity 0.6 along (1,2,3)/sqrt(14)). For a P-wave state exactly at rest, TIRust (pin 6ba4b53) gives wrong DP integrals for 1 -> 3 decays, and the QP rescue refuses the threshold Coulomb boxes of incoming constituents. MadLoop reports those points unstable (420); see `references/tirust_traces/onium_decay`. The S-wave results are the same in both frames to every digit.

## Results

**The cut.** With E_gamma > E_cut in the rest frame of H (m12^2 < m_H^2 - 2 m_H E_cut, their text), the 100 MeV entries agree. Their 300 and 500 MeV entries are higher, already at LO: up to 7% for chi_c1 (`data/hh_summary_egamma.json`). With m12 < m_H - E_cut (`HH_CUT_MODE=mH-m12`), all 24 LO entries agree to their printed digits (`data/hh_summary_mhcut.json`). An example:

| | MadLoop, m12 < m_H - E_cut | He-Huang |
|---|---|---|
| eta_c > mu+ mu- a, LO, 100 / 300 / 500 MeV (eV) | 71.426 / 71.094 / 70.403 | 71.4 / 71.1 / 70.4 |

The MadLoop LO of eta_c also equals their eq. (14a), integrated on the same grid.

**NLO.** With one alpha_s per state, all their NLO entries are reproduced within their rounding (`data/hh_alpha_fit.json`). The largest residual is 1.65 half-units of their last digit.

| State | Fitted alpha_s | One-loop running from alpha_s(m_Z) = 0.118, 4 flavours below m_b = 4.7 |
|---|---|---|
| eta_c | 0.2347 | 0.2351 |
| chi_c2 | 0.2230 | 0.2229 |
| chi_c0 | 0.2245 | 0.2256 |
| chi_c1 | 0.2265 | 0.2237 |

The chi_c0 and chi_c1 corrections are small, so their alpha_s is loosely determined.

**NLO coefficients K, with NLO = LO (1 + K alpha_s/pi), at 100 / 300 / 500 MeV:**

| | e | mu |
|---|---|---|
| eta_c | -3.392 / -3.391 / -3.390 | -3.430 / -3.428 / -3.425 |
| chi_c0 | 0.335 / 0.210 / 0.185 | 0.597 / 0.268 / 0.198 |
| chi_c1 | 1.382 / -0.339 / -0.881 | 1.386 / -0.336 / -0.879 |
| chi_c2 | -4.185 / -4.842 / -5.035 | -2.954 / -4.192 / -4.592 |

For comparison, eta_c > a a has K = -(20-pi^2)/3 = -3.377.

## Running it

```text
export HH_WORK=<a scratch directory>
python3 -I hh.py generate etac e                    # output and driver
python3 -I hh.py run etac e 24                      # E_gamma > E_cut
HH_CUT_MODE=mH-m12 python3 -I hh.py run etac e 24   # m12 < m_H - E_cut
```

`generate` uses `loop_sm_onia-full` and sets the masses, alpha and the LDME in the cards. A P-wave output takes a few minutes to build. A run takes about 10 s per channel.
