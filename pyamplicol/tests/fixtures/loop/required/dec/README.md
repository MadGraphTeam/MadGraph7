<!-- SPDX-License-Identifier: 0BSD -->
# Decay references (M-FINAL MF-DEC-1)

- `tree_<row>.json`: MG5_aMC 3.7.0 `output standalone` tree values (binary64
  `SMATRIX`) of six decays on loop_sm at the scheme's param_card
  (`tools/loop_reference/decay_tree_mg5.py`).
- `<run>.json`: MadLoop (MG5_aMC 3.7.0) one-loop references of rows XX01-XX08
  (`tools/loop_reference/dec_campaign.json`, `mqcd_campaign.py --campaign`):
  every point in MadLoop's quadruple-precision stage, every squared split order
  (`rows[].sectors`), three exact rest-frame points (`decay_points.py`, printed
  to 32 digits; the exact rationals in `point_construction.exact_points`) plus
  a copy of point 1 at mu_R = M/2. Cards: `providers.madloop.settings.param_card`.

MadLoop's conventions for the decaying leg (read off the exports and the
NLOCT UFOs; every scheme has the decaying particle's own width at zero):

- Averaging: Born and virtual are summed over the final helicities and colours
  and divided by the decaying particle's helicity count times its colour
  dimension (6 for the top, 3 for W and Z, 1 for the Higgs) and the final
  identical-particle factor (`normalization`). XX01's Born equals the MG5
  standalone tree `tree_t_bwp.json` bitwise.
- External self-energy: loop diagrams are amputated; no self-energy is
  inserted on the decaying leg, so its absorptive part enters only through
  the field renormalisation below.
- External wave-function renormalisation: the UV counterterm vertices of the
  UFO carry the decaying leg's field constant (half of it), e.g.
  `c_UVepemZ1EW` of `z > e- e+` holds `ZZWcft_UV_EW` (delta Z_ZZ) and the
  mixing `AZWcft_UV_EW` (delta Z_AZ); the top's EW constants
  `tWcft_UV_EW_L/R`, its QCD one `UVWfct_t_0`. Their finite parts are written
  as `recms(CMSParam == 1 and W != 0, x)`, i.e. `Re x` at zero width: the
  real on-shell delta Z = -Re Sigma'(M^2), the absorptive part of the
  decaying particle's self-energy at its mass dropped.
- Points: MadLoop's quadruple-precision point improvement for a decay is its
  PSMC method (ORIG needs two massless beams); the binary64 input is the
  32-digit point.
