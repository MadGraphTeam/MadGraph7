# TIRust QP refusal of S-wave quarkonium (Coulomb, zero-Gram) lanes

`trace_gg_1S08_qp.bin`: the TIRust bridge calls (`tools/tirust_capture/shim.c`
format, kind-5 records with the exact ASCII binary128 inputs) of MadLoop's QP
evaluation (`CTModeRun=4`) of g g > etac(1|1S08) [virt=QCD] (loop_sm_onia-c_mass)
at one point, with the onia QP improvement of the onia_loop branch. Three lanes
are refused (n=3 rank 2, n=4 rank 3 twice): "TIRust QP (256 bits):
Unstable(ill-conditioned Gram matrix ...)", also with the origin shift set to
exactly zero; the same lanes succeed through evaluate_dp. Replay:

    python3 replay_mp.py trace_gg_1S08_qp.bin

TIRust pin 1c130a81 (vendor/tirust/Cargo.toml), HEPTools/tirust of this checkout.

# TIRust DP refusals with two S-wave bound states

`trace_ee_psi_etac_dp.bin`: TIRust DP calls (`CTModeRun=1`) of
e+ e- > Jpsi(1|3S11) etac(1|1S01) / z QED=2 [virt=QCD] at sqrt(s) = 10.58,
m_c = 1.5 (one point). 8 of 116 lanes are refused:
- boxes (n=4, rank 3) "singular Cayley matrix for [2, 3]" / "[0, 3]": a pinched
  bubble of two heavy-quark propagators carrying the bound-state momentum P,
  P^2 = 4 m^2 (a bubble at its threshold);
- pentagons (n=5, rank 3) "singular Gram matrix for [0, 1, 2]; unsupported
  reduction: direct engine: linearly dependent propagator sets raising
  different propagators" (two Coulomb relations, one per bound state) and
  "singular Cayley matrix for [0, 1, 2]".

    python3 replay_dp.py trace_ee_psi_etac_dp.bin
