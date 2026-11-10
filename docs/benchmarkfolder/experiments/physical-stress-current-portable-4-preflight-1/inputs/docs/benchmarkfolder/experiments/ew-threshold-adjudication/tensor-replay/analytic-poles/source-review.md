Independent source review passed for the specialized QED pole formula in
`audit.py`. The review was performed by `ct_qp_source_audit`, with a separate
flavor, photon-tag and counterterm review by `geometry_probe_audit`. No reducer,
compiler, Born evaluation or numerical replay was invoked for this review.
The reviewed `audit.py` SHA-256 is
`f9f8bdae70cccf9c0601cf77d21538d72717561332b59cebd3e161fcf68be8c8`.
This is a source-consistency review, not a complete-amplitude acceptance result.

All paths below are relative to the repository root. `EXPORT` denotes the
historical generated process at
`.tirust-build/physical-stress/unchecked/ew_n3_epem_ttxa/candidate-9-qp-init-dp`.

- `Template/NLO/SubProcesses/fks_singular.f:7752–7753,7788–7805` uses
  `real(GAL(1))**2/(8*pi**2)` and the QED Born for the pole normalization.
  The two massless incoming leptons give double-pole factor `-2` and
  single-pole diagonal `-3`. The massive top pair contributes `-8/9` to the
  single-pole diagonal. An untagged external photon contributes `-gamma_ph`.
- The same file at `8022–8034` gives `gamma_ph=-40/9` for five massless quarks
  and three massless charged leptons. These are the actual generated constants
  in `EXPORT/Source/MODEL/coupl.inc:19–22`, also checked against the physical
  card. `madgraph/iolibs/export_v4.py:11206–11213` obtains them from the model;
  the run-card jet-flavor setting is not their source. The combined diagonal
  factor is therefore `5/9`.
- `madgraph/iolibs/template_files/sborn_sf_fks.inc:43–47` multiplies the physical
  charges and reverses their product for an initial/final pair. Equivalently,
  the all-outgoing charges in external-leg order are
  `(-1,+1,+2/3,-2/3,0)`. `fks_singular.f:7835–7875` sums each distinct pair once
  with factor `-2`, including the massless, mixed-mass and massive soft factors
  implemented by `audit.py`. Its massive-pair branch and small-velocity series
  at `7855–7861` match the audit, including the `v <= 1e-6` branch boundary.
- `EXPORT/Cards/proc_card_mg5.dat:66` specifies
  `generate e+ e- > t t~ a [virt=QED]`. No photon is tagged; the parser explicitly
  rejects tagged particles in `virt` mode at
  `madgraph/interface/madgraph_interface.py:6912–6916`. The audit extracts the
  electron, top and photon charges from the exact imported UFO. A full-NLO
  tagged-photon process has a different prescription:
  `Template/NLO/SubProcesses/BinothLHA.f:280–299` adds a Gmu/alpha(0) conversion
  to the single pole and finite term when tagged photons are present. That
  conversion must not be appended to this untagged specialization.
- `fks_singular.f:7670–7674,7882` explains the FKS prefactor convention.
  Choosing `QES2=mu2` makes the explicit `double*log(mu2/QES2)` term vanish;
  either value of `fksprefact` then gives the same pole prediction. This choice
  does not claim that an actual FKS caller selected `fksprefact=.false.`.

The formula applies to the UV-renormalized full interference. It is not a
prediction for individual bare tensors or for an amplitude missing loop groups.
Both bare UV and IR poles are retained: the frozen trace card has
`LoopPoleMode=0` at
`docs/benchmarkfolder/experiments/ew-threshold-adjudication/MadLoopParams.dat:25–26`,
and `EXPORT/Source/MODEL/intparam_definition.inc:17–19` sets both `EPSUV` and
`EPSIR` to one.
The generated DP path includes counterterm interference in all three Laurent
slots at `EXPORT/SubProcesses/P0_epem_ttxa/loop_matrix.f:1053–1081`; the QP path
does so in `mp_compute_loop_coefs.f:308–317,348–369` in that same directory.
The model top-mass single-pole counterterm at
`EXPORT/Source/MODEL/intparam_definition.inc:291–296` is proportional to
`EPSUV`. No separate mass-renormalization pole should be added to the universal
FKS prediction after UV cancellation; its massive external-leg and pair terms
are already included. This review does not independently prove that every
generated counterterm or every numerical cancellation is correct.
The generated charge counterterms already include
`ECOUP_A0_UV_EW_1EPS_ + ECOUP_DGMUA0_UV_EW_1EPS_` in
`EXPORT/Source/MODEL/couplings3.f:48–49,55–56,61–62`. This independently supports
omitting an additional Gmu/alpha(0) conversion in this process.

Additional source identities used for the counterterm review are:

| Source relative to `EXPORT` | SHA-256 |
|---|---|
| `Source/MODEL/intparam_definition.inc` | `70ee1e7c51ade2f8e5160eedc61f6e0087879c4cd18810c564cccdcb62ebdbb0` |
| `Source/MODEL/couplings3.f` | `41c821b6e04a8a986008255ab076a4b2d4ea6f3a1ead26f6b0885054ca09014e` |
| `SubProcesses/P0_epem_ttxa/mp_compute_loop_coefs.f` | `7518213871f13f57fa2b666b8bf3b4e488392373e1bf74ecbafb0adad8a3bd8d` |

The precision limit is explicit in the generated code. `loop_matrix.f:793–809`
computes the returned `REAL*8` Born from the original `P_USER` and leaves it
unchanged. Only later, at `977–979`, does it call `SET_MP_PS`; that routine at
`2130–2157` promotes and repairs the momenta. The QP loop coefficients then use
the repaired `MP_P` at `1014`. The trace routine
`docs/benchmarkfolder/experiments/ew-threshold-adjudication/trace.f:74,92`
prints the DP Born after promotion to `REAL*16`; this is not a newly evaluated
QP Born. The audit therefore combines that original-point DP Born with the
recorded repaired-QP eikonals and real QP coupling. Exact binary128 decoding and
90-digit arithmetic do not remove the input convention mismatch or provide a
numerical bound on it. Mathematical pi also makes this a higher-precision
specialization of the formula rather than a bitwise reproduction of DP FKS.

The predicted single-to-double-pole ratio, `-single_factor/2`, cancels the Born
and coupling prefactor exactly. A comparison with saved, complete reducer pole
ratios therefore avoids that normalization uncertainty, while retaining the
kinematic and completeness limitations. Absolute pole comparisons must retain
the DP-Born caveat. Agreement of either comparison supports pole consistency;
it does not establish the finite term, independent QP Born accuracy, automatic
stability acceptance or whole-amplitude correctness. In particular, the
historical threshold TIRust result omits 37 refused groups and cannot serve as
an amplitude reference, regardless of its reported pole values.
