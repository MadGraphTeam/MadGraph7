# One-loop virtuals of quarkonium processes in MadLoop

MadLoop computes the one-loop QCD virtual of processes with S-wave bound
states (quarkonia, leptonia) in the final state, colour singlets and octets of
any total angular momentum J, together with elementary particles. A process
may also have one P-wave bound state (see below). The bound states are those
of the tree-level implementation (arXiv:2510.26773, arXiv:2607.26739),
projected the same way.

## Usage

```text
import model loop_sm_onia-c_mass
generate g g > etac(1|1S01) [virt=QCD]
output standalone gg_etac
```

`loop_sm_onia` is `loop_sm` with the Fock states of `sm_onia`; it is written by
`tools/make_onia_model.py models/loop_sm models/loop_sm_onia`, which works for
other loop UFO models too. The constituents of a bound state must be massive:
`loop_sm_onia-c_mass` has massive charm, bottom and top (3-flavour scheme).

The public MadLoop routines (`SLOOPMATRIX`, `SLOOPMATRIX_THRES`) take the
momenta of the physical process, with one momentum per bound state. The mass
of a bound state is `m1 + m2`, the sum of its constituent masses, as in
`pmass.inc`. `check_sa.f` and `tools/onia_loop_check.py` (deterministic
points, several reduction libraries, JSON output) drive the output. The LDMEs
are read from `Cards/onia_card.dat`.

The sum over the polarisations of a bound state can be restricted with
`ONIA_SELECT_JZ(K, NJZ, JZS)`: bound state K (in the order of the process)
then takes only the NJZ values `JZS(1:NJZ)` of J_z (NJZ = 0 restores the
sum). J_z is the helicity of the bound state in the frame of the momenta
given to `SLOOPMATRIX` (its polarisation vectors from `VXXXXX`). With
momenta in the partonic centre-of-mass frame, J_z = +-1 and 0 of a J = 1
state are its transverse and longitudinal polarisations in the helicity
frame. The Born, the virtual and the predicted poles are those of the
selected J_z. `tools/onia_loop_check.py` exposes it as `jz=(K, [J_z, ...])` of
`run_points` (driver option `--jz`).

Supported: `[virt=QCD]` with S-wave bound states in the final state, and at
most one P-wave bound state. A P-wave output needs a TIRust installation with
raised propagator powers (`raised_powers` in its metadata; `install tirust
--update`), and must be output alone or with other P-wave processes: its
HELAS library is in dual numbers.

Two P-wave bound states (for example `e+ e- > chic0(1|3P01) hc(1|1P11)`) are
generated with both precisions. The quadruple-precision path takes the dual
numbers of `aloha/template_files/mp_dual2_variables.f` (both directions and
the mixed derivative; one-P-wave outputs keep `mp_dual_variables.f`) and the
`MP_` copies of the two-state routines of `onia_loop.f`. It costs about 60
times the double-precision time, nearly all of it in the binary128 loop
numerators. Their predicted poles are complete, the dipole-dipole term
included. Their virtual needs TIRust integrals with two dotted propagators
or a tripled one, at the thresholds of both pairs, and with a jet pentagons
of rank 7:
- **With the pinned TIRust (1c130a81).** The virtual is reported unstable.
- **With the TIRust candidate da7902b and the bridge patch v16** (TIRust PR
  #71, not yet in MadGraph). It serves every integral of the processes
  below. The earlier candidate 4f1b06c (TIRust main) with the bridge patch v9
  serves `e+ e- > chic0 hc` and `g g > chic0 chic0`, the processes of the
  report's tables.

Checked for every colour assignment of the two states, on da7902b
(`docs/benchmarkfolder/benchmark.pdf`, section 5.2;
`research/two_pwave/FIXTURES.md`):
- **Two singlets.** `e+ e- > chic0 hc`, where the whole pole is the
  dipole-dipole term (see below; on da7902b through the acceptance test),
  and `g g > chic0 chic0`, where all its parts contribute. For the latter, up
  to 20 GeV, both poles are as predicted to 1e-8 in double precision, and to
  3e-14 in quadruple precision. At 200 GeV double precision loses the poles,
  and MadLoop's stability test (default threshold) rescues the points in
  quadruple precision. da7902b corrects rank-6 hexagons with two doubled
  lines. The results of 4f1b06c move by at most 3e-9, apart from the
  double-precision 1/eps at the forward point of 50 GeV (4e-5), where the
  pole is what is left of a large cancellation; the stability test's
  results stay within their estimates.
- **A singlet and an octet.** `u u~ > chic0(1|3P01) Jpsi(1|3P08)`. The 1/eps
  is as predicted to 3e-12 at 8 GeV in double precision, and to 4e-14 at
  20 GeV in quadruple precision.
- **Two octets.** `g g > Jpsi(1|3P28) Jpsi(1|3P18) g`, which needs the
  rank-7 pentagons. The 1/eps is as predicted to 4e-11 at 20 GeV in either
  loop direction; a pass takes about 8000 s.
- **Unequal flavours and J = 1.** `g g > chic1(1|3P11) chib0(1|3P01)`. Its
  Born equals the tree output's bit for bit when both have the same masses:
  the loop layer always takes M = m1 + m2, while the tree output takes the
  model's physical masses. The 1/eps is as predicted to 8e-13 at 20 GeV.

In double precision the poles lose digits at high energy and near the beam,
where they are what is left of a large cancellation. MadLoop's stability
test estimates the error there; where it was run, its estimates are of the
size of the true errors, and points beyond its threshold go to quadruple
precision. The
acceptance tests `test_two_pwave_loop_standalone`,
`test_two_pwave_gg_loop_standalone` and
`test_two_pwave_singlet_octet_loop_standalone` (`test_cmd.py`) check the
poles and finite parts of three of these processes. They take the TIRust of
`$TIRUST_LIB`, or the configured one, and are skipped without raised powers.

Such outputs are built with the medium code model on x86-64, since their loop
arrays can exceed 2 GB. Their loop numerators reach rank R+2, and the
polynomial routines are written for that rank (`loop_onia.loop_rank_increase`).

The FKS modes `[QCD]` and `[real=QCD]` go to the fixed-order NLO of bound
states (FKS subtraction, `madgraph/fks/fks_onia.py`), with its own checks.

Refused with an explanatory error:
- more than two P-wave bound states, and states with L >= 2;
- loop-induced processes (`[sqrvirt=QCD]`);
- `[virt=QED]`;
- massless constituents;
- the option `loop_color_flows`;
- a Born with more than one squared split order.

## Conventions

The output is MadLoop's: the UV-renormalised virtual
`2 Re(M_Born^* M_loop)`, as finite part and the coefficients of 1/eps and
1/eps^2, in the 't Hooft-Veltman scheme. The normalisation is
`alpha_s/(2 pi) (4 pi)^eps/Gamma(1-eps) (mu^2/mu_R^2)^eps`.

- **Projectors:** 4-dimensional external projectors on the constituent
  spinors. The gamma5 of a spin singlet is the 4-dimensional, anticommuting
  one.
- **Threshold:** the constituents are at zero relative velocity, and the
  integrals are their dimensionally regulated values at v = 0. The Coulomb
  singularity, a scaleless power divergence there, is absent: it belongs to
  the LDME.
- **Renormalisation:** heavy quarks are renormalised on shell (mass and wave
  function). alpha_s is MSbar with the light flavours, the heavy quarks
  decoupled (zero-momentum subtraction), as in `loop_sm`.

This is the normalisation of the onia FKS layer (`onia_fks`,
`BinothLHA_onia.f`). CDR results normalised to the D-dimensional Born, such
as those of Petrelli et al., hep-ph/9707223, agree with it.

## How it works

MadLoop computes the loop amplitudes of the constituent process,
`g g > c c~` for `g g > etac`, with the constituents of a bound state of
momentum K at `p_i = (m_i/(m1+m2)) K`. For equal masses that is exactly K/2,
so both constituents carry the same binary64 momentum.

**Colour.** The colour projectors (`delta_ij/sqrt(Nc)` and
`sqrt(2) T^a_ij`) multiply the colour strings of the Born, loop and UV
counterterm diagrams (`loop_color_amp.py`). MadLoop's colour factors between
loop and Born amplitudes are therefore those of the projected bases.

The onia layer's own colour data (the projected Born, the colour-linked
Borns and the soft currents of P-wave states) is written in the basis of
the Born colour flows, not between Born amplitudes. Each matrix has the
form `C_ab = sum_cd conj(l_c(a)) X_cd r_d(b)`, with `l` and `r` the sparse
amplitude-to-flow maps of the Born basis and of the basis on the right
(the Born basis, or that of an S-wave state of a soft current).
`onia_loop.f` computes the flows `J = l U` of the amplitudes and contracts
them with the flow matrices `X` (`ONIA_COLOUR_FLOWS`, `ONIA_SANDWICH`).
The data then grows with the square of the number of flows, not of Born
amplitudes. For `g g > Jpsi(1|3P28) Jpsi(1|3P18) g`, with 786 Born
amplitudes and 36 flows, that is a factor of about 480. The
colour links `<T_j.T_k>` are computed in the projected Born basis, with
the total charge `T_Q + T_Qbar` of a bound state on its octet index
(`loop_onia.flow_color_links`). A unit test checks them against the
insertion into the diagrams before the projection, as in MadFKS.

**Spin and J.** The spin projector is a weighted sum over the constituent
helicities h, `B(J_z) = sum_h PROJ(h,J_z) A(h)`, as at tree level. MadLoop
contracts each loop amplitude with the conjugate Born of the same helicity
configuration before the reduction. Because the projection is linear, the
projected interference is MadLoop's ordinary one with each Born amplitude
A(h) replaced by the effective Born

    W(h) = F sum_{J_z} conj(PROJ(h,J_z)) B(J_z),

where F is the colour and LDME normalisation (`onia_loop.f`). The reduction,
the helicity sum and the stability checks are unchanged. Which constituent
helicities contribute depends on the frame: constituents at rest have no
helicity axis of their own. For that reason MadLoop's learnt helicity filter
is switched off for bound states (`HelicityFilterLevel = 0`), and the
configurations without effective Born are skipped point by point.

**Reduction.** With `p_Q = p_Qbar` every Gram determinant of a loop with both
constituents vanishes. In the Coulomb-exchange loops the propagators are
linearly dependent, `2 D_g = D_Q + D_Qbar`. TIRust (backend 9) reduces these
integrals (huashengshao/TIRust#38) and is the only library of the default
chain for bound states (`MLReductionLib = 9`). The other libraries fail
there, and MadLoop's stability tests do not always notice: COLLIER alone, for
instance, gives -13.545 instead of -32/3 for `e+ e- > Jpsi(1|3S11)` and passes
them. The public `SLOOPMATRIX` of a bound-state output therefore reports the
result of any library other than TIRust as unstable (return code `4x0`, a
warning on the first such point). They remain selectable for cross-checks.
The guard applies when the bound states have coloured constituents
(`ONIA_THRESHOLD_LOOPS` in `onia_loop_data.inc`): no QCD loop holds both
constituents of a leptonium. (IREGI has a separate limitation there: at
leptonium scales, M = 20 MeV and below, it drops the loop and reports the
result stable; it is right at M = 2 GeV.)
When TIRust refuses an integral (two bound states in one process, see the
open items of the benchmark report; the binary128 path of the pinned
revision, for instance at most points of `g g > Jpsi(1|3S11) g` at
sqrt(s) = 5 GeV with `CTModeRun = 4`), its contribution is missing and the
point is reported unstable (`H=4`, accuracy 1) in every mode. With
`CTModeRun = -1` MadLoop's stability tests do this: a refusal counts as a
floating-point exception of the reduction (`FPE_IN_DP/QP_REDUCTION`), so
MadLoop tries the next library or quadruple precision first. The forced
modes (`CTModeRun = 1` or 4) and the initialisation points run no test.
There the public `SLOOPMATRIX` reads the same flags and reports the point
unstable itself, with a warning on the first such point.

**Stability test.** MadLoop's default stability test reverses the loop
direction, which leaves the counterterm amplitudes unchanged. Bound-state
outputs therefore add a counterterm rounding estimate: the double-precision
accuracy is at least `epsilon sqrt(sum t^2) / |c0|`, over the terms t of the
counterterm interference (`ONIA_CTSQ` of `loop_matrix.f`). A point beyond
`MLStabThres` is rescued in quadruple precision, which computes the
counterterms in quadruple precision too. The estimate is conservative: in the
comparison below it exceeds the real error by 40 to 6e3.


The counterterm amplitudes can lose digits near a soft jet without trace:
- the R2 gluon self-energy terms of a heavy quark carry its mass squared;
- they multiply current contractions that vanish exactly;
- their rounding is then amplified by the soft gluon.

For `g g > Jpsi(1|3S11) g` with a 1.5 MeV gluon, the double-precision result
was off by 1.7e-4 with the physical b and t masses, and by 22% with
m_b = m_t = 1e4 GeV. MadLoop estimated the accuracy at 1e-9 in both cases.

**What it catches.** The counterterm estimate catches every such point. With
1e4 GeV b and t, the four soft points (sqrt(s) <= 3.15) are rescued in
quadruple precision, and every point and J/psi polarisation is right. With
physical masses only the 1.5 MeV point is rescued, and the other accuracies
stay near 1e-9.

MadLoop's rotation of the point (`NRotations_DP = 1`, a permutation of the
axes) catches the worst point, but not the transverse J/psi alone
(`ONIA_SELECT_JZ`) there, which stays 33% off. It also costs one more
double-precision evaluation per point, about +50%. Bound-state outputs leave
it at MadLoop's default, 0.

**Precision rescue.** The quadruple-precision rescue improves the physical
momenta, each bound state on its mass shell, and splits the bound states
again (`MP_IMPROVE_PS_POINT_PRECISION` of `onia_loop.f`). TIRust's binary128
path does not yet reduce the threshold integrals, so a point that fails the
double-precision tests is reported unstable.

**Diagnostic.** `ONIA_SET_RELATIVE_MOMENTUM(Q)` gives the constituents the
relative momentum `+-q` in the rest frame of the pair (`--relmom` of
`tools/onia_loop_check.py`). The integrals are then ordinary ones, which any
library reduces; the Coulomb singularity `C pi^2/v` appears in the finite
part (`C = C_F` for a singlet, `C_F - C_A/2` for an octet). Subtracted and
extrapolated to `v = 0`, it gives an independent check of the threshold
value (`relmom_check` of `tools/onia_loop_validate.py`).
The physical momenta should put the constituents on their mass shell, a
bound state of mass `2 sqrt(m^2 + q^2)` (equal masses): the
quadruple-precision improvement of the point puts them there, so it moves a
point given with the mass `2m` (MadLoop's usual rule for off-shell external
legs).

**P-wave bound states.** The amplitude of a P-wave state is the derivative
of the projected constituent amplitude in the relative momentum q of the
constituents, `p_Q = K/2 + q`, `p_Qbar = K/2 - q`, along the polarisation
vectors `eps(L_z)` of the orbital angular momentum. The spin projectors are
combined with the Clebsch-Gordan coefficients `<1 L_z; S S_z|J J_z>`, as at
tree level (`matrix_standalone_v4_onia_pwave.inc`).

MadLoop computes the constituent amplitudes in first-order dual numbers:
value and derivative along the current direction `eps(L_z)`, one direction
at a time. The pieces are:
- **HELAS library.** The double-precision routines of the output are the
  dual ones of the tree-level P-wave library, its loop vertex routines
  included. Their wavefunctions, amplitudes and loop momenta are shared
  through the module `PW_DUAL` of `polynomial.f`.
- **Loop polynomials.** The polynomial updates are the plain ones, applied to
  the value and the derivative: `WL' = U(WL, C0)` and
  `DWL' = U(DWL, C0) + U(WL, C1)`, for vertex coefficients `C0 + d C1`.
- **Propagators.** The derivative of the loop propagators,
  `-2 (S_i - l).dS_i / D_i^2`, gives integrals of rank R+1 with line i
  doubled. TIRust reduces them (`tirust_tensor_evaluate_powers`). Their
  numerators come from four extra polynomials per loop group.
- **Weights.** The interference with the projected Born uses the effective
  Born `W1` on the value and `W2(L_z)` on the derivative of each direction
  (`ONIA_PREPARE` of `onia_loop.f`).
- **Quadruple precision.** The QP path repeats all of this in `MP_Dual`
  numbers (`aloha/template_files/mp_dual_variables.f`; the external
  wavefunctions in `aloha_functions_mp_dual.f`, generated by
  `tools/make_mp_dual_aloha_functions.py`). The QP loop vertices are the
  dual ones of the output, the doubled lines are reduced by TIRust's
  binary128 path, and the effective Born and counterterm interference are
  `MP_ONIA_PREPARE` and `MP_ONIA_PW_CT`. A P-wave output always computes
  its QP integrand in QP.
- **Extended precision (optional).** With `#PWaveExtendedPrecision .TRUE.`
  in `Cards/MadLoopParams.dat` (off by default), the double-precision passes
  (`CTModeRun` -1, 1, 2) run in extended precision. That is the x87 80-bit
  format, or binary128 on hosts without it (kind
  `EPK = SELECTED_REAL_KIND(18)`). The PS point, the dual wavefunctions, the
  loop coefficients, the projections and the contractions are all computed in
  this format. The tensor integrals come from TIRust's double-precision entry,
  with shifts and invariants passed as binary64 hi + lo words
  (`tirust_tensor_evaluate_powers_ext`, unit powers included). The EP code is
  the QP code with its names (MP_ -> EP_) and kind changed, derived when the
  output is written (`madgraph/loop/onia_ep.py`). Only the dispatch and the
  TIRust routines are in the templates. The quadruple-precision rescue is
  unchanged. The code is written for one P-wave state with TIRust from an
  installation with `raised_power_extended_inputs` in its metadata. Other
  P-wave outputs print a warning and ignore the switch; non-P-wave outputs
  ignore it silently. At 1 TeV (u u~ > chic0 g, g g > chic0 g) the stability
  test then gains about three digits at about 5 times the double-precision
  time. This needs TIRust's check pass to shift extended inputs at their own
  precision (bridge v2); with the earlier bridge, whose shift is 2^-52, the
  gain is about one digit. Details in `research/pwave_stability/README.md`.

Status:
- At zero relative momentum, TIRust 48958d1 (part of huashengshao/TIRust PR #56,
  stacked on #47 and #55, not yet in the pinned revision) serves every
  integral of the 2 -> 1 processes, the Coulomb box with its gluon line
  doubled included. Earlier revisions refuse that box, and the point is then
  reported unstable. With 48958d1 the one-loop literature values are
  reproduced to 1e-12 or better (`research/pwave_fd/README.md`):
  - `a a > chic0/chic2` (Barbieri et al.);
  - `g g > chic0/chic2` (Petrelli et al.), and their octets with the
    corrected colour factor;
  - `a g > Jpsi(1|3P08)/(1|3P28)`;
  - `g g > etac(1|1P18)` (the `onia_fks` constant, P->S pole included).

  It also serves `u u~ > chic0 g` and `g u > chic0 u` (poles checked), but
  not the doubled-gluon pentagons of `g g > chic0 g`; TIRust 370b632 (in
  PR #56) serves those too (poles checked, finite part equal to QP).
- When TIRust refuses an integral with a doubled line, its contribution is
  missing: the point is reported unstable (`H=4`) in every mode
  (`CTModeRun`), with a warning on the first such point.
- The quadruple-precision path agrees with the double-precision one
  (1e-14 to 4e-13 away from threshold) and rescues points that fail the
  double-precision tests.
- `docs/benchmarkfolder/research/pwave_fd/` checks the derivative, doubled
  lines included, against finite differences at a relative momentum
  `q0 != 0`.

**IR poles.** `ONIA_IR_POLES` predicts the poles from the colour-linked
projected Borns (Catani-Dittmaier-Trocsanyi; arXiv:2402.19221, eqs.
(216)-(217)). An octet is a massive adjoint particle with the colour charge
`T_Q + T_Qbar`, and a singlet is colourless. A P-wave state adds the P->S
single pole of appendix B (`ONIA_PW_SOFT_POLE`, the formula of
`getpoles_onia` of the onia FKS layer): a soft gluon turns the P wave into
the S-wave states S' of colour C' (P[1] -> S[8]; P[8] -> S[1], S[8]),
whose amplitudes are the values of the Born amplitudes at q = 0 with the
spin projectors of S_z = J_z - L_z. Their colour is generated by
`loop_onia.pwave_soft_current`. The term is the sum of the cross vectors
`<P| T_k.Q_eff |S'>` of the coloured legs k. For equal masses,
`Q_eff = 2 (T_Q - T_Qbar)`, the derivative of the soft functions of the
heavy quarks in q.

The virtual has no self-like pole of the P-wave state, for any constituent
masses: the intra-pair terms of the IR operator are of order q^2 (eq. (B.2)
of the two-P-wave manuscript). The self term of the soft current and the
LDME counterterm of arXiv:2402.19221, appendix B, belong to the real
emission and the NRQCD renormalisation, and cancel each other there; with
the coefficient 8/(m_Q m_Qbar) of that appendix they cancel only for equal
masses. `ONIA_IR_POLES` included them until the B_c P-wave check of
`docs/benchmarkfolder` (sect. 5.2.1, `research/pwave_fd/unequal_mass.py`),
where they were off by up to 90% for an octet.

The term vanishes when the Born of every S' vanishes, as for
`g g > chic0/chic2`, and when the bound state has a single coloured partner,
as in `e+ e- > Jpsi(1|3PJ8) g`. For `g g > etac(1|1P18)` it is -2 C_A, and
the prediction is the computed 1/eps coefficient, -18.

**Two P-wave states.** The single pole has three parts
(`docs/benchmarkfolder/research/two_pwave/README.md`):
- the colour-linked Borns of the mixed derivative, as for one state;
- the P->S term of each state, with the other one still differentiated
  (`ONIA_PW2_SOFT_TERM`). The other state counts as a coloured leg if it is an
  octet;
- the dipole-dipole exchange between the two states
  (`ONIA_PW2_DIPOLE_DIPOLE`), which needs the eight-quark operators of He,
  Kniehl and Wang (arXiv:1809.07993). It takes their kernel f1, f2 of
  w = v1.v2, with its series near w = 1, and the colour data of
  `loop_onia.pwave_dipole_dipole`.

The virtual stays the plain QCD one. `ONIA_NRQCD_CT(P_PHYS, MU_L, CT)` returns
the MS-bar NRQCD counterterm, `-PDD [1/eps + ln(mu_R^2/mu_L^2)]`, in the units
of `ANS` (CT(1) finite, CT(2) the 1/eps coefficient), for the NLO layer. For
`e+ e- > chic0 hc` the predicted 1/eps coefficient per Born reproduces the
independent evaluation of the research folder: 0.71157569957729 at
sqrt(s) = 10.6 GeV, 0.72155466286996 at 7 GeV and 0.67519653345551 at 30 GeV,
to 6e-15.

With the first TIRust candidate with two dotted propagators (59aee45 and the
bridge patch v3), every integral of `e+ e- > chic0 hc` is served in both
passes, and the computed poles match the prediction:
- the 1/eps coefficient to 3e-12 at 7 and 10.58 GeV, 1.5e-11 at 20 GeV and
  2e-10 at 30 GeV;
- no 1/eps^2.

Three further checks:
- **State order.** The other order of the two states gives the same virtual.
- **Finite part.** At a relative momentum q != 0, the mixed dual derivative
  agrees with finite differences
  (`docs/benchmarkfolder/research/pwave_fd/README.md`).
- **Quadruple precision** (`research/two_pwave/qp_check.py`, TIRust d5442e6,
  four points at 7, 10.58 and 30 GeV):
  - The QP 1/eps coefficient matches the prediction to 1e-15 to 2e-14, where
    double precision gets 3e-12 to 2e-10.
  - The finite part agrees with double precision to 3e-13 to 2e-11, and to
    1e-9 at one point where the double-precision value is the less accurate
    one: in QP both 10.58 GeV points give the same finite part per Born in all
    printed digits.
  - Both loop directions (`CTModeRun` 4, 5) agree exactly in double precision.
  - With a forced rescue (`CTModeRun` -1, `MLStabThres` 1d-30) every point
    returns 323.

## Validation

See `docs/benchmarkfolder/benchmark.pdf` and `tools/onia_loop_validate.py`.
