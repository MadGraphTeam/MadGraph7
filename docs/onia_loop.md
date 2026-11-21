# One-loop virtuals of quarkonium processes in MadLoop

MadLoop computes the one-loop QCD virtual of processes with S-wave bound
states (quarkonia, leptonia) in the final state, colour singlets and octets of
any total angular momentum J, together with elementary particles. The bound
states are those of the tree-level implementation (arXiv:2510.26773,
arXiv:2607.26739), projected the same way.

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

Supported: `[virt=QCD]` with S-wave bound states in the final state.

Refused with an explanatory error:
- P-wave states;
- the FKS modes `[QCD]` and `[real=QCD]` (developed on the `onia_fks` branch);
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
When TIRust refuses a point (two bound states in one process, see the open
items of the benchmark report), the point is reported unstable.

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

**IR poles.** `ONIA_IR_POLES` predicts the poles from the colour-linked
projected Borns (Catani-Dittmaier-Trocsanyi; arXiv:2402.19221, eqs.
(216)-(217)). An octet is a massive adjoint particle with the colour charge
`T_Q + T_Qbar`, and a singlet is colourless.

## Validation

See `docs/benchmarkfolder/benchmark.pdf` and `tools/onia_loop_validate.py`.
