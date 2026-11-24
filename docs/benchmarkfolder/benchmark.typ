// MadLoop for quarkonium at one loop: validation, reducers and performance.
// Tables are read from data/*.json (scripts/collect.py, from the campaigns
// of tools/onia_loop_validate.py and tools/onia_loop_timing.py). Build from
// this directory:  typst compile benchmark.typ benchmark.pdf
#import "style.typ": *

#let meta = json("data/meta.json")
#let val = json("data/validation.json")
#let red = json("data/reducers.json")
#let tim = json("data/timing.json")
#let mg = json("data/mg255.json")
#let mgp = json("data/mg255_pol.json")
#let pw = json("data/pwave.json")
#let stab = json("data/stability.json")

#show: report-style.with(
  title: "MadLoop for quarkonium at one loop — validation, reducers and performance",
  short: [MadLoop for quarkonium at one loop], date: meta.date, revision: meta.revision)

#let case(name) = val.find(c => c.name == name)
#let lbl(c) = text(8.4pt, font: "Libertinus Serif", eval(c.label, mode: "math"))
#let dev-cell(x, target: 1e-8) = table.cell(fill: acc-fill(x, target))[#sci(x)]
#let codes(c) = c.return_codes.map(str).join(", ")

#align(center)[
  #v(0.1em)
  #text(24pt, weight: "bold", fill: navy)[MadLoop for quarkonium at one loop]
  #v(-0.5em)
  #text(12pt, fill: luma(80))[S- and P-wave one-loop QCD virtuals: validation, reduction and performance]
  #v(-0.3em)
  #text(9pt, fill: luma(110))[#meta.date · branch `onia_loop` · revision #meta.revision · #meta.host]
]

MadLoop now computes the one-loop QCD virtual of processes with S-wave bound
states — quarkonia and leptonia, colour singlets and octets of any $J$ — in the
final state, together with elementary particles: the UV-renormalised
interference $2 op("Re")(cal(M)_"Born"^* cal(M)_"loop")$, its finite part and
its infrared poles, in MadLoop's normalisation ('t Hooft–Veltman scheme,
4-dimensional external projectors, $alpha_s\/2pi (4pi)^epsilon\/Gamma(1-epsilon)$),
with the constituents at zero relative velocity, where the Coulomb singularity
is absent in dimensional regularisation (it is part of the long-distance
matrix element). The bound states are those of the tree-level implementation
(arXiv:2510.26773, arXiv:2607.26739) and the model is `loop_sm_onia`, the
`loop_sm` UFO model with the Fock states of `sm_onia`. A process may also have
one P-wave bound state (@pwave), and the S-wave onium + jet virtuals are compared
with those of the MG5 2.5.5 onia outputs (@mg255).

#let analytic = val.filter(c => c.reference != none and c.at("status", default: "ok") == "ok")
// The target: 1e-8, the precision of the published value (e+ e- K factors),
// or MadLoop's own accuracy estimate when that is larger.
#let target(c) = calc.max(if c.name.starts-with("ee_") and c.name != "ee_jpsi" { 1e-6 } else { 1e-8 }, c.at("accuracy", default: 0))
#let passed = analytic.filter(c => c.max_deviation != none and c.max_deviation <= target(c))
#let rm-cases = val.filter(c => c.at("relmom", default: none) != none)
#let worst-pole = calc.max(..rm-cases.map(c => c.max_pole_deviation))
#let worst-rm = calc.max(..rm-cases.map(c => c.relmom.at("9").relative))
#let worst-up = calc.max(..tim.map(r => r.upper95))
#let pw-pass = pw.two_one.filter(c => c.max_deviation <= c.target)
#let worst-mg = calc.max(..mg.filter(c => c.reference == "direct").map(c => c.max_fin))
#let all-identical = tim.all(r => r.identical)
#let one-state = analytic.filter(c => c.stable > 0)
#let badge = [Analytic: #passed.len() of #one-state.len() within target#if one-state.len() < analytic.len() [ (+ #(analytic.len() - one-state.len()) with two bound states, flagged)] · 2 → 2 poles ≤ #sci(worst-pole, d: 0), $v -> 0$ ≤ #sci(worst-rm, d: 0) · P-wave: #pw-pass.len() of #pw.two_one.len() within target · 2.5.5 onium + jet ≤ #sci(worst-mg, d: 0) · timing: upper 95% ≤ #fixed(worst-up, 2)#if all-identical [, bit-identical]]

#grid(columns: (1.2fr, 1fr), gutter: 8pt,
  box-note[
    #set par(justify: false)
    *Published on `onia_loop`* (from `main` #raw(meta.baseline))
    #v(-0.2em)
    #set text(8.2pt)
    #table(columns: (auto, 1fr), stroke: none, inset: (x: 2pt, y: 2.4pt),
      ..meta.commits.map(((sha, subject)) => (raw(sha), subject)).flatten())
    #v(-0.3em)
    #text(8.2pt)[*Model* `loop_sm_onia` (`loop_sm` with the Fock states of `sm_onia`, by
      `tools/make_onia_model.py`). *Reducer* TIRust (pin `1c130a81`), alone by default.]
  ],
  block(width: 100%, stroke: 0.6pt + luma(195), inset: 9pt, radius: 3pt, {
    set text(9pt)
    set par(justify: false)
    [*Validation and performance: policy and status*]
    v(-0.1em)
    set text(8.4pt)
    meta.status_lines.map(l => [- #eval(l, mode: "markup")]).join()
    v(0.2em)
    align(center, box(fill: warm, inset: (x: 8pt, y: 4pt), radius: 3pt,
      text(8.6pt, weight: "bold", fill: rgb("#8a4b08"), badge)))
  }),
)

#grid(columns: (1fr, 1fr), gutter: 8pt,
  box-note(fill: white, stroke: 0.6pt + luma(205))[
    #set par(justify: false)
    *Open, not on `onia_loop`* (@open)
    #set text(8.4pt)
    #meta.open_items.map(l => [- #eval(l, mode: "markup")]).join()
  ],
  box-note(fill: white, stroke: 0.6pt + luma(205))[
    #set par(justify: false)
    *User decisions of 21 November*
    #set text(8.4pt)
    #meta.decisions.map(l => [- #eval(l, mode: "markup")]).join()
  ],
)

= Implementation <impl>

*Constituent process.* A bound state of momentum $K$ and mass $m_1 + m_2$ is
generated as its two constituent legs; MadLoop computes the loop amplitudes of
that process (`g g > c c~` for `g g > etac`) with the constituents at
$p_i = m_i\/(m_1+m_2) K$, exactly $K\/2$ for equal masses.

*Colour.* The projectors $delta_(i j)\/sqrt(N_c)$ and $sqrt(2) T^a_(i j)$
multiply the colour strings of the loop, UV-counterterm and Born diagrams, so
MadLoop's loop–Born colour factors are those of the projected bases.

*Spin and $J$.* MadLoop contracts each loop amplitude with the conjugate Born
amplitude of the same helicity configuration $h$ before the reduction. The
projection being linear,
$ sum_(J_z) B(J_z)^* L(J_z) = sum_h W(h)^* L(h), quad
  B(J_z) = sum_(h') "PROJ"(h', J_z) A(h'), quad
  W(h) = F sum_(J_z) "PROJ"(h, J_z)^* B(J_z), $
with $F$ the colour and LDME normalisation of the tree-level output: the
reduction, the helicity sum and the stability checks are MadLoop's own. Which
constituent helicities contribute depends on the frame, so MadLoop's learnt
helicity filter is off for bound states and the configurations with $W(h) = 0$
are skipped point by point. The public `SLOOPMATRIX` takes the physical momenta.

*Integrals.* With $p_Q = p_(overline(Q))$ every Gram determinant of a loop with
both constituents vanishes, and in the Coulomb-exchange loops the propagators
are linearly dependent, $2 D_g = D_Q + D_(overline(Q))$. TIRust (pin `1c130a81`,
huashengshao/TIRust\#38) reduces them by partial fractions with Taylor jets and
closed-form IR-singular bases, in dimensional regularisation at $v = 0$.

*Poles.* `ONIA_IR_POLES` evaluates the Catani–Dittmaier–Trócsányi structure
(arXiv:2402.19221, eqs. (216)–(217)): a colour singlet is colourless, an octet a
massive adjoint particle of mass $m_1+m_2$ with colour charge
$T_Q + T_(overline(Q))$. A P-wave state adds the $P -> S$ pole of its soft current
(@pwave).

*P waves.* The amplitude of a P-wave state is the derivative of the projected
constituent amplitude in the relative momentum $q$ ($p_Q = K\/2 + q$,
$p_(overline(Q)) = K\/2 - q$) along the polarisation vectors $epsilon(L_z)$,
combined with the spin projectors by $chevron.l 1 L_z; S S_z|J J_z chevron.r$ as at
tree level. MadLoop computes it in first-order dual numbers: dual HELAS and loop
vertex routines, loop polynomials that carry their derivative, and the
derivative of the propagators, $-2 (S_i - l) dot d S_i\/D_i^2$, as tensor
integrals with one doubled propagator, reduced by TIRust
(`tirust_tensor_evaluate_powers`). The quadruple-precision rescue repeats it in
`MP_Dual` numbers.

= Validation against analytic results <analytic>

Deterministic points of `tools/tirust_benchmark_points.py` (2 → 1: the rest
frame and longitudinal boosts with rapidity in $[-2, 2]$), charm mass
$m_c = 1.55$ GeV unless stated, $mu = M$, TIRust in double precision with
MadLoop's stability checks. Coefficients in units of the Born times
$alpha_s\/2pi$; literature values in units of $alpha_s\/pi$ doubled (sources in
@sources and `research/literature.md`).

#figure(caption: [Finite part $c_0$ and poles $c_1$, $c_2$ of the virtual against the analytic results. "dev." is the largest deviation over the points (relative for $c_0$, absolute for the poles), green within the target: $10^(-8)$, the precision of the published value, or MadLoop's own accuracy estimate ("acc.", the largest over the points) when that is larger.])[
  #tstyle(table(columns: (auto, 1fr, auto, auto, auto, auto, auto, auto, auto),
    stroke: tstroke, fill: thead-fill, align: (left, left, right, right, right, right, right, center, right),
    [Process], [Reference], [$c_0$ ref.], [$c_0$ MadLoop], [$c_1$, $c_2$], [dev.], [acc.], [stable], [ms/pt],
    ..analytic.map(c => (lbl(c), text(7pt, c.ref_short), dec(c.reference.at(0), 8),
      if c.stable == 0 { table.cell(fill: warm)[#flag[exceptional]] } else { dec(c.finite, 8) },
      [#dec(c.single, 4) #linebreak() #dec(c.double, 4)],
      if c.stable == 0 { table.cell(fill: warm)[—] } else { dev-cell(c.max_deviation, target: target(c)) },
      sci(c.accuracy), [#c.stable/#c.npoints], ms2(c.time_ms))).flatten()))
]

#box-note[
  *Massive quark loops.* The $q overline(q) -> c overline(c)[3S_1^((8))]$
  reference adds to Petrelli et al.'s constant the vacuum polarisation of the
  bound charm, $-16\/9 T_F$ (Hagiwara, Sumino, Yokoya, 0804.1014), and of the
  massive $b$ and $t$ quarks of the model, $4 integral_0^1 x(1-x) ln(1 - x(1-x) M^2\/m^2) d x$
  each ($-0.0609$ and $-4 dot 10^(-5)$); with $m_b = 10$ TeV the plain $n_l = 3$
  value is recovered. The $e^+ e^-$ references are computed for $m_c = 1.5$ GeV,
  $sqrt(s) = 10.58$ GeV with $b$, $t$ decoupled ($m_b$, $m_t -> infinity$) and photon
  exchange only.
]

#box-note(fill: warm)[
  *Two bound states.* With the pinned TIRust every $e^+ e^- -> J\/psi thin eta_c$
  point meets refused lanes (a threshold bubble pinch, and pentagons holding the
  Coulomb relations of both onia) and is reported exceptional. The second row
  uses TIRust 370b632 (in PR \#56, stacked on \#47 and \#55; not yet pinned), which
  serves them: every point stable in double precision (code 213), poles zero to
  $1.3 dot 10^(-7)$, the finite part within $3.1 dot 10^(-7)$ of Gong and Wang.
  Forced quadruple precision gives poles of $10^(-15)$ and the finite part
  within $6 dot 10^(-8)$, inside the 7 published digits; what is left comes from
  the double-precision momenta, whose onium masses sit a few ulp off $2 m_c$.
]

= 2 → 2 processes: poles and an independent check of the finite parts <twotwo>

No point-wise one-loop results are published for hadronic 2 → 2 quarkonium
production. The poles are compared with the universal prediction, and the
finite parts with a calculation that does not use the degenerate
threshold integrals: the constituents are given a relative momentum $plus.minus q$
($v = |q|\/E_Q$ in the rest frame of the pair), where every integral is an
ordinary one (TIRust and COLLIER agree there within #sci(calc.max(..val.filter(c => c.at("relmom_libs_agree", default: none) != none).map(c => c.relmom_libs_agree)))), the
results for $plus.minus q$ are averaged, the Coulomb singularity $C pi^2\/v$ with
$C = C_F$ (singlet) or $C_F - C_A\/2$ (octet) is subtracted, and
$v = 0.001 ... 0.02$ is extrapolated to $v = 0$ (quartic fit).

#let rm = val.filter(c => c.at("relmom", default: none) != none)
#figure(caption: [Poles against the prediction (largest absolute deviation over the points) and the $v -> 0$ extrapolation of the finite part at the first point against MadLoop's value at $v = 0$. $C_"fit"$ is the fitted coefficient of the Coulomb term.])[
  #tstyle(table(columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: tstroke, fill: thead-fill, align: (left, right, right, right, right, right, right, right),
    [Process], [poles dev.], [stable], [$c_0(v = 0)$], [extrapolated (TIRust)], [extrapolated (COLLIER)], [rel. diff.], [$C_"fit"$ / $C$],
    ..rm.map(c => (lbl(c), dev-cell(c.max_pole_deviation, target: 1e-9), [#c.stable/#c.npoints],
      dec(c.relmom.at("9").v0, 6), dec(c.relmom.at("9").intercept, 6), dec(c.relmom.at("7").intercept, 6),
      dev-cell(c.relmom.at("9").relative, target: 1e-4),
      [#dec(c.relmom.at("9").coulomb, 5) / #dec(c.relmom.at("9").expected, 5)])).flatten()))
]

= S-wave onium + jet against MG5 2.5.5 <mg255>

The one-loop virtuals of $p p -> eta_c, J\/psi, eta_b, Upsilon + "jet"$ are
compared with the `V0` virtuals of the MG5 2.5.5 onia outputs of
`references/NLO_Onia_Nov2026` and, for $J\/psi$, of its standalone package
`Jpsi_double_cs_NEW`:
- *Channels.* The $g g$, $g u$, $u g$ and $u overline(u)$ channels of
  $eta_c$ and $eta_b$, and the $g g$ channels of $J\/psi$ and $Upsilon$.
- *Settings.* The constituent masses are $m_c = 1.55$ and $m_b = 4.75$ GeV,
  with $n_l = 3$ and 4 light flavours, and $mu_R = 11$ GeV, so the
  $log(mu^2\/m_Q^2)$ terms of the scheme are tested.
- *Scheme.* The 2.5.5 code passes `V0` to MadFKS unchanged (`BinothLHA.f`,
  no DR to CDR conversion), and its `renormalisation_scheme = 1` decouples
  the heavy quark as MadLoop does. Charmonium uses `loop_sm_onia-c_mass`
  with $m_b = m_t = 10^4$ GeV; bottomonium uses `loop_sm_onia-no_tau_mass`
  with massless charm and $m_t = 10^4$ GeV.
- *Normalisation.* Both results are in units of the Born times
  $alpha_s\/2pi$; the 2.5.5 Born is its double pole divided by
  $-sum_i C_i$.
- *Points.* The reference's `PS*.input` points ($J\/psi$: those of
  $g g -> eta_c g$) and three random points at each of three energies;
  MadLoop uses TIRust with its stability checks.
- *Reference evaluation.* The reference is its direct evaluation
  (`calcoption = 3`, OneLOop boxes in quadruple precision). The $J\/psi$
  package splits the virtual into transverse and longitudinal $J\/psi$; their
  sum is compared.

#let mg-target(c) = if c.reference == "direct" { 1e-5 } else { 1e-3 }
#block(breakable: false)[#figure(caption: [MadLoop against the 2.5.5 virtuals at the points MadLoop reports stable (return codes `1xx` to `3xx`): the largest relative difference of $c_0$ and the largest absolute differences of $c_1$, $c_2$, green within $10^(-5)$. "grid" is the 2.5.5 grid against its own direct evaluation; "unstable" counts the points with a soft or low-$p_T$ jet that TIRust refuses (largest $sqrt(s)$ among them).])[
  #tstyle(table(columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: tstroke, fill: thead-fill, align: (left, right, left, right, right, right, right, right, right),
    [Process], [$m_Q$, $n_l$], [ref.], [stable ($sqrt(s)$, GeV)], [$c_0$ dev.], [$c_1$ dev.], [$c_2$ dev.], [grid], [unstable],
    ..mg.map(c => (lbl(c), [#c.mq, #c.nl], [#c.reference],
      [#c.stable (#fixed(c.sqrts_min, 2)–#fixed(c.sqrts_max, 0))],
      dev-cell(c.max_fin, target: mg-target(c)), sci(c.max_c1), sci(c.max_c2),
      if c.grid_vs_direct == none [—] else [#sci(c.grid_vs_direct)],
      if c.unstable == 0 [0] else [#c.unstable (≤ #fixed(c.unstable_sqrts_max, 2))])).flatten()))
] <tab:mg255>]

#figure(caption: [The lowest- and highest-energy stable point of each process: the finite part $c_0$ of MadLoop and of the 2.5.5 reference and their relative difference, and the $1\/epsilon$ coefficient $c_1$ (MadLoop; the reference agrees within the deviation of @tab:mg255).])[
  #tstyle(table(columns: (auto, auto, auto, auto, auto, auto),
    stroke: tstroke, fill: thead-fill, align: (left, right, right, right, right, right),
    [Process], [$sqrt(s)$ (GeV)], [$c_0$ MadLoop], [$c_0$ 2.5.5], [rel. diff.], [$c_1$],
    ..mg.map(c => c.samples.map(s => (lbl(c), fixed(s.sqrts, 2), dec(s.ml_fin, 8), dec(s.ref_fin, 8),
      dev-cell(s.rel, target: mg-target(c)), dec(s.ml_c1, 6))).flatten()).flatten()))
]

#box-note[
  - *Borns.* They agree up to the LDMEs of the two models: the ratio is
    1.000086 for $eta_c$ (0.3867 against 1.16/3), 0.999892 for $eta_b$
    (3.093 against 9.28/3), and 1 for $J\/psi$ and $Upsilon$.
  - *Grids.* The 2.5.5 grids agree with their direct evaluation within
    $10^(-8)$, except for $Upsilon$ ($1.1 dot 10^(-4)$) and the $J\/psi$
    package ($6.8 dot 10^(-4)$); against the grid of the `2.5.5_Onia` output,
    $J\/psi$ agreed within $1.3 dot 10^(-4)$.
  - *Unstable points.* Every point MadLoop reports unstable has a soft or
    low-$p_T$ jet: $p_T < 0.05$ GeV for charmonium, and up to 0.86 GeV in
    $g g -> eta_b\/Upsilon thin g$, which are stable from 1.05 GeV. All of them
    are in the $g g$ and $u overline(u)$ channels; the $q g$ channels are
    stable down to $p_T = 6$ MeV.
  - *What TIRust refuses there.* In both precisions it refuses the Coulomb
    triangle, and the Coulomb boxes and pentagons with the jet attached.
    MadLoop returns code `4x0` with accuracy 1, and its $c_1$ misses $-2 C_F$
    or $-C_F\/2$ of the Born. The pole prediction `ONIA_IR_POLES` agrees with
    the reference at these points.
  - *With TIRust 08edee3* (in PR \#56, not pinned) all 34 such points are served,
    and most agree with the reference. At the softest ones the double-precision
    result is wrong, while MadLoop's tests report an accuracy of $10^(-8)$ to
    $10^(-12)$. For $g g -> Upsilon g$ at $sqrt(s) = 9.51$ GeV (a 10 MeV gluon),
    $c_0 = -333.247$ against $-335.116$. Quadruple precision (`CTModeRun = 4`)
    gives the reference value.
  - *The cause is MadLoop's, not TIRust's.* Substituting all of TIRust's
    arrays with their quadruple-precision values leaves the error unchanged.
    The cause is the R2 counterterms of the gluon self-energy with the $b$ and
    $t$ quarks:
    - with the $10^4$ GeV masses that decouple these quarks here, the terms
      carry $M^2 approx 10^8$;
    - they multiply current contractions that vanish exactly;
    - they leave double-precision rounding, enhanced by the soft gluon.
  - *Scaling with the mass.* The error scales with $M^2$. For
    $g g -> J\/psi thin g$ at $sqrt(s) = 3.1015$ GeV it is:
    - 22% with $m_b = m_t = 10^4$ GeV;
    - $5 dot 10^(-3)$ with $10^3$ GeV;
    - $1.7 dot 10^(-4)$ with the physical masses.
    MadLoop's default stability test (reversing the loop direction) varies
    only the loop part, not the counterterm amplitudes.
  - *The fix: an estimate of the counterterm rounding.* Bound-state outputs
    now add to the double-precision accuracy an estimate of the rounding of
    the counterterm interference: $epsilon sqrt(sum t^2) \/ |c_0|$, over its
    terms $t$.
    - This bounds the real error at every point of the comparison (by 40 to
      $6 dot 10^3$).
    - It costs one multiplication per term.
    - A point beyond `MLStabThres` is rescued in quadruple precision, which
      computes the counterterms in quadruple precision too.
    - With the $10^4$ GeV $b$ and $t$, the four soft points are rescued, and
      every point and polarisation is right (Table 5).
    - With physical masses only the 3.1015 GeV point is rescued (its
      double-precision error is $1.7 dot 10^(-4)$); the other accuracies stay
      near $10^(-9)$.
  - *Rotations are not enough.* MadLoop's rotation of the point
    (`NRotations_DP = 1`, a permutation of the axes) catches the worst point,
    but not the transverse $J\/psi$ alone there, which stays 33% off. It also
    costs about +50% per point. Bound-state outputs leave it at MadLoop's
    default, 0.
]

*$J\/psi$ polarisation.* `Jpsi_double_cs_NEW` gives the virtual of the
transverse and the longitudinal $J\/psi$ in the helicity frame of the
partonic centre-of-mass frame. MadLoop restricts a bound state to chosen
$J_z$ with `ONIA_SELECT_JZ`. Its $J_z$ is the helicity in the frame of the
momenta, here that same frame. Below, T is $J_z = plus.minus 1$ and L is
$J_z = 0$, each in units of its own Born.

#let pol-label = (T: [transverse ($J_z = plus.minus 1$)], L: [longitudinal ($J_z = 0$)], all: [sum])
#figure(caption: [$g g -> J\/psi thin g$ per polarisation against the direct evaluation of `Jpsi_double_cs_NEW` (largest relative deviation of $c_0$, absolute of $c_1$, each in units of its own Born):
- MadLoop with the pinned TIRust at the points it reports stable;
- MadLoop with TIRust 08edee3 in forced quadruple precision (`CTModeRun = 4`) at all 16 points, including the soft ones;
- the same with MadLoop's stability checks, before the counterterm estimate ("08edee3 before": every point reported stable, code 213) and with it ("now": the soft points rescued in quadruple precision, codes #mgp.at(0).ct_codes.map(str).join("/")).])[
  #tstyle(table(columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: tstroke, fill: thead-fill, align: (left, right, right, right, right, right, right, right),
    [Polarisation], [Born ratio], [pinned: stable], [$c_0$ dev.], [$c_1$ dev.], [08edee3 QP: $c_0$, $c_1$ dev.], [08edee3 before: $c_0$ dev.], [now: $c_0$ dev.],
    ..mgp.map(c => (pol-label.at(c.pol), [#fixed(c.born_ratio.at(0), 9)–#fixed(c.born_ratio.at(1), 9)],
      [#c.stable/#c.npoints], dev-cell(c.pin_fin, target: 1e-5), sci(c.pin_c1),
      [#sci(c.qp_fin), #sci(c.qp_c1)],
      table.cell(fill: acc-fill(c.dp_fin, 1e-5))[#sci(c.dp_fin)],
      dev-cell(c.ct_fin, target: 1e-5))).flatten()))
] <tab:jpsipol>

The longitudinal Born is suppressed, so its double-precision value carries
MadLoop's rounding of $10^(-7)$ relative to it; in quadruple precision every
polarisation agrees to $3 dot 10^(-12)$. The red column is the double-precision
rounding of the R2 counterterms of the $10^4$ GeV $b$ and $t$ quarks at the soft
points (above): at $sqrt(s) = 3.1015$ GeV the longitudinal $c_0$ was off by 440%
while reported stable. With the counterterm estimate every point is right.
Those soft points are rescued in quadruple precision.

= P-wave bound states <pwave>

The P-wave results use TIRust 48958d1 (in huashengshao/TIRust PR \#56, stacked on
\#47 and \#55, not yet the pinned revision). That revision serves every integral
of the $2 -> 1$ processes at zero relative momentum, including the Coulomb box
with its gluon line doubled. Earlier revisions refuse that box, and the point is
then reported unstable (`H = 4`). The settings are $mu = M = 2 m_c = 3.1$ GeV,
$n_l = 3$, four points per process (2 → 1: the rest frame and longitudinal
boosts), in units of the Born times $alpha_s\/2pi$, with the literature values
converted by section B.5 of `research/literature.md`.

#block(breakable: false)[#figure(caption: [P-wave $2 -> 1$ processes against the literature: "dev." is the largest deviation over the points (relative for $c_0$, absolute for the poles), green within $10^(-8)$, or $10^(-7)$ for the 8-digit $gamma g$ values; "poles" is the largest deviation of $c_1$, $c_2$ from the prediction of `ONIA_IR_POLES`, whose $P -> S$ term is given in the last column.])[
  #tstyle(table(columns: (auto, 1fr, auto, auto, auto, auto, auto, auto, auto),
    stroke: tstroke, fill: thead-fill, align: (left, left, right, right, right, right, right, right, center),
    [Process], [Reference], [$c_0$ ref.], [$c_0$ MadLoop], [$c_1$, $c_2$], [dev.], [poles], [$P -> S$], [stable],
    ..pw.two_one.map(c => (lbl(c), text(7pt, eval(c.source, mode: "markup")), dec(c.reference.at(0), 8),
      dec(c.finite, 8), [#dec(c.single, 4) #linebreak() #dec(c.double, 4)],
      dev-cell(c.max_deviation, target: c.target), dev-cell(c.max_pole_deviation, target: 1e-9),
      if calc.abs(c.ps_term.at(0)) < 1e-12 and calc.abs(c.ps_term.at(1)) < 1e-12 [0] else [#dec(c.ps_term.at(0), 6)],
      [#c.stable/#c.npoints])).flatten()))
] <tab:pwave>]

*The $P -> S$ pole.* A soft gluon turns a P-wave state into the S-wave states
$S'$ of colour $C'$: $P^((1)) -> S^((8))$, and $P^((8)) -> S^((1)), S^((8))$
(arXiv:2402.19221, appendix B). Their amplitudes are the values of the Born
amplitudes at $q = 0$ with the spin projectors of $S_z = J_z - L_z$, and
`loop_onia.pwave_soft_current` generates their colour. The single pole the
transitions add to the virtual has three parts:
- *Cross vectors,* $chevron.l P|T_k dot Q_"eff"|S' chevron.r$ of the coloured
  legs $k$. For equal masses, $Q_"eff" = 2 (T_Q - T_(overline(Q)))$, the
  derivative of the heavy quarks' soft functions in $q$.
- *Self term,* $-2\/(3 P^2) sum_(S') C_"eff" (S') thin g_(mu nu) M^(mu nu)(S')$,
  quadratic in the $S'$ amplitudes.
- *LDME counterterm,* $-8\/(m_Q m_(overline(Q))) sum_(S') c_(S') M(r_(S'))$, the
  pole absorbed into the long-distance matrix element of $S'$.

The formula is that of `getpoles_onia` on `onia_fks` (`ONIA_PW_SOFT_POLE`). For
equal masses the self term and the LDME counterterm cancel. The cross vectors
vanish by colour conservation when the bound state has a single coloured
partner, as in $e^+ e^- -> c overline(c)[P^((8))] thin g$, or when every $S'$ Born
vanishes, as in the $2 -> 1$ processes but $g g -> c overline(c)[""^1 P_1^((8))]$.

#block(breakable: false)[#figure(caption: [The poles of $2 -> 2$ P-wave processes against the prediction, TIRust 48958d1 ($g g -> chi_(c 0) thin g$: 370b632), four points at the given $sqrt(s)$: the range of $c_1$ over the points, the range of the $P -> S$ term of the prediction, and the largest deviation of $c_1$, $c_2$ from the prediction (green within $10^(-8)$). The finite parts have no reference.])[
  #tstyle(table(columns: (auto, auto, auto, auto, auto, auto, auto),
    stroke: tstroke, fill: thead-fill, align: (left, right, right, right, right, right, center),
    [Process], [$sqrt(s)$ (GeV)], [$c_2$], [$c_1$], [$P -> S$ term], [poles dev.], [stable],
    ..pw.two_two.map(c => (lbl(c), fixed(c.energy, 0), dec(c.double, 4),
      [#dec(c.single_range.at(0), 4) … #dec(c.single_range.at(1), 4)],
      if calc.abs(c.ps_term.at(0)) < 1e-9 and calc.abs(c.ps_term.at(1)) < 1e-9 [0 (≤ #sci(calc.max(calc.abs(c.ps_term.at(0)), calc.abs(c.ps_term.at(1)))))]
      else [#dec(c.ps_term.at(0), 2) … #dec(c.ps_term.at(1), 2)],
      dev-cell(c.max_pole_deviation, target: 1e-8), [#c.stable/#c.npoints])).flatten()))
] <tab:pwave22>]

#box-note[
  - *Derivative.* At a relative momentum $q_0 != 0$, where every integral is
    an ordinary one, the dual derivative, doubled lines included, agrees with
    finite differences: 2e-14 to 7e-13 on the finite parts of
    $u overline(u) -> "Ps"[""^3 P_1]$, $gamma gamma -> chi_(c 0)$ and
    $g g -> chi_(c 0)$, in double and in quadruple precision
    (`research/pwave_fd`).
  - *Quadruple precision.* It agrees with double precision to
    $10^(-14)$–$4 dot 10^(-13)$. A rescue forced by a tight `MLStabThres`
    returns code 323.
  - *Refused integrals.* When TIRust refuses an integral with a doubled
    line, the point is reported unstable in every `CTModeRun` mode, with a
    warning. TIRust 48958d1 serves $u overline(u) -> chi_(c 0) g$ and
    $g u -> chi_(c 0) u$, but not the doubled-gluon pentagons of
    $g g -> chi_(c 0) g$ (equal-mass heavy bubbles at zero momentum with
    raised powers). TIRust 370b632 (in PR \#56, not yet pinned)
    serves them: every point stable (code 213), the poles as predicted to
    $1.3 dot 10^(-9)$, the finite parts equal to forced quadruple precision
    within $4 dot 10^(-12)$, at 0.9 s per point.
]

== An S-wave and a P-wave state: $e^+ e^- -> J\/psi thin chi_(c J)$ <sp>

#let sp = pw.at("sp", default: ())
#let spx = pw.at("sp_exact", default: ())
#let sp-dp(c) = c.modes.at("-1")
#let sp-qp(c) = c.modes.at("4")
#let sp-lbl(c) = text(8.4pt, font: "Libertinus Serif", $J\/psi thin chi_(c #c.J)$)
A process may have one P-wave state together with S-wave states.
$e^+ e^- -> J\/psi thin chi_(c J)$ ($J = 0, 1, 2$; photon exchange only, no QED
fragmentation, $b$ and $t$ decoupled) has a published one-loop reference.
- *Its NLO correction is the virtual alone.* The final state is exclusive.
- *The virtual is UV and IR finite.* The S-wave $J\/psi$ removes the
  non-factorisable soft exchange of the P wave, so `ONIA_IR_POLES` predicts
  zero poles.
- *Wang, Ma and Chao* (arXiv:1107.2646) give the NLO cross sections in closed
  form.
- *Dong, Feng and Jia* (arXiv:1107.4351) give the helicity amplitudes and the
  cross section of each helicity. Both papers agree with each other and
  correct the earlier $K = 2.8$ for $chi_(c 0)$.

Born and virtual have the form $alpha + beta cos^2 theta$ (one photon), so
four Gauss–Legendre nodes in $cos theta$ give the exact angular average of the
finite part, $chevron.l c_0 chevron.r = integral B c_0 \/ integral B$.

The references are in the $n_f = 4$ scheme with the charm active. In
MadLoop's scheme ($n_l = 3$, charm decoupled) the average is
$2 (A_J + 2 T_F\/3 ln(mu^2\/m_c^2))$, where
$sigma_"NLO" = sigma_"LO" (1 + alpha_s A_J\/pi)$ (`research/literature.md`,
C.6).

The closed form is converted mechanically from the paper's LaTeX source
(`research/pwave_fd/psi_chic_ref.py`). It reproduces:
- the K factors of both papers, e.g. 1.573, 0.911 and 0.784 at $mu = 2 m_c$
  against 1.57, 0.91 and 0.78;
- the LO ratios between the $J$ to all digits.

The settings are $sqrt(s) = 10.58$ GeV, $m_c = 1.5$ GeV and $mu = 2 m_c$,
with TIRust 08cca00 (huashengshao/TIRust\#58, stacked on \#56, not yet
pinned). The installation of the pinned TIRust has no raised powers, and
MadGraph refuses the P-wave output.

#block(breakable: false)[#figure(caption: [The angular average of the finite part against Wang, Ma and Chao, in double precision with MadLoop's stability test (`CTModeRun` −1, threshold $10^(-3)$; every node returns 213) and in quadruple precision (`CTModeRun` 4; 120). "dev." is relative, green within $10^(-6)$, the target of the other $e^+ e^-$ references (in double precision: or MadLoop's accuracy estimate, "acc.", the largest over the nodes, when that is larger). "poles" is the largest $|c_1|$, $|c_2|$ over the nodes in double (top) and quadruple precision; the prediction is 0.])[
  #tstyle(table(columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
    column-gutter: 3pt,
    stroke: tstroke, fill: thead-fill, align: (left, right, right, right, right, right, right, right, right),
    [$e^+ e^- ->$], [$chevron.l c_0 chevron.r$ ref.], [DP], [dev.], [acc.], [QP], [dev.], [poles], [s/pt DP, QP],
    ..sp.map(c => (sp-lbl(c), dec(sp-dp(c).c0_reference, 8), dec(sp-dp(c).c0, 8),
      table.cell(fill: acc-fill(sp-dp(c).c0_relative_deviation, calc.max(1e-6, sp-dp(c).max_accuracy)))[#sci(sp-dp(c).c0_relative_deviation, d: 2)],
      sci(sp-dp(c).max_accuracy), dec(sp-qp(c).c0, 8),
      table.cell(fill: acc-fill(sp-qp(c).c0_relative_deviation, 1e-6))[#sci(sp-qp(c).c0_relative_deviation, d: 2)],
      [#sci(sp-dp(c).max_pole) #linebreak() #sci(sp-qp(c).max_pole)],
      [#fixed(sp-dp(c).ms_per_point / 1000, 2) #linebreak() #fixed(sp-qp(c).ms_per_point / 1000, 1)])).flatten()))
] <tab:sp>]

#block(breakable: false)[#figure(caption: [Helicity content. $lambda = lambda_(J\/psi) - lambda_chi$: $lambda = 0$ goes with $sin^2 theta$ and $|lambda| = 1$ with $1 + cos^2 theta$. The LO share of $lambda = 0$ is compared with the helicity amplitudes of Dong et al. (their eq. (3.3)). The coefficient $A$ of each group ($n_f = 4$, units of $alpha_s\/pi$) is compared with their table 2, which has two to three digits; the MadLoop value is from quadruple precision.])[
  #tstyle(table(columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: tstroke, fill: thead-fill, align: (left, right, right, right, right, right, right, right),
    [$e^+ e^- ->$], [LO share $lambda = 0$], [Dong et al.], [dev.], [$A(lambda = 0)$], [Dong et al.], [$A(|lambda| = 1)$], [Dong et al.],
    ..sp.map(c => (sp-lbl(c), dec(sp-qp(c).lo_shares.at(0), 8), dec(sp-qp(c).lo_shares_reference.at(0), 8),
      dev-cell(sp-qp(c).lo_share_deviation, target: 1e-12),
      dec(sp-qp(c).group_A_nf4.at(0), 3), dec(sp-qp(c).group_A_dfj_table2.at(0), 2),
      dec(sp-qp(c).group_A_nf4.at(1), 3), dec(sp-qp(c).group_A_dfj_table2.at(1), 2))).flatten()))
] <tab:sphel>]

#box-note[
  - *Poles.* Both poles vanish, as predicted: at most
    #sci(calc.max(..sp.map(c => sp-dp(c).max_pole))) in double precision and
    #sci(calc.max(..sp.map(c => sp-qp(c).max_pole))) in quadruple precision.
  - *Double precision.* Every node is stable (code 213), and the averages are
    within $1.1 dot 10^(-6)$ of the reference. For $chi_(c 2)$ at the forward and
    backward nodes, double precision differs from quadruple precision by up to
    $5 dot 10^(-6)$, while MadLoop's loop-reversal estimate there is
    $5 dot 10^(-7)$. This is far inside the stability threshold.
  - *Quadruple precision.* It is not better than $10^(-8)$–$10^(-7)$:
    - The effect is not the double-precision momenta. At $sqrt(s) = 18.5$ GeV,
      with momenta exact in binary64 (`psi_chic.py --exact`), the averages
      deviate by #spx.map(c => sci(sp-qp(c).c0_relative_deviation)).join(", "),
      and the virtual departs from the form $alpha + beta cos^2 theta$ by up to
      #sci(calc.max(..spx.map(c => sp-qp(c).form_residual_virtual))). The Born
      keeps the form to $2 dot 10^(-15)$.
    - The two quadruple-precision directions (`CTModeRun` 4 and 5) are
      identical bit for bit, so the rescue's accuracy estimate is 0.
    - `MP_TIRUST_LOOP` returns each loop group to MadLoop as a double-precision
      number. Both directions round to the same values, so the comparison
      cannot see an error common to both.
    - Where the $10^(-8)$ comes from has not been traced further.
  - *Helicities.* The LO shares equal the helicity amplitudes of Dong et al. to
    $2 dot 10^(-15)$. Their table's NLO corrections agree within its rounding.
]

= Reduction libraries at threshold <reducers>

#let caught = rgb("#f8d4cc")
#let state-fill(st) = if st == "ok" { good } else if st == "flagged" { warm } else if st == "guarded" { caught } else if st == "silent" { rgb("#d6604d") } else { luma(240) }
#let lib-keys = red.at(0).libs.keys()
#figure(caption: [The same points with each reduction library alone (MadLoop's stability checks on). Each cell gives the largest deviation from the reference (the analytic $c_0$ and the predicted poles; for the $2 -> 2$ processes without an analytic result TIRust's value at the same point, validated in @twotwo), the return codes (`2xx` stable, `4xx` unstable or exceptional) and the median time per point. Green: every point agrees within $10^(-6)$; amber: MadLoop's stability tests catch every wrong point; red: a wrong point passes MadLoop's tests and is reported unstable only because a bound-state result of a library other than TIRust always is (the guard of `onia_loop.f`); dark red: a wrong point is reported stable; grey: no result within 120 s or a crash. Before that guard, the red cells were reported stable.])[
#tstyle(table(columns: (auto,) + lib-keys.map(_ => 1fr),
  stroke: tstroke, fill: thead-fill, align: left,
  [Process], ..lib-keys.map(k => [#meta.libnames.at(k)]),
  ..red.map(c => (lbl(c), ..lib-keys.map(k => {
    let r = c.libs.at(k, default: none)
    if r == none { [—] }
    else if r.status != "ok" { table.cell(fill: luma(240))[#tiny[#r.status]] }
    else { table.cell(fill: state-fill(r.state))[
      #if r.max_deviation != none [#sci(r.max_deviation)] else [—] #linebreak()
      #tiny[#r.return_codes.map(str).join(",") · #ms2(r.time_ms) ms]] }
  }))).flatten()))
] <tab:reducers>

#box-note(fill: warm)[
  *Only TIRust reduces the threshold integrals.* The other libraries return
  wrong values, and MadLoop's stability tests (rotation, loop-direction
  reversal, COLLIER's internal estimate) do not catch all of them: the
  same degenerate integrals come out the same wrong way in every test
  configuration. A bound-state output therefore uses TIRust alone by default,
  and its `SLOOPMATRIX` reports the result of any other library as unstable
  (return code `4x0`), whatever its estimated accuracy; the libraries remain
  selectable for cross-checks. The relative-momentum diagnostic, whose
  integrals are ordinary ones, and leptonia, where no QCD loop holds both
  constituents, are exempt. Separately, IREGI drops the loop of the
  leptonium process at its scale ($M = 2 m_e$; also at $M = 20$ MeV, while it is
  right at 2 GeV) and reports it stable: a limitation of IREGI at small
  scales, outside the default chain. Open TIRust items (reproducers in
  `research/tirust_qp_repro/`):
  (1) the binary128 path refuses the degenerate lanes even with exact inputs,
  so there is no QP rescue for bound states yet;
  (2) processes with two bound states ($e^+ e^- -> J\/psi eta_c$) meet refused
  lanes (a threshold bubble pinch; two Coulomb relations in one pentagon),
  served by TIRust 370b632 (PR \#56, not yet pinned; @analytic);
  (3) P-wave states need the doubled-propagator integrals at threshold:
  TIRust 48958d1 has them for the $2 -> 1$ processes (@pwave), not yet pinned.
]

= Stability profiles <stability>

These profiles follow appendix C of arXiv:1604.01363 (Hirschi, Peraro), which
characterises MadLoop's reductions by its own double-precision accuracy
estimate over random phase-space points.
- *Points.* Random points with $p_T > p_T^"min"$ for every final-state particle
  and $Delta R > 0.5$ between them.
- *Accuracy.* For each point, MadLoop's estimate combines its two tests: the
  reversed loop direction, a pass in which TIRust also moves each invariant
  by one unit in its last place, and one Lorentz rotation of the point
  (`NRotations_DP = 1`). For bound states it is at least the counterterm
  rounding estimate (@mg255).
- *No rescue.* The quadruple-precision rescue is switched off
  (`MLStabThres` $= 10^(99)$), so what is recorded is the double-precision
  estimate. A refused integral sets it to 1.
- *Setups.* Setup A is the paper's, $sqrt(s) = 1$ TeV with $p_T > 50$ GeV;
  setup B is a quarkonium-scale one, $sqrt(s) = 20$ GeV with $p_T > 2$ GeV.
- *Samples and models.* $10^4$ points per $2 -> 2$ process (the paper used
  $10^5$) and $2 dot 10^3$ for each P-wave colour singlet, with physical masses:
  - charmonium with `loop_sm_onia-c_mass` ($m_c = 1.55$ GeV, massive $b$ and
    $t$, $n_l = 3$);
  - bottomonium with `loop_sm_onia-no_tau_mass` (massless charm,
    $m_b = 4.7$ GeV, $m_t = 173$ GeV, $n_l = 4$).

  The $u overline(u)$ P waves and the P-wave colour octets use TIRust 48958d1. The $g g$ P waves,
  $g g -> chi_(c 0) thin g$ and $g g -> chi_(b 0) thin g$, use TIRust 370b632
  (PR \#56), because 48958d1 refuses their doubled-gluon pentagons. Every
  process but the reference is also run with TIRust 5025e12 (head of PR \#56;
  not pinned) and a bridge patch for its _relation tiers_ and a binary64 gate.
  Every P wave is run a third time with the _80-bit switch_ on
  (`#PWaveExtendedPrecision`, @ep): TIRust d17085a (PR \#56) with the
  bridge patch v2, which adds low words and extended inputs to the gate and
  the relation tiers. Neither is in MadGraph yet.
- *P-wave colour octets.* $e^+ e^- -> Q overline(Q) thin [n^((8))] thin g$ for
  $n = ""^3 P_0, ""^3 P_1, ""^3 P_2, ""^1 P_1$, the charmonium octets of the
  $2 -> 2$ pole checks (@tab:pwave22) and their bottomonium analogues, in a
  figure of their own (@fig:stability-octets).
- *Reference.* $g g -> t overline(t) thin g$, of the paper's first process
  class, with MadLoop's default chain, as a reference.

#figure(caption: [Fraction of the points whose double-precision accuracy estimate is at least $Delta$ (cumulative, logarithmic axes), charmonium top, bottomonium bottom. Colour: the process (a bottomonium process has that of its charmonium analogue). Line style: the TIRust build, solid the pinned one (S waves), 48958d1 or 370b632 (P waves), dotted 5025e12 (PR \#56) with the bridge's gate and relation tiers, dash-dotted the P waves with the 80-bit switch (@ep); dashed: the reference. The vertical line marks $Delta = 10^(-3)$, MadLoop's rescue threshold (`MLStabThres`).])[
  #image("figures/stability.svg", width: 100%)
] <fig:stability>

#figure(caption: [As @fig:stability, for the P-wave colour octets $e^+ e^- -> Q overline(Q) thin [n^((8))] thin g$ with TIRust 48958d1 (solid), 5025e12 with the bridge's gate and relation tiers (dotted, PR \#56) and with the 80-bit switch (dash-dotted, @ep), charmonium (top) and bottomonium (bottom). The colours here mark the state; at 1 TeV the $""^3 P_0^((8))$ and $""^3 P_2^((8))$ curves lie on top of each other. Dashed: the reference process.])[
  #image("figures/stability_octets.svg", width: 100%)
] <fig:stability-octets>

#figure(caption: [The profiles in numbers (the table view of @fig:stability and @fig:stability-octets): the reducer, the number of points, the fractions of points with an estimate of at least $10^(-3)$ and $10^(-6)$, the median estimate and the median time per point with the three double-precision evaluations of the test.])[
  #tstyle(table(columns: (auto, auto, 1fr, auto, auto, auto, auto, auto),
    stroke: tstroke, fill: thead-fill, align: (left, center, left, right, right, right, right, right),
    table.header([Process], [setup], [reducer], [points], [$>= 10^(-3)$], [$>= 10^(-6)$], [median], [ms/pt]),
    ..stab.map(c => (text(8.4pt, font: "Libertinus Serif", eval(c.label, mode: "math")), [#c.setup], text(7pt)[#c.reducer], [#c.npoints],
      table.cell(fill: if c.frac_1e3 <= 0.01 { good } else { warm })[#fixed(100 * c.frac_1e3, 2)%],
      [#fixed(100 * c.frac_1e6, 2)%], sci(c.median), ms2(c.ms))).flatten()))
] <tab:stability>

#box-note[
  - *Refusals of the pinned TIRust.* The plateaus of the S-wave processes
    are the same points in all three processes of a family, reported
    unstable with estimate 1 because the pinned TIRust refuses integrals
    there. They are 3.0% (charmonium) and 3.0% (bottomonium) in setup A, and
    1.2% and 1.7% in setup B.
    - Most are forward jets, $|cos theta| > 0.95$. In setup A this means
      every point with $p_T < 100$ GeV and 44% of those between 100 and
      150 GeV (charmonium).
    - In setup B, bottomonium also has refusals around $theta = 90 degree$,
      where $t approx u$: 16% to 42% of the points with $|cos theta| < 0.05$.
    - TIRust 5025e12 (PR \#56) serves all of them. With the bridge's binary64
      gate no S-wave point is then above $10^(-6)$, in either setup, and
      every point is stable.
  - *S waves otherwise.* Apart from the refusals, the S-wave profiles are
    steep: the median estimate is $10^(-13)$ to $10^(-8)$, and no point is
    between $10^(-3)$ and 1.
    - With the pinned TIRust, the $J\/psi$ and $Upsilon$ processes have a
      tail between $10^(-6)$ and $10^(-3)$: 1.4% and 1.3% of the points at
      1 TeV, at most 0.03% at 20 GeV.
    - The tail is a true double-precision error. At 16 of its points
      ($g g -> J\/psi thin g$, 1 TeV), against quadruple precision,
      5025e12 without the gate is off by $1.2 dot 10^(-8)$ to
      $3.2 dot 10^(-6)$ (MadLoop's estimate is at least twice that), and the
      pinned TIRust by as much at ten of them and by $1.1$ to
      $1.7 dot 10^(-5)$ at five. With the gate, 5025e12 is off by at most
      $2.3 dot 10^(-10)$.
    - It is the bridge's binary64 gate (patch 0006, MadGraph7\#528) that
      removes it, not the TIRust revision. It sends lanes whose conditioning
      is below $10^(-10)$ (TIRust's default: $10^(-12)$) to the double-double
      tier. Without it, 5025e12 keeps the tail: 1.8% ($J\/psi$) and 1.7%
      ($Upsilon$) of the points at 1 TeV are above $10^(-6)$, the largest at
      $10^(-4)$. In production settings the gate costs nothing measurable here
      ($g g -> J\/psi thin g$: 88 ms per point with it, 89 without).
  - *The P-wave singlets at 1 TeV.* At $sqrt(s) = 1$ TeV the estimate is above
    $10^(-3)$ at 75% of the $u overline(u) -> chi_(c 0) thin g$ points, 40% of the
    $g g -> chi_(c 0) thin g$ points, 8.8% of the $u overline(u) -> chi_(b 0) thin g$
    points and 0.2% of the $g g -> chi_(b 0) thin g$ points. At 20 GeV no singlet
    point is.
    - The derivative of the threshold amplitude cancels the more strongly
      the smaller $M\/sqrt(s)$ is: $3 dot 10^(-3)$ for $chi_(c 0)$ against
      $10^(-2)$ for $chi_(b 0)$.
    - *Diagnosis* (24 points each, `research/pwave_stability`): against
      quadruple precision, 4/24 ($u overline(u)$) and 1/24 ($g g$) are truly above
      $10^(-3)$ (largest $3.4 dot 10^(-2)$ and $3.9 dot 10^(-2)$).
    - Replacing only the double-precision TIRust pentagons with a doubled
      propagator by their quadruple-precision values brings the ten points
      tested to $6 dot 10^(-7)$–$2.4 dot 10^(-5)$. These pentagons have the heavy line
      between the two constituents doubled, and a vanishing Gram determinant
      (both constituents carry $P\/2$). Their arrays are off by only
      $5 dot 10^(-12)$, but the P-wave sum amplifies that by about $10^8$.
    - The reversed-loop pass meets the same pentagons less accurately (up to
      2.3 off), so MadLoop's estimate is pessimistic as well. With accurate
      pentagons both passes agree to about $10^(-6)$.
    - In production these points are rescued in quadruple precision. Sending
      these lanes from MadLoop through TIRust's quadruple-precision path works
      for $u overline(u) -> chi_(c 0) g$ (largest error $5 dot 10^(-5)$), but not
      for $g g -> chi_(c 0) g$: its threshold relations hold only to
      double-precision rounding, and that path refuses them.
  - *The fix in TIRust: relation tiers* (PR \#56, at 5025e12; dotted in
    @fig:stability). Lanes
    that TIRust serves through an affine relation (a constant identity or a
    Coulomb partial fraction) are evaluated again in double-double, from the
    same double-precision inputs. The bridge enables this for calls with
    raised powers only, i.e. for the P waves.
    - No P-wave point is then above $10^(-3)$. At 1 TeV the median estimate is
      $3.9 dot 10^(-5)$ ($u overline(u) -> chi_(c 0) thin g$), $8.1 dot 10^(-6)$
      ($g g -> chi_(c 0) thin g$), $3.1 dot 10^(-7)$ ($u overline(u) -> chi_(b 0) thin g$)
      and $8.0 dot 10^(-8)$ ($g g -> chi_(b 0) thin g$); the largest is
      $2.3 dot 10^(-4)$.
    - Against quadruple precision at the same 24 points, the largest error
      drops from $3.4 dot 10^(-2)$ to $2.7 dot 10^(-5)$ ($u overline(u)$) and from
      $3.9 dot 10^(-2)$ to $4.2 dot 10^(-6)$ ($g g$), the level the
      quadruple-precision pentagons gave above. The reversed pass agrees
      (largest $4.3 dot 10^(-5)$), and MadLoop's estimate is at least 1.8 times
      the true error at every one of these points.
    - Cost (paired runs with 14f2696, which costs the same as 8b436f9 within
      3%; MadLoop's default settings): at 1 TeV no point needs
      the quadruple-precision rescue any more, and the mean time per point
      falls from 3.5 s to 0.10 s ($u overline(u) -> chi_(c 0) thin g$) and from
      18 s to 1.2 s ($g g -> chi_(c 0) thin g$). At 20 GeV, where nothing was
      rescued, the double-double lanes add 49% ($u overline(u)$: 66 to 99 ms) and
      18% ($g g$: 0.92 to 1.08 s).
    - *What is left* at 1 TeV is the double-precision floor. Two tests at the
      24 points (`research/pwave_stability`) show that rounding only the
      integrals to double, with MadLoop in quadruple precision, leaves the
      error as it is. So does rounding only MadLoop's loop coefficients,
      with exact doubled-line integrals and contractions. Lowering it needs
      both beyond double precision. The quadruple-precision rescue does it
      at about 40 times the time; the opt-in 80-bit switch of @ep does it at
      5 to 6 times. At the default threshold no point needs the rescue.
    - The P-wave checks of @pwave are unchanged with it: the same finite parts
      within $10^(-10)$, and the poles as predicted to $5 dot 10^(-11)$.
    - Without the gate, 5025e12, 8b436f9 and 14f2696 (the commit that
      introduced the relation tiers) give the same results: bit-identical but
      for estimates of about $10^(-12)$ in the $g g$ channels at 20 GeV. The
      gate leaves the P-wave checks bit-identical and the errors at the 24
      points the same; it only lowers the largest $g g$ estimate at 20 GeV,
      from $9 dot 10^(-7)$ to $3 dot 10^(-8)$.
  - *The P-wave octets* (@fig:stability-octets). No point of
    $e^+ e^- -> Q overline(Q) thin [n^((8))] thin g$ has an estimate above
    $10^(-4)$, in either setup. The largest is $2.2 dot 10^(-5)$, for
    $c overline(c) thin [""^1 P_1^((8))]$ at 1 TeV. These processes have boxes
    with a doubled propagator, but no pentagons.
    - At 1 TeV the $""^3 P_0$, $""^3 P_2$ and $""^1 P_1$ octets have a median
      estimate of $1.3$ to $1.5 dot 10^(-6)$ (charmonium) and $2.4 dot 10^(-8)$
      (bottomonium). The $""^3 P_1$ octet stays below $10^(-10)$.
    - Against quadruple precision (24 points of
      $e^+ e^- -> c overline(c) thin [""^3 P_0^((8))] thin g$ at 1 TeV,
      `research/pwave_stability`), the true error is at most
      $7.7 dot 10^(-8)$. The estimate comes from the reversed-loop pass,
      which is off by $1.5 dot 10^(-7)$ to $2.5 dot 10^(-6)$, so it is
      pessimistic here as well.
    - At 20 GeV every estimate is below $2 dot 10^(-10)$.
    - With TIRust 5025e12 the charmonium octets at 1 TeV drop to a
      median of $7 dot 10^(-8)$ to $10^(-7)$, with no point at or above
      $10^(-6)$, and the bottomonium ones to $10^(-9)$. The cost is about 20%
      ($c overline(c) thin [""^3 P_0^((8))]$: 25 to 30 ms per point).
  - *The reference.* For $g g -> t overline(t) thin g$, 0.04% of the points are
    above $10^(-3)$, which is the steep profile of the paper's first process
    class.
]

== The 80-bit switch <ep>

`#PWaveExtendedPrecision .TRUE.` in `MadLoopParams.dat` (off by default)
runs the double-precision passes of an output with one P-wave state in
extended precision. That is the x87 80-bit format, 64 bits of mantissa
(binary128 on hosts without it). The PS point, the dual wavefunctions, the
loop coefficients, the projections and the contractions are computed in it.
TIRust's double-precision entry takes the integral inputs (shifts and
invariants) as binary64 hi + lo words and returns the result as hi + lo.
- *Generated code.* The extended code is MadLoop's quadruple-precision code
  with its names and kind changed. `madgraph/loop/onia_ep.py` derives it
  when the output is written; only the dispatch and the TIRust routines are
  in the templates. Outputs without a P-wave state are unchanged but for the
  card parameter.
- *TIRust.* The code is written only for an installation whose metadata
  has `raised_power_extended_inputs`. Here that is TIRust d17085a (PR \#56)
  with the bridge patch v2 (gate, relation tiers, low words, extended
  inputs). The bridge is not in MadGraph yet.
- *Check pass.* In its check pass (the reversed loop of MadLoop's
  stability test) TIRust moves the invariants by $2^(-52)$, one binary64
  unit. With extended inputs the bridge v2 moves them by
  $2^(-(b-1))$, $2^(-63)$ for $b = 64$ bits. With $2^(-52)$ the reversed
  pass kept a binary64 input error. MadLoop's mean of the two passes then
  gained only about one digit: 18 times ($u overline(u)$) and 24 times
  ($g g$) in the worst point.

#figure(caption: [The 80-bit switch at the 24 points of `research/pwave_stability` ($sqrt(s) = 1$ TeV), against quadruple precision: the largest and the median error of the finite part and the time per point, with `CTModeRun -1` (MadLoop's stability test, no rescue) or `CTModeRun 1` (one pass).])[
  #tstyle(table(columns: (1fr, auto, auto, auto, auto),
    stroke: tstroke, fill: thead-fill, align: (left, right, right, right, right),
    table.header([Mode], [$u overline(u) -> chi_(c 0) thin g$], [s/pt], [$g g -> chi_(c 0) thin g$], [s/pt]),
    [double precision, `CTModeRun -1`], [$2.9 dot 10^(-5)$ / $5.3 dot 10^(-6)$], [0.11], [$4.1 dot 10^(-6)$ / $1.1 dot 10^(-6)$], [1.01],
    [80-bit, one pass (`CTModeRun 1`)], [$2.2 dot 10^(-8)$ / $4.1 dot 10^(-9)$], [0.50], [$2.2 dot 10^(-9)$ / $6.0 dot 10^(-10)$], [5.10],
    [80-bit, `CTModeRun -1`], [$2.0 dot 10^(-8)$ / $2.5 dot 10^(-9)$], [0.54], [$2.3 dot 10^(-9)$ / $6.2 dot 10^(-10)$], [5.32],
    [80-bit, `CTModeRun -1`, check pass $2^(-52)$], [$1.6 dot 10^(-6)$ / $9.8 dot 10^(-8)$], [0.54], [$1.7 dot 10^(-7)$ / $4.1 dot 10^(-8)$], [5.45]))
] <tab:ep-accuracy>

#figure(caption: [Cost of the 80-bit switch: paired alternating runs of two copies of one output (TIRust d17085a with the bridge patch v2), off and on, in MadLoop's default settings (`CTModeRun -1`, rescue at $10^(-3)$, no rotation); median of the batch medians of the time per point, four pairs.])[
  #tstyle(table(columns: (1fr, auto, auto, auto, auto, auto),
    stroke: tstroke, fill: thead-fill, align: (left, center, right, right, right, right),
    table.header([Process], [$sqrt(s)$], [points], [off, ms], [on, ms], [ratio]),
    [$u overline(u) -> chi_(c 0) thin g$], [1 TeV], [200], [94.6], [548], [5.8],
    [$u overline(u) -> chi_(c 0) thin g$], [20 GeV], [200], [94.3], [544], [5.8],
    [$g g -> chi_(c 0) thin g$], [1 TeV], [48], [1029], [5519], [5.4],
    [$g g -> chi_(c 0) thin g$], [20 GeV], [48], [1036], [5508], [5.3]))
] <tab:ep-cost>

#box-note[
  - *Accuracy* (@tab:ep-accuracy). One 80-bit pass reproduces the prototype
    of `research/pwave_stability` to the digit. MadLoop's stability test
    keeps it: about three digits more than double precision at 1 TeV.
  - *MadLoop's estimate* behaves as in double precision: an estimate, not a
    bound, and never 0. At the 24 points it is below the true error at 5
    points of each process, by at most a factor 4 ($u overline(u)$) and 3
    ($g g$). In double precision that happens at 3 and 1 points, by up to
    3 and 10.
  - *Cost* (@tab:ep-cost). 5.3 to 5.8 times double precision, at both
    energies, with no point rescued in quadruple precision. The quadruple
    precision rescue of the whole point costs about 40 times.
  - *Profiles* (dash-dotted in @fig:stability and @fig:stability-octets, all
    P waves, both setups): no P-wave point has an estimate at or above $10^(-7)$; the largest is
    $9.9 dot 10^(-8)$, for $u overline(u) -> chi_(c 0) thin g$ at 1 TeV.
    - At 1 TeV the median estimate falls by a factor of about 2000 against
      5025e12: $3.9 dot 10^(-5)$ to $1.9 dot 10^(-8)$
      ($u overline(u) -> chi_(c 0) thin g$), $8.1 dot 10^(-6)$ to $3.8 dot 10^(-9)$
      ($g g -> chi_(c 0) thin g$), $3.1 dot 10^(-7)$ to $1.5 dot 10^(-10)$
      ($u overline(u) -> chi_(b 0) thin g$) and $8.0 dot 10^(-8)$ to
      $3.7 dot 10^(-11)$ ($g g -> chi_(b 0) thin g$).
    - The charmonium octets go from $10^(-7)$ to $5 dot 10^(-11)$, the
      bottomonium ones from $10^(-9)$ to $5 dot 10^(-13)$.
    - At 20 GeV every estimate is below $2 dot 10^(-11)$.
    - In these settings (three evaluations per point, the rotation included)
      the median time per point is 4.7 to 6.4 times that of the 5025e12 runs.
      Those runs were made on another day, so the ratio is not paired;
      @tab:ep-cost has the paired cost.
]

= Performance <perf>

MadLoop's standalone output compiled its Fortran without optimisation except
`polynomial.f`. Five changes of the generated code keep every result bit for
bit and are now on `onia_loop`:
- *A:* `-O2` for the MadLoop sources (they come before `FFLAGS`, which can still override them);
- *B:* the colour factors and split-order indices of the counterterm contraction tabulated at initialisation with the expressions they replace;
- *B2:* split-order tables in `CREATE_LOOP_COEFS`;
- *D:* the IEEE rounding and halting modes, which the private DP-terminal updaters need, checked once per coefficient batch instead of in every call. The fail-closed proof of these updaters certifies the new caller statements: a canonical block at the top level of the batch, no jump target between it and the private calls, its result ANDed into each captured INEXACT flag. Combined with the batch check, the updaters' frozen IEEE admission corpus (384 000 trials, every rounding mode and trap) gives the same answers as before;
- *E:* the `DHELAS` and `MODEL` libraries of the standalone output at `-O2` (`ALOHA_FLAG`, `MODEL_FLAG` in `make_opts`, which the proof admits with exactly that value).
Paired alternating runs on this shared host (AMD EPYC 9754, load 80–100,
single core): the median time per point after the initialisation, ratio of the
candidate to the frozen baseline `5c5b573794` (ordinary processes) or to
`onia_loop` before these changes (bound states), with the upper bound of the 95%
confidence interval of the geometric-mean ratio.

#block(breakable: false)[#figure(caption: [Time per phase-space point, baseline and candidate, and their ratio. Acceptance: upper 95% bound ≤ 1.05 and identical results.])[
  #tstyle(table(columns: (1fr, auto, auto, auto, auto, auto, auto, auto),
    stroke: tstroke, fill: thead-fill, align: (left, left, right, right, right, right, right, center),
    [Process (`generate ...`)], [libs], [points × pairs], [baseline ms], [candidate ms], [median ratio], [upper 95%], [identical],
    ..tim.map(r => (raw(r.process), raw(r.libs), [#r.npoints × #r.pairs], sig(r.base_ms), sig(r.cand_ms),
      table.cell(fill: ratio-fill(r.median_ratio))[#fixed(r.median_ratio, 3)],
      table.cell(fill: if r.upper95 <= 1.05 { good } else { bad })[#fixed(r.upper95, 3)],
      if r.identical [yes] else [#sci(r.max_relative_difference)])).flatten()))
]]
#legend()

#eval(meta.perf_note, mode: "markup")

= Open items <open>

#meta.open_items.map(l => [- #eval(l, mode: "markup")]).join()

= Sources and reproducibility <sources>

#set text(9pt)
- Campaign revisions: validation and reducers at `a82b34d314`, timing at `5aadddcbd8`,
  the 2.5.5 comparison at `8effaa7a29`, the P-wave checks at `eda7d7fe04`.
  The changes in between (D and E of @perf) give identical results; the timing
  campaign compares the results of the two bound-state processes bit for bit.
- Model: `tools/make_onia_model.py models/loop_sm models/loop_sm_onia`.
- Validation and reducers (one campaign, `data/raw/validate.json`):
  `tools/onia_loop_validate.py --work DIR --libs 9 3 8 6 7 1 --npoints 10 --timeout 120 --json validate.json`
  (cases and references in the tool and `research/literature.md`).
  The second $e^+ e^- -> J\/psi thin eta_c$ row: the same with `--cases ee_psi_etac --libs 9 --tirust LIB --tirust-label '370b632 (PR #56, candidate)' --json data/raw/validate_tirust370b632.json`, `LIB` a TIRust 370b632 installation (`research/pwave_fd/build.sh`).
- Timing (`data/raw/timing/`): `tools/onia_loop_timing.py --root base=... --root cand=... --process ... --pairs 9`, settings as in @perf.
- One process: `tools/onia_loop_check.py --model loop_sm_onia-c_mass --process "g g > etac(1|1S01) [virt=QCD]" --out DIR --libs 9`.
- S-wave onium + jet against MG5 2.5.5 (`research/mg255_swave/data`): `build_ref.sh`,
  `gen.py`, `compare.py <case>` of `research/mg255_swave` (README there).
- P-wave (`data/raw/pwave/`): `research/pwave_fd/pw_check.py <case>` with
  `TIRUST_LIB` a TIRust 48958d1 installation (`build.sh`), `ENERGY=10` for the
  $e^+ e^-$ cases and 20 for the quark ones, and `research/pwave_fd/ps_term.py`
  for the $P -> S$ term.
- $e^+ e^- -> J\/psi thin chi_(c J)$ (`data/raw/pwave/psi_chic_<J>.json`,
  `psi_chic_exact_<J>.json`): `research/pwave_fd/psi_chic.py --work DIR --json OUT --J <J> --tirust LIB --tirust-label 08cca00`, and the same with `--exact --modes=-1,4,5 --tirust-label 08cca00` in the same `DIR` (without `--tirust`, which would regenerate the outputs). `LIB` is TIRust 08cca00 built by `research/pwave_fd/build.sh` with the bridge patch v8 of the two-P-wave work (`research/two_pwave/FIXTURES.md`). The reference is `research/pwave_fd/psi_chic_ref.py`.
- Stability profiles (`data/raw/stability/`): `research/stability/stability.py WORK data/raw/stability CASE:SETUP ...` (cases and setups in the script; `TIRUST_LIB48958`, `TIRUST_LIB370`, `TIRUST_LIB08`, `TIRUST_LIB56`, `TIRUST_LIBEP` for the scratch TIRust builds; the last, for the 80-bit cases `...ep`, is TIRust d17085a built by `research/pwave_fd/build.sh` with the vendor part of `madloop-tirust-gate-relation-tiers-low-words-extended-inputs-v2.patch` and the metadata key `raised_power_extended_inputs` of its installer; 5025e12 is built by `research/pwave_fd/build.sh` with `BRIDGE_PATCH=madloop-tirust-gate-and-relation-tiers.patch` and `ONELOOP_REV=cdbc912ab7b8…`), then `scripts/stability_plot.py data/raw/stability figures/stability.svg`, which also writes `figures/stability_octets.svg`.
- The 80-bit switch (@ep): `research/pwave_stability/ep_switch.py OUTDIR ESTIM.json NWORKERS MODE ...` at the 24 points (data in `research/pwave_stability/data/ep_switch_*`), and `research/pwave_stability/cost.py CASE SETUP OFF_OUTDIR ON_OUTDIR NPOINTS 4 production+ep 8` for the cost.
- This report: `python3 scripts/collect.py ... --validation-extra data/raw/validate_tirust370b632.json --mg255 research/mg255_swave/data --pwave data/raw/pwave; typst compile benchmark.typ benchmark.pdf`.
- References: Petrelli, Cacciari, Greco, Maltoni, Mangano, hep-ph/9707223;
  Hagiwara, Sumino, Yokoya, 0804.1014; Sang, Chen, 0910.4071; Gong, Wang,
  0712.4220; Catani, Dittmaier, Trócsányi, hep-ph/0011222; quarkonium in
  MadGraph: arXiv:2510.26773, arXiv:2607.26739 (tree level), arXiv:2402.19221
  (NLO, FKS and poles); 1511.06288 and 2501.04188 ($chi_(c J) -> gamma gamma$,
  $gamma g$); Wang, Ma, Chao, 1107.2646 and Dong, Feng, Jia, 1107.4351
  ($e^+ e^- -> J\/psi thin chi_(c J)$). Style after the pyAmpliCol one-loop report.
