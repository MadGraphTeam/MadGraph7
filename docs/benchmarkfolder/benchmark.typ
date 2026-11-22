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
  - *With TIRust 08edee3* (PR \#51, not pinned) all 34 such points are served,
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
    MadLoop's stability tests vary only the loop part, not the counterterm
    amplitudes.
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
- the same in double precision with the stability checks, every point reported stable (code 213).])[
  #tstyle(table(columns: (auto, auto, auto, auto, auto, auto, auto),
    stroke: tstroke, fill: thead-fill, align: (left, right, right, right, right, right, right),
    [Polarisation], [Born ratio], [pinned: stable], [$c_0$ dev.], [$c_1$ dev.], [08edee3 QP: $c_0$, $c_1$ dev.], [08edee3 DP: $c_0$ dev.],
    ..mgp.map(c => (pol-label.at(c.pol), [#fixed(c.born_ratio.at(0), 9)–#fixed(c.born_ratio.at(1), 9)],
      [#c.stable/#c.npoints], dev-cell(c.pin_fin, target: 1e-5), sci(c.pin_c1),
      [#sci(c.qp_fin), #sci(c.qp_c1)],
      table.cell(fill: acc-fill(c.dp_fin, 1e-5))[#sci(c.dp_fin)])).flatten()))
] <tab:jpsipol>

The longitudinal Born is suppressed, so its double-precision value carries
MadLoop's rounding of $10^(-7)$ relative to it; in quadruple precision every
polarisation agrees to $3 dot 10^(-12)$. The red column is the double-precision
rounding of the R2 counterterms of the $10^4$ GeV $b$ and $t$ quarks at the soft
points (above): at $sqrt(s) = 3.1015$ GeV the longitudinal $c_0$ is off by 440%
while reported stable.

= P-wave bound states <pwave>

The P-wave results use TIRust 48958d1 (huashengshao/TIRust PR \#50, stacked on
\#47 to \#49, not yet the pinned revision). That revision serves every integral
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

#block(breakable: false)[#figure(caption: [The poles of $2 -> 2$ P-wave processes against the prediction, TIRust 48958d1, four points at the given $sqrt(s)$: the range of $c_1$ over the points, the range of the $P -> S$ term of the prediction, and the largest deviation of $c_1$, $c_2$ from the prediction (green within $10^(-8)$). The finite parts have no reference.])[
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
    raised powers; open TIRust item).
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
  lanes (a threshold bubble pinch; two Coulomb relations in one pentagon);
  (3) P-wave states need the doubled-propagator integrals at threshold:
  TIRust 48958d1 has them for the $2 -> 1$ processes (@pwave), not yet pinned.
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

#figure(caption: [Time per phase-space point, baseline and candidate, and their ratio. Acceptance: upper 95% bound ≤ 1.05 and identical results.])[
  #tstyle(table(columns: (1fr, auto, auto, auto, auto, auto, auto, auto),
    stroke: tstroke, fill: thead-fill, align: (left, left, right, right, right, right, right, center),
    [Process (`generate ...`)], [libs], [points × pairs], [baseline ms], [candidate ms], [median ratio], [upper 95%], [identical],
    ..tim.map(r => (raw(r.process), raw(r.libs), [#r.npoints × #r.pairs], sig(r.base_ms), sig(r.cand_ms),
      table.cell(fill: ratio-fill(r.median_ratio))[#fixed(r.median_ratio, 3)],
      table.cell(fill: if r.upper95 <= 1.05 { good } else { bad })[#fixed(r.upper95, 3)],
      if r.identical [yes] else [#sci(r.max_relative_difference)])).flatten()))
]
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
- Timing (`data/raw/timing/`): `tools/onia_loop_timing.py --root base=... --root cand=... --process ... --pairs 9`, settings as in @perf.
- One process: `tools/onia_loop_check.py --model loop_sm_onia-c_mass --process "g g > etac(1|1S01) [virt=QCD]" --out DIR --libs 9`.
- S-wave onium + jet against MG5 2.5.5 (`research/mg255_swave/data`): `build_ref.sh`,
  `gen.py`, `compare.py <case>` of `research/mg255_swave` (README there).
- P-wave (`data/raw/pwave/`): `research/pwave_fd/pw_check.py <case>` with
  `TIRUST_LIB` a TIRust 48958d1 installation (`build.sh`), `ENERGY=10` for the
  $e^+ e^-$ cases and 20 for the quark ones, and `research/pwave_fd/ps_term.py`
  for the $P -> S$ term.
- This report: `python3 scripts/collect.py ... --mg255 research/mg255_swave/data --pwave data/raw/pwave; typst compile benchmark.typ benchmark.pdf`.
- References: Petrelli, Cacciari, Greco, Maltoni, Mangano, hep-ph/9707223;
  Hagiwara, Sumino, Yokoya, 0804.1014; Sang, Chen, 0910.4071; Gong, Wang,
  0712.4220; Catani, Dittmaier, Trócsányi, hep-ph/0011222; quarkonium in
  MadGraph: arXiv:2510.26773, arXiv:2607.26739 (tree level), arXiv:2402.19221
  (NLO, FKS and poles); 1511.06288 and 2501.04188 ($chi_(c J) -> gamma gamma$,
  $gamma g$). Style after the pyAmpliCol one-loop report.
