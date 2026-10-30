# SPDX-License-Identifier: 0BSD
"""Listed loop-admission refusals of v2 generation (fail closed).

A structure the engine does not cover is refused at generation, before any
plan is exported, with a typed error (``LoopRequestUnsupported``, a
``GenerationError``) whose ``key`` names a listed rule
(``LOOP_ADMISSION_REFUSALS``; the dry run lists them as
``loop_admission_refusals``, as it lists ``native_trees.NATIVE_EXPORT_REFUSALS``
as ``native_export_refusals``), never with a silently wrong value. Each rule
is decided by a structural criterion of the request's model and trees, never
by process strings, particle or parameter names, PDG codes or leg and line
counts (second genericity audit, ``genericity-audit-2/REPORT.md``):

* ``colour-accuracy-nc-exact`` (G2-M1): leading and next-to-leading colour
  project the exact colour cells at formal N_c, which needs every loop-side
  tree block to have one N_c grade (``BlockColourFlow.nc_grade``, design
  §3.2 item 4 and §3.3). LC and NLC requests compile their loop-request
  DAGs with the N_c grade split (``generation.dag_grades`` and the native
  ``loop_v2::dag::grades``), which copies every current whose derivations
  mix U(1)-subtraction grades into one current per grade, so every block of
  such a request has one grade and the guard passes (identical-flavour quark
  lines included). The rule is a guard only: a block whose trees still mix
  grades (its ``nc_grade`` is ``None``, i.e. a DAG compiled without the
  split) is not exact in N_c, so such a request is refused at LC and NLC
  (``require_nc_exact``, right after the trees are built) and every export
  asserts it again (``require_formal_nc_tables``): the native runtime and the
  SDKs can never serve a grade-0 projection of such blocks. Full colour keeps
  every power at N_c = 3, is exact without grades and never splits.
* ``fermion-line-representation`` (G2-m1): the loop-request blocks sign
  coloured fermion lines by their colour flow and colour-singlet ones by the
  Wick order of their external fermions (``generation.fermion_ordering``,
  colour-flow convention). A vertex whose fermions have different sign
  ownership (a coloured and a colour-singlet fermion: one fermion line that
  changes colour representation, e.g. a leptoquark u e S vertex) has no sign
  in either convention. ``require_colour_preserving_fermion_lines`` refuses a
  request that can use such a vertex before any tree is built, naming it; the
  native DAG builder refuses the same merge with the listed key
  (``rusticol_core::loop_v2::dag::FERMION_LINE_REPRESENTATION``), which
  ``dag_input`` raises as this error.
* ``veto-unlabelled-counterterms`` (WP3.2 EWI-5 interim, until EWI-7's [U2]
  veto guard and C1 UV certifier): ``particle_veto`` reaches the
  counterterms through their UFO ``loop_particles`` labels only
  (REQUIREMENTS_ADDENDUM J: a term whose label holds a vetoed particle is
  dropped). That removes a vetoed species' loops from the counterterms only
  when the labels partition every counterterm coupling by its loop content.
  A model that writes a counterterm term of a correction with no loop
  content at all (an empty label, the admission's wildcard: MG5_aMC's NLOCT
  electroweak UV counterterms, one coupling over every loop of a vertex)
  does not, and its other labels are nominal too (``loop_qcd_qed_sm``
  labels the photon-and-Z R2 of ``e+ e- a`` by ``[e-, Z]`` and the
  electroweak UV counterterm of ``u u~ g`` by ``[u, Z]``), which no
  structure tells from a faithful label. A veto that removes a loop species
  of such a correction then keeps vetoed loops in some counterterms and
  drops kept ones from others: a silently wrong virtual (photon-only
  ``e- e+ > mu- mu+`` on loop_qcd_qed_sm: single pole -3.487e-4 against the
  QED Catani operator's -1.841e-4; ``poles = "uv"`` cannot see it, since
  the field factors' UV part is the remainder). The criterion is the
  model's own labelling: a vetoed species perturbative in a requested
  correction (``perturbative_species_orders``) while some counterterm term
  of the model (``LoopModelIR.kernels``, every vertex, not only those the
  request reaches) whose coupling orders hold a requested correction has
  an empty label (``unlabelled_counterterm_vertices``).
  ``require_labelled_counterterms_for_veto`` refuses such a request before
  any tree, naming the vetoed species and those vertices. It is also the
  [U2] anomaly guard of electroweak corrections on such models: a veto that
  breaks an anomaly-free multiplet (a quark of a doublet, a whole lepton
  generation) removes a species perturbative in the electromagnetic
  correction and is refused here; QCD corrections take the Levi-Civita
  cells' guard (``ct_recursion``, "Anomalous vetoes"; WP3.2 EWI-7).
* ``loop-vertex-outside-tree-model`` (fail-closed-2, a guard): loop lines
  read the colour and Lorentz structures of their vertices from the
  request's tree model (one set of conventions for loops, blocks and the
  Born). A vertex of the loop model whose couplings at an order vanish
  identically at the model's restriction (``ct_sites.restricted_to_zero``,
  decided from the coupling values: e.g. a W vertex of a CKM element that a
  diagonal-CKM restriction sets to zero, which the tree compiler drops) is
  no loop vertex there (``ct_sites.vanishing_loop_vertices``,
  ``families.ModelLoopCatalogue``). Any other loop vertex the tree model does
  not hold at the order a loop position takes it has no structure to read:
  the request is refused when a loop position uses it
  (``colour.SeedColour.states``), naming the vertex and the order.
* The QCD guards of the public route (m-qcd MQ-MODEL; user rulings
  2026-10-28 ~15:25): ``loop-induced`` (no Born amplitude within the
  request's coupling-order limits, ``generation.loop_service._v2_request``),
  ``coloured-width`` (a coloured particle with a nonzero width in a tree
  current, ``artifact.require_zero_coloured_widths``), and at model
  preparation (``loops.covariant_model.build_covariant_structure``)
  ``lorentz-slot-arity`` (a vertex tensor whose slots do not match its
  particles once the structural source repair,
  ``covariant_model.repair_source_declarations``, has moved the constant
  scalar ones), ``majorana-fermion`` and ``multi-fermion-contact``. Each is
  decided by structure: widths, spin slots, the self-conjugacy of a
  fermion, the number of fermions at a vertex. Process-request refusals
  (1 -> n decays, MG5_aMC syntax) are listed in ``processes.refusals``.
  A massive (Dirac) current closing on a massless (Weyl) one (a fermion line
  joining massive and massless fermions through a chiral vertex, e.g.
  b b~ > w+ w- with a massless b) is no loop refusal since m-qcd MQ-5F: the
  loop Born DAG records it as a marker root and the Born closes at a vertex
  (``generation.dag_ordering._direct_contraction_ir`` with ``loop_born``);
  the tree lanes refuse it (``dag_ordering.DiracWeylClosureUnsupported``,
  key ``dirac-weyl-closure``).
* ``forced-seed-fft-unserved`` (MQ-FFT, a guard): ``colour_contraction =
  "fft"`` requires the certified symmetric-group FFT of the seed contraction
  (design §7.3). Orbits of the external adjoint labels are found and
  certified on the exact seed cells (``colour.seed_fourier``: a structural
  criterion, no process or multiplicity key); a table without a certified
  pair, a leading-colour request (diagonal, contracted directly) and declared
  observables (their columns are contracted directly) are refused
  (``LoopSlice._plan_contraction``, ``ObservableSlice.of``) instead of
  silently contracting directly. ``"auto"`` serves all of them, reporting the
  algorithm it ran.
* ``selected-correlators`` (MQ-SELECT, a guard): declared colour or spin
  correlations (or the Ward mode) with a leading-colour selection
  (``selection.Selection``) are refused, as the tree engine's correlated
  generation refuses its selectors (``generation.correlated``): on the
  public route by the selectors set (``loop_service._check_correlators``),
  in ``ObservableSlice.of`` by a selection that keeps fewer helicities or
  flows (``Selection.restricts``).
* ``subleading-colour-flows`` (MQ-SELECT): a generation-time colour-flow
  selection none of whose flows reaches the process's leading N_c degree in
  a squared Born order they reach (``LoopSlice._refuse_inexact_flows``,
  from the slice's own grading of the unrestricted Born, identical-flavour
  grade splits included): its leading-colour value there is zero, while the
  restricted Born alone would set a lower degree. Next to a leading flow
  such a flow contributes exact zeros and is kept.
* ``interfering-colour-flows`` (MQ-SELECT): a generation-time colour-flow
  selection one of whose flows has a leading-power term with an unselected
  flow in a squared Born order (``LoopSlice._refuse_inexact_flows``;
  possible between two Born orders only, e.g. QCD and photon Born
  amplitudes): the restricted Born would drop that term, while the
  evaluation-time selection keeps it.
* ``colour-declaration-coverage`` (MQ-NC): LC and NLC grade the counterterm
  and external-field couplings by the formal-N_c continuation of the
  request's colour declaration (design §6.7). A declaration covers the
  counterterm kernels (``LoopColorDeclaration.kernel_keys``, plus the
  kernels of its whole-rule recipes) and field factors (``field_keys``) it
  states; the shipped ones (``models.loop_color_declarations``) cover the
  vertices whose structural digest they record. A covered vertex's N_c
  dependence is declared (and proved against the source at N_c = 3); an
  uncovered one has none, and grading it by the identities alone could
  read an N_c = 3 number as N_c-independent. So an LC or NLC request is
  refused when its counterterm program can insert an uncovered vertex (one
  that a tree of the request can hold: ``ct_closure.InsertionReachability``,
  as for an unlowerable vertex, REQUIREMENTS_ADDENDUM J; one no tree can
  hold contributes nothing and is left alone) or an external leg takes an
  uncovered field factor (``require_declared_colour_rules``,
  ``require_declared_colour_fields``), naming the vertices and factors.
  Full colour needs no declaration.
* The mass scheme (M-SM MS-CMS, ``loops.model_cms``): the declaration's
  ``mass_scheme`` is ``"real-on-shell"`` or ``"complex-mass"``
  (``mass-scheme`` otherwise). Real on-shell requests propagate no width in
  coloured trees (``coloured-width``) nor, if electroweak, in any tree
  (``electroweak-width``, [U8]); the complex-mass scheme is MG5_aMC's
  transform of the model (``complex_mass_model``), which refuses a model
  outside its assumptions (``complex-mass-scheme``), a width-carrying
  particle of internal mass (``cms-internal-mass``), a fixed Yukawa input
  (``cms-yukawa-input``) and an external particle of a complex pole mass
  (``cms-external-width``, EW-25). Fixed parameters are real in both schemes
  (``complex-fixed-parameter``). Each is decided by the model's records and
  the card: widths, parameter natures and SLHA blocks, the external legs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from pyamplicol.api.errors import GenerationError


@dataclass(frozen=True)
class LoopAdmissionRefusal:
    """A listed structure v2 generation refuses (``LoopRequestUnsupported``):
    ``key`` names it, ``feature`` says what is refused, ``rule`` why (the
    rule the message names), ``remedy`` what the user can request instead,
    and ``owner`` lifts it with ``follow_up``."""

    key: str
    feature: str
    rule: str
    remedy: str
    owner: str
    follow_up: str


LOOP_ADMISSION_REFUSALS = (
    LoopAdmissionRefusal(
        "colour-accuracy-nc-exact",
        "a leading- or next-to-leading-colour request whose loop-side tree "
        "currents are not exact in N_c (blocks whose trees still mix "
        "U(1)-subtraction grades: a loop-request DAG compiled without the "
        "N_c grade split)",
        "LC and NLC project the colour cells at formal N_c, which needs every "
        "tree block to have one N_c grade (design §3.2 item 4, §3.3)",
        'full colour (color.accuracy = "full") is exact without grades',
        "none: a guard only (LC and NLC compile with the N_c grade split, "
        "generation.dag_grades and loop_v2::dag::grades, so every block of "
        "their requests has one grade)",
        "none (kept as the guard of the grade split)",
    ),
    LoopAdmissionRefusal(
        "fermion-line-representation",
        "a vertex joining a coloured and a colour-singlet fermion (one fermion "
        "line changing colour representation, e.g. a leptoquark u e S vertex)",
        "the loop-request blocks sign coloured fermion lines by their colour "
        "flow and colour-singlet ones by their Wick order "
        "(generation.fermion_ordering), so such a line has no sign",
        "veto a particle of the vertex (particle_veto) or bound its coupling "
        "order so that the request cannot use it",
        "A native-dag with WP3.2 EWI-5 (ew-leptons)",
        "Wick ranks for every fermion line, coloured or not",
    ),
    LoopAdmissionRefusal(
        "veto-unlabelled-counterterms",
        "a particle_veto that removes a loop species of a requested correction "
        "on a model whose counterterm terms of that correction do not all "
        "declare their loop content (loop_particles = [], e.g. MG5_aMC's "
        "loop_qcd_qed_sm and its G_mu variant: photon-only QED corrections)",
        "the veto drops the counterterm terms whose loop_particles hold a "
        "vetoed particle (REQUIREMENTS_ADDENDUM J), which removes the vetoed "
        "loops only when the labels partition every counterterm coupling by "
        "loop content; unlabelled terms show that this model's labels do not",
        "the complete corrections (no particle_veto of their loop species), or "
        "a model whose counterterm terms each declare their loop content",
        "a UFO whose counterterm couplings are split by loop content (MG5_aMC's "
        "NLOCT electroweak counterterms are not), or WP3.5 C1 (UV certifier)",
        "per-loop-content counterterm couplings from the model; the C1 "
        "certifier can certify a veto's UV poles, not the on-shell finite "
        "parts of the counterterms, so photon-only corrections on these UFOs "
        "stay refused (WP3.2 EWI-7)",
    ),
    LoopAdmissionRefusal(
        "loop-vertex-outside-tree-model",
        "a loop vertex that the request's restricted tree model does not hold at "
        "the coupling order a loop of the request takes it, while its couplings "
        "there are not shown to vanish at the model's restriction",
        "loop lines read their vertices' colour and Lorentz structures from the "
        "tree model; a vertex whose couplings vanish identically at the "
        "restriction leaves the loops (ct_sites.restricted_to_zero), any other "
        "vertex the tree model drops has no structure to read",
        "a restriction under which the vertex's couplings vanish exactly, or one "
        "that keeps the vertex in the tree model; or a particle_veto or "
        "coupling-order bound that keeps it out of the request's loops",
        "fail-closed-2 (a guard)",
        "none (kept as the guard of the restriction rule)",
    ),
    LoopAdmissionRefusal(
        "loop-induced",
        "a request without a Born amplitude within its coupling-order limits "
        "(a loop-induced process, MG5_aMC [noborn=QCD]: g g > h h, g g > z z, "
        "g g > a a, ...)",
        "v2 virtuals interfere one-loop and Born amplitudes; a loop-squared "
        "matrix element is not served",
        "a process with a Born amplitude, or coupling-order limits "
        "(process.max_coupling_orders) that admit one",
        "after M-SM (user ruling 2026-10-28 ~15:25 (2))",
        "loop-squared matrix elements",
    ),
    LoopAdmissionRefusal(
        "coloured-width",
        "a real on-shell request whose tree currents propagate a coloured "
        "particle with a nonzero width (e.g. the top at a restriction with "
        "DECAY 6 1.4915)",
        "real on-shell virtuals: loop propagators carry no width and the model "
        "counterterms renormalise the real mass, so a width in the tree "
        "currents makes the Born, its colour correlations and the virtual "
        "inconsistent",
        'the complex-mass scheme ([loop.model.scheme] mass_scheme = "complex-mass", '
        "MG5_aMC's set complex_mass_scheme True), a restriction with every width "
        'at zero (model.restriction = "no_widths"), or the width fixed to zero '
        'in [loop.preparation] fixed_parameters (e.g. WT = "0")',
        "none (a guard of the real on-shell scheme; a resonant particle takes "
        "the complex-mass scheme)",
        "none",
    ),
    LoopAdmissionRefusal(
        "electroweak-width",
        "a real on-shell electroweak request (a correction other than the strong "
        "one, ew_policy.electroweak) whose tree currents propagate a particle "
        "with a nonzero width, coloured or not (e.g. the Z at the G_mu UFO's "
        "default card)",
        "real on-shell virtuals: loop propagators carry no width and the model "
        "counterterms renormalise the real masses, so a width in the Born, the "
        "loop-side or the counterterm trees mixes two schemes ([U8])",
        'the complex-mass scheme ([loop.model.scheme] mass_scheme = "complex-mass"), '
        "or the widths fixed to zero in [loop.preparation] fixed_parameters",
        "none (a guard of the real on-shell scheme; a resonant particle takes "
        "the complex-mass scheme)",
        "none",
    ),
    LoopAdmissionRefusal(
        "mass-scheme",
        "a loop model declaration whose mass_scheme is neither "
        '"real-on-shell" nor "complex-mass"',
        "the mass scheme decides the model the virtual is computed in; an "
        "unknown value (a misspelt complex-mass request) is never computed "
        "real on-shell",
        '[loop.model.scheme] mass_scheme = "real-on-shell" or "complex-mass"',
        "none (a guard of the declaration)",
        "none",
    ),
    LoopAdmissionRefusal(
        "complex-mass-scheme",
        "a complex-mass request on a model the transform (a port of MG5_aMC's "
        "change_mass_to_complex_scheme, loops.model_cms) cannot take: no raw "
        "UFO records, an external CMSParam, an internal width without a source "
        "expression, a Yukawa input without one PDG code, no ZERO parameter, "
        "a CMASS_ or LMASS_ name already taken; or a public-route request "
        "with complex pole masses in a build whose stored artifacts' native "
        "point inputs take none (model_cms.NATIVE_POLE_MASSES; this build takes "
        "them, M-SM MS-CMS-N)",
        "the complex-mass scheme is MG5_aMC's transform of the model's own "
        "records; a model outside its assumptions has no such transform",
        'the real on-shell scheme (mass_scheme = "real-on-shell") with the '
        "widths fixed to zero, or a model within the transform's assumptions",
        "none (a guard of the transform); the native pole masses: M-SM MS-CMS-N",
        "complex plan masses through the native trees and executor",
    ),
    LoopAdmissionRefusal(
        "cms-internal-mass",
        "under the complex-mass scheme, a particle with a nonzero width whose "
        "mass is an internal parameter (loop_sm's W: MW derived from MZ, "
        "alpha and G_F)",
        "MG5_aMC switches such a model to (MZ, MW, alpha) inputs through an "
        "SM-specific rule (PDG 24 and the MW expression), which the "
        "genericity rule forbids; without it the pole mass has no input",
        "the width fixed to zero, or a model whose mass is an external "
        "parameter (the G_mu or alpha(M_Z) NLOCT UFO under the complex-mass "
        "scheme)",
        "none (deliberate deviation from MG5_aMC, m-sm PLAN §1.7)",
        "none",
    ),
    LoopAdmissionRefusal(
        "cms-yukawa-input",
        "under the complex-mass scheme, a request that fixes a Yukawa input "
        "(an SLHA YUKAWA parameter)",
        "MG5_aMC's complex-mass scheme removes the YUKAWA block: the Yukawa "
        "couplings follow the (pole) masses",
        "fix the mass instead, or the real on-shell scheme",
        "none (MG5_aMC's rule)",
        "none",
    ),
    LoopAdmissionRefusal(
        "cms-external-width",
        "under the complex-mass scheme, an external particle with a nonzero "
        "width (its mass a complex pole mass)",
        "external states are stable (EW-25): an on-shell external state has a "
        "real mass",
        "the external particle's width fixed to zero in [loop.preparation] "
        "fixed_parameters (MG5_aMC computes it with the complex mass)",
        "none (EW-25)",
        "none",
    ),
    LoopAdmissionRefusal(
        "complex-fixed-parameter",
        "a fixed parameter with a nonzero imaginary part ([loop.preparation] "
        "fixed_parameters)",
        "external parameters are real (as MG5_aMC's); the complex-mass scheme "
        "derives its complex pole masses from real masses and widths",
        "real fixed parameters; complex pole masses through mass_scheme = "
        '"complex-mass"',
        "none (a guard)",
        "none",
    ),
    LoopAdmissionRefusal(
        "lorentz-slot-arity",
        "a vertex that references a Lorentz tensor whose spin slots differ from "
        "its particles, other than a constant scalar tensor with a unique "
        "same-valued tensor of the right slots (repaired by the source rule "
        "covariant_model.repair_source_declarations, e.g. loop_sm's V_1001)",
        "a tensor whose slots do not match the vertex's particles has no "
        "meaning on them",
        "a model whose vertices reference tensors of their own spin slots, or "
        "coupling-order limits or a particle_veto that keep the vertex out of "
        "the request",
        "none (a guard of the source declarations)",
        "none",
    ),
    LoopAdmissionRefusal(
        "majorana-fermion",
        "a model with a Majorana (self-conjugate) fermion",
        "Majorana / fermion-number-violating fermion flow is not implemented",
        "a model without Majorana fermions",
        "after M-FINAL",
        "fermion-number-violating fermion flow",
    ),
    LoopAdmissionRefusal(
        "multi-fermion-contact",
        "a vertex with more than two fermions (a four-fermion contact "
        "interaction) that the request can reach",
        "loop and tree lines carry one fermion line per vertex",
        "coupling-order limits or a particle_veto that keep the contact vertex "
        "out of the request",
        "after M-FINAL",
        "multi-fermion contact vertices",
    ),
    LoopAdmissionRefusal(
        "forced-seed-fft-unserved",
        'a forced seed FFT (color.contraction = "symmetric-group-fft", the '
        "slice's colour_contraction 'fft') that the FFT cannot serve: a seed "
        "table without a certified symmetric-group orbit pair (e.g. fewer than "
        "two permutable external gluons), leading colour, or declared colour, "
        "spin or Ward observables",
        "a forced FFT requires the FFT and never contracts silently directly; "
        "the orbits of the external adjoint labels are detected and certified "
        "structurally on the exact seed cells (colour.seed_fourier), leading "
        "colour is diagonal and observable columns are contracted directly "
        "(design §7.3)",
        'color.contraction = "auto" (the certified pairs by the FFT, the rest '
        'directly, the algorithm reported) or "direct"',
        "MQ-FFT (a guard)",
        "none (kept: a forced FFT is never silently direct)",
    ),
    LoopAdmissionRefusal(
        "selected-correlators",
        "declared colour or spin correlations (or the Ward mode) with a "
        "leading-colour helicity or colour-flow selection (the public route's "
        "process.selected_source_helicities, selected_color_sector_ids, "
        "max_color_sectors or reference_color_order; SliceRequest.selection)",
        "correlated generation requires complete colour and helicity coverage, "
        "as the tree engine's correlated generation: a spin source on a leg "
        "whose helicity is selected would ignore the selection",
        "generate the correlations without selectors (an inclusive artifact), "
        "and the selection in an artifact of its own",
        "MQ-SELECT (a guard)",
        "none (the tree engine's rule)",
    ),
    LoopAdmissionRefusal(
        "subleading-colour-flows",
        "a leading-colour colour-flow selection at generation (the public "
        "route's selected_color_sector_ids or a truncating max_color_sectors; "
        "SliceRequest.selection) none of whose flows reaches the leading N_c "
        "power of the squared Born at a squared Born order the flows reach "
        "(e.g. u u~ > d d~ g's tree flows joining each quark to its own "
        "antiquark)",
        "the loop's leading colour is the leading N_c power of the squared "
        "matrix element: there such a selection's leading-colour Born and "
        "virtual are zero, while its restricted Born alone would set a lower "
        "power and return a subleading value",
        "select at least one flow of the process's leading colour with it "
        "(the others then contribute exact zeros), or select the flows at "
        "evaluation (Runtime.evaluate(..., color_flows=...): exactly zero)",
        "MQ-SELECT (a guard)",
        "none (the leading-colour rule)",
    ),
    LoopAdmissionRefusal(
        "interfering-colour-flows",
        "a leading-colour colour-flow selection at generation (the public "
        "route's selected_color_sector_ids or a truncating max_color_sectors; "
        "SliceRequest.selection) one of whose flows interferes at the leading "
        "N_c power of a squared Born order with an unselected flow (several "
        "Born orders, e.g. u u~ > d d~'s QCD flow (2,4,3,1) with the photon "
        "Born's (2,1,3,4) at QCD^2 QED^2)",
        "a generation-time selection restricts the Born itself, so it would "
        "drop that leading term of the squared Born, which the "
        "evaluation-time selection of the flow keeps",
        "select the interfering flows with it, or select the flows at "
        "evaluation (Runtime.evaluate(..., color_flows=...): exact)",
        "MQ-SELECT (a guard)",
        "none (the leading-colour rule)",
    ),
    LoopAdmissionRefusal(
        "colour-declaration-coverage",
        "a leading- or next-to-leading-colour request whose counterterm program "
        "can insert a counterterm vertex, or whose external legs take a "
        "field factor, that its colour declaration does not cover (e.g. a "
        "vertex whose content differs from the shipped loop_sm data's)",
        "LC and NLC grade counterterm and field couplings by the declaration's "
        "formal-N_c continuation (design §6.7); an uncovered coupling has no "
        "declared N_c dependence",
        'full colour (color.accuracy = "full"), or a colour declaration that '
        "covers the vertex (with_loop_color_declaration)",
        "MQ-NC (a guard)",
        "none (kept as the guard of declared N_c data)",
    ),
    LoopAdmissionRefusal(
        "external-photon-scheme",
        "tagged on-shell external photons (loop.external_photons; MG5_aMC's "
        "!a!) in a request with the electromagnetic correction where the "
        "alpha(0) convention cannot be applied: a model without the conversion "
        "counterterm delta Z_e^Gmu - delta Z_e^alpha(0) (eCoup_DGmua0_UV_EW) or "
        "without an adjustable 1/alpha(0) (SMINPUTS 1), loops that already "
        "couple with that parameter, 1/alpha(0) fixed to zero, a particle_veto "
        "of an electromagnetically perturbative species, or a request without "
        "the electromagnetic correction; and any on-shell external photon of an "
        "electroweak request on a model with several photon candidates "
        "(massless, neutral, colourless vectors: ambiguous)",
        "a tagged photon couples with alpha(0) and takes the conversion term "
        "(MG5_aMC's alpha(0) formula over the tagged legs), which needs the "
        "model's conversion counterterm and its adjustable 1/alpha(0); MG5_aMC "
        "refuses tagged photons on such a model too",
        'leave the photons untagged (loop.external_photons = "none", or a leg '
        "list without them: they keep the loops' coupling and their "
        "gamma_gamma pole), or a model that declares the conversion with an "
        "adjustable 1/alpha(0) (MG5_aMC's loop_qcd_qed_sm_Gmu under its "
        "restriction a0)",
        "MS-PHOTON (a guard)",
        "none (the model must declare the conversion)",
    ),
    LoopAdmissionRefusal(
        "external-photon-legs",
        "a loop.external_photons leg list naming a leg that is not an on-shell "
        "external photon of the process (a massless, neutral, colourless "
        "physical vector) or no leg of the process",
        "only on-shell external photons can couple with alpha(0) (MG5_aMC's "
        "!a! tags photons only)",
        "list the 1-based labels of the process's on-shell photons (initial "
        'legs allowed), or "final", "all" or "none"',
        "MS-PHOTON (a guard)",
        "none",
    ),
    LoopAdmissionRefusal(
        "feynman-gauge",
        "a loop request on a model that cannot be taken in 't Hooft-Feynman "
        "gauge: its source declares no gauge 1, or an electroweak request on "
        "a model bundle prepared in another gauge",
        "loop lines, the Born and the counterterms take the gauge of the "
        "model's UV and R2 rules, 't Hooft-Feynman gauge (design §6.8), in "
        "which a massive vector's loop line has rank <= 1",
        "an NLO UFO that declares gauge 1 (MG5_aMC's loop models), compiled "
        "from its source rather than a prepared bundle",
        "WP3.2 EWI-4b (a guard)",
        "none",
    ),
    LoopAdmissionRefusal(
        "antiparticle-metadata",
        "a source model whose particle-antiparticle pairs are not reciprocal "
        "or differ in spin, mass, width, propagation, Goldstone flag, PDG "
        "sign, colour or conjugate charges (covariant_model._validate_pairs)",
        "the loop model's covariant structure conjugates fields by their "
        "declared antiparticles; the Goldstone flag MG5_aMC's legacy UFO "
        "object library drops is repaired by structure first",
        "a source whose antiparticles agree (MG5_aMC's loop models do, after "
        "the repair)",
        "m-sm MS-MODEL (model preparation)",
        "none",
    ),
    LoopAdmissionRefusal(
        "anomalous-veto",
        "a particle_veto that breaks the cancellation of a Levi-Civita "
        "counterterm cell (a quark of a doublet, a whole lepton generation) "
        "whose vertex a tree of the request can hold "
        "(ct_recursion.AnomalousVetoError)",
        "such a cell cancels only with every flavour of its multiplet admitted "
        "([U2]); under the veto the remaining gamma-five terms are "
        "scheme-dependent",
        "veto whole anomaly-free multiplets, or set "
        "generation.allow_anomalous_veto = true to lower the cells anyway",
        "WP3.2 EWI-7 ([U2])",
        "none (the [U2] rule)",
    ),
)

# Configuration-level refusals (m-sm MS-MODEL): settings of the run card
# and the loop model declaration that v2 one-loop generation does not take,
# decided before any tree from the configuration and the model's own
# declarations (``generation.loop_service``, ``models.loop_declaration``).
# They are raised as ``LoopRequestUnsupported`` like the admission keys and
# listed by the dry run as ``loop_configuration_refusals``.
LOOP_CONFIGURATION_REFUSALS = (
    LoopAdmissionRefusal(
        "loop-model-declaration",
        "a loop request on a model with no [loop.model] declaration: none in "
        "the card, none in the source's metadata, and the source is not one "
        "of the shipped profiles (loops.shipped_profiles, by the digest of its "
        "declarations)",
        "v2 takes the scheme and the completeness of the model's UV and R2 "
        "rules from a declaration, never from model names",
        "[loop.model] and [loop.model.scheme] in the card, or a shipped model "
        "(loop_sm, loop_qcd_qed_sm, loop_qcd_qed_sm_Gmu) by name",
        "m-qcd MQ-MODEL, m-sm MS-MODEL",
        "none",
    ),
    LoopAdmissionRefusal(
        "loop-model-corrections",
        "corrections (process.loop_coupling_orders) outside the "
        "loop_coupling_orders of the model's [loop.model] declaration (e.g. "
        "QED on loop_sm, whose shipped declaration covers QCD)",
        "the declaration states which corrections the model's UV and R2 rules complete",
        "request covered corrections, or a model whose declaration covers "
        "them (QED and QCD+QED: loop_qcd_qed_sm, loop_qcd_qed_sm_Gmu)",
        "m-sm MS-MODEL",
        "none",
    ),
    LoopAdmissionRefusal(
        "loop-model-completion",
        "a [loop.model] declaration without complete UV counterterms or "
        "complete model R2 rules",
        "the production path is the model's own UV and R2 rules "
        "(REQUIREMENTS_ADDENDUM J); direct R2 comes after M-FINAL",
        "a model and declaration with uv_complete and model_r2_complete",
        "m-sm MS-MODEL",
        "none",
    ),
    LoopAdmissionRefusal(
        "undeclared-corrections",
        "corrections the source model does not expand perturbatively (its "
        "coupling order's perturbative_expansion is below 1)",
        "v2 would admit no loop species of such an order and return a "
        "vanishing correction",
        "corrections the source expands (an NLO UFO with that order's loop content)",
        "m-sm MS-MODEL",
        "none",
    ),
    LoopAdmissionRefusal(
        "scale-parameter",
        "a model whose renormalisation-scale parameter is not identified: no "
        "adjustable external parameter at (LOOP, 1) and not exactly one in "
        "block LOOP",
        "the counterterms' logarithms read the renormalisation scale from that "
        "parameter (_v2_scale_parameter)",
        "keep the model's scale parameter adjustable (not in "
        "loop.preparation.fixed_parameters, not fixed by its restriction)",
        "m-sm MS-MODEL",
        "none",
    ),
    LoopAdmissionRefusal(
        "veto-external-particle",
        "a generation.particle_veto that names an external particle of the process",
        "a veto removes internal lines and loops; an external leg cannot be removed",
        "veto internal particles only",
        "m-sm MS-MODEL",
        "none",
    ),
    LoopAdmissionRefusal(
        "tree-only-model",
        "a loop request on a compiled model without its NLO UFO sidecar (a tree model)",
        "v2 reads loop content, counterterms and declarations from the exact "
        "raw NLO source",
        "an NLO UFO source (e.g. loop_sm, loop_qcd_qed_sm, loop_qcd_qed_sm_Gmu)",
        "m-sm MS-MODEL",
        "none",
    ),
    LoopAdmissionRefusal(
        "fft-basis-trace",
        "color.fft_basis = 'trace' for a one-loop request",
        "the one-loop seed FFT (design §7.3) acts on the external adjoint labels",
        "color.fft_basis = 'adjoint', or color.contraction = 'direct'",
        "MQ-FFT",
        "none",
    ),
    LoopAdmissionRefusal(
        "colour-selectors",
        "helicity or colour selectors (process.selected_source_helicities, "
        "selected_color_sector_ids, max_color_sectors, reference_color_order) "
        "at next-to-leading or full colour",
        "v2 selects the flows of the leading-colour plan (MQ-SELECT)",
        "color.accuracy = 'lc' with the selectors, or inclusive selectors",
        "MQ-SELECT",
        "none",
    ),
    LoopAdmissionRefusal(
        "process-aliases",
        "permutation aliases of a one-loop request (process aliases)",
        "plan v4 artifacts publish no permutation-alias records",
        "request each permuted process as its own entry",
        "m-sm MS-MODEL",
        "M-FINAL (m-sm PLAN §1.7)",
    ),
    LoopAdmissionRefusal(
        "process-group-analysis",
        "a numerical process-group analysis (process.analyze_process_groups) "
        "of a one-loop request",
        "plan v4 artifacts group processes by an exact species-substitution "
        "proof only (process.flavor_blind), never numerically",
        "process.analyze_process_groups = 'none' (process.flavor_blind groups "
        "identity-port flavour members exactly)",
        "m-sm MS-MODEL",
        "M-FINAL (m-sm PLAN §1.7)",
    ),
    LoopAdmissionRefusal(
        "direct-r2",
        "loop.r2_strategy = 'direct'",
        "the model's R2 rules are the production path "
        "(REQUIREMENTS_ADDENDUM J); direct rational R2 comes after M-FINAL",
        "loop.r2_strategy = 'model'",
        "m-sm MS-MODEL",
        "after M-FINAL",
    ),
    LoopAdmissionRefusal(
        "relation-discovery",
        "tree numerical current relation discovery "
        "(generation.relation_discovery) for a one-loop request",
        "v2 builds its tree currents structurally",
        "generation.relation_discovery.mode = 'off'",
        "m-sm MS-MODEL",
        "none",
    ),
    LoopAdmissionRefusal(
        "evaluator-backend",
        "evaluator.backend other than 'jit' for a one-loop request",
        "the loop coefficients compile with SymJIT",
        "evaluator.backend = 'jit'",
        "m-sm MS-MODEL",
        "none",
    ),
    LoopAdmissionRefusal(
        "evaluator-tuning",
        "nondefault tree evaluator tuning (evaluator.jit, .optimization, "
        ".recurrence settings, evaluator.output_chunk_size) for a one-loop "
        "request",
        "the loop compiler selects its own kernels and schedules and would "
        "not apply them (resource cores stay shared)",
        "the default evaluator settings",
        "m-sm MS-MODEL",
        "none",
    ),
    LoopAdmissionRefusal(
        "covariant-multiplets",
        "loop.preparation.covariant_multiplets for a v2 one-loop request",
        "v2 virtuals read the model's own fields",
        "no loop.preparation covariant multiplets",
        "m-sm MS-MODEL",
        "none",
    ),
)
_REFUSALS = {
    refusal.key: refusal
    for refusal in (*LOOP_ADMISSION_REFUSALS, *LOOP_CONFIGURATION_REFUSALS)
}

# Internal refusals (m-sm MS-MODEL): a condition the engine meets while
# generating a request it admitted, raised by its own checks deep in the
# generation (``refusal_key`` of the class). Fail closed: the generation
# stops with the key (``LoopEngineRefusal``) instead of a silent value; such
# a request is outside what the engine covers today. Listed by the dry run as
# ``loop_internal_refusals``; none of them is decidable before generation.
LOOP_INTERNAL_REFUSALS = (
    LoopAdmissionRefusal(
        "fermion-sign",
        "a fermion line the loop slice's sign rule does not cover "
        "(conventions.FermionSignError)",
        "every fermion line of a loop takes one relative sign against the "
        "tree engine's Wick order; a line it cannot sign would give a wrong "
        "relative sign",
        "report the request; bound its coupling orders or veto the particle",
        "v2 engine",
        "engine fix per case",
    ),
    LoopAdmissionRefusal(
        "reading-sign",
        "a vertex with no sign between the tree engine's reading and a loop "
        "segment's or root closure's (conventions.ReadingSignError)",
        "its colour tensor has no sign under the transposition of its adjoint "
        "generators",
        "report the request",
        "v2 engine",
        "engine fix per case",
    ),
    LoopAdmissionRefusal(
        "unlowered-counterterm",
        "a counterterm rule the tree recursion cannot lower whose vertex can "
        "enter a tree of the request (ct_recursion.UnloweredCountertermError)",
        "a reachable counterterm vertex is inserted exactly once or refused "
        "(REQUIREMENTS_ADDENDUM J); an unreachable one is inert",
        "report the request; bound its coupling orders",
        "v2 engine",
        "engine fix per case",
    ),
    LoopAdmissionRefusal(
        "counterterm-coupling",
        "a counterterm coupling the tree model cannot express exactly "
        "(ct_recursion.CountertermCouplingError)",
        "counterterm couplings enter the parameter program exactly, as "
        "polynomials in their counterterm scalars",
        "report the request",
        "v2 engine",
        "engine fix per case",
    ),
    LoopAdmissionRefusal(
        "loop-kernel-catalogue",
        "a model vertex without a complete exact loop kernel "
        "(catalogue.LoopKernelCatalogueError)",
        "loop lines read exact kernels of every vertex they use",
        "report the request; veto the particle or bound the coupling orders",
        "v2 engine",
        "engine fix per case",
    ),
    LoopAdmissionRefusal(
        "counterterm-completeness",
        "a divergent witness covered neither by a counterterm rule nor by a "
        "certified zero (ct_sites.CountertermCompletenessError)",
        "the model's counterterms must cover every UV-divergent structure "
        "the request's loops produce",
        "report the request (the model's declaration of completeness does "
        "not hold for it)",
        "v2 engine",
        "engine fix per case",
    ),
)
_INTERNAL = {refusal.key: refusal for refusal in LOOP_INTERNAL_REFUSALS}


def refusal_text(key: str, message: str) -> str:
    """The text of a listed refusal (admission or configuration): the key,
    the message, the rule and the remedy."""

    refusal = _REFUSALS.get(key)
    if refusal is None:
        raise ValueError(f"unknown loop admission refusal {key!r}")
    return f"{key}: {message}. Rule: {refusal.rule}. Instead: {refusal.remedy}"


class LoopRequestUnsupported(GenerationError):
    """A listed structure or setting v2 generation refuses
    (``LOOP_ADMISSION_REFUSALS``, ``LOOP_CONFIGURATION_REFUSALS``); ``key``
    is its key. The message starts with the key and names the rule and the
    remedy."""

    def __init__(self, key: str, message: str) -> None:
        super().__init__(refusal_text(key, message))
        self.key = key


class LoopRequestUnsupportedValue(LoopRequestUnsupported, ValueError):
    """A listed refusal (``LoopRequestUnsupported``) raised where callers
    handle the untyped ``ValueError`` it replaces (``lorentz-slot-arity`` at
    model preparation)."""


class LoopEngineRefusal(GenerationError):
    """An internal refusal (``LOOP_INTERNAL_REFUSALS``): an engine condition
    met while generating an admitted request; ``key`` is its key."""

    def __init__(self, key: str, message: str) -> None:
        refusal = _INTERNAL.get(key)
        if refusal is None:
            raise ValueError(f"unknown loop internal refusal {key!r}")
        super().__init__(
            f"{key}: {message}. Rule: {refusal.rule}. Instead: {refusal.remedy}"
        )
        self.key = key


class ListedBlocker(str):
    """A configuration refusal found before generation (the dry run's
    unsupported feature): its text (``refusal_text``) with its ``key``."""

    key: str
    message: str

    def __new__(cls, key: str, message: str) -> ListedBlocker:
        blocker = super().__new__(cls, refusal_text(key, message))
        blocker.key = key
        blocker.message = message
        return blocker

    def error(self) -> LoopRequestUnsupported:
        return LoopRequestUnsupported(self.key, self.message)


def keyed_error(error: BaseException) -> BaseException:
    """``error`` as the typed refusal of its key: an engine error class that
    carries a listed ``refusal_key`` (``LOOP_INTERNAL_REFUSALS``, or an
    admission key such as ``anomalous-veto``) becomes ``LoopEngineRefusal``
    or ``LoopRequestUnsupported`` with the original message; a native export
    refusal (``native_trees.NativeExportUnsupported``) keeps its feature key
    (``NativeExportRefused``); an error that already has a key is returned
    as it is, any other error too."""

    if getattr(error, "key", None) is not None:
        return error
    key = getattr(type(error), "refusal_key", None)
    if key in _INTERNAL:
        return LoopEngineRefusal(key, str(error))
    if key in _REFUSALS:
        return LoopRequestUnsupported(key, str(error))
    feature = getattr(error, "feature", None)
    if isinstance(error, NotImplementedError) and feature is not None:
        return NativeExportRefused(feature, str(error))
    return error


class NativeExportRefused(GenerationError):
    """A process whose point inputs the native export does not cover (a
    listed ``native_trees.NATIVE_EXPORT_REFUSALS`` feature): a plan v4 process
    is stored with them, so it is not stored. ``key`` (= ``feature``) is the
    export refusal's key, the message starts with it."""

    def __init__(self, key: str, message: str) -> None:
        super().__init__(
            message if message.startswith(f"{key}:") else f"{key}: {message}"
        )
        self.key = key
        self.feature = key


def listed_refusal_keys() -> tuple[str, ...]:
    """The admission keys (``LOOP_ADMISSION_REFUSALS``)."""

    return tuple(refusal.key for refusal in LOOP_ADMISSION_REFUSALS)


def listed_configuration_keys() -> tuple[str, ...]:
    return tuple(refusal.key for refusal in LOOP_CONFIGURATION_REFUSALS)


def listed_internal_keys() -> tuple[str, ...]:
    return tuple(_INTERNAL)


# ------------------------------------------------------------ N_c exactness


def mixed_grade_blocks(blocks: Any) -> tuple[Any, ...]:
    """The loop-side tree blocks of ``blocks`` (``TreeBlocks``) that are not
    exact in N_c: their trees mix U(1)-subtraction grades
    (``BlockColourFlow.nc_grade`` is ``None``). ``SeedColour.formal_nc`` is
    true exactly when there is none."""

    return tuple(
        block
        for block in blocks.blocks
        if block.key.auxiliary_kind is None and block.colour_flow.nc_grade is None
    )


def _block_currents(blocks: Iterable[Any], limit: int = 4) -> str:
    """The distinct tree currents of ``blocks`` (their ordered external legs
    and particle; blocks of one current differ by colour, helicity or
    chirality state), the first ``limit`` of them spelled out."""

    currents = dict.fromkeys(
        (tuple(block.key.word), block.key.particle_id) for block in blocks
    )
    listed = [
        f"legs {' '.join(map(str, word))} as PDG {particle}"
        for word, particle in currents
    ]
    shown = ", ".join(listed[:limit])
    more = len(listed) - limit
    return f"{len(listed)} current(s): {shown}" + (
        f", and {more} more" if more > 0 else ""
    )


def require_nc_exact(blocks: Any, accuracy: str, process: str) -> None:
    """Refuse (``colour-accuracy-nc-exact``) an LC or NLC request for
    ``process`` (its expression) whose loop-side tree blocks are not exact in
    N_c (``mixed_grade_blocks``); full colour needs no grade."""

    if accuracy == "full":
        return
    mixed = mixed_grade_blocks(blocks)
    if mixed:
        raise LoopRequestUnsupported(
            "colour-accuracy-nc-exact",
            f"{process!r}: {accuracy.upper()} needs N_c-exact tree currents, and "
            f"{len(mixed)} loop-side tree block(s) mix U(1)-subtraction grades "
            f"({_block_currents(mixed)})",
        )


def require_formal_nc_tables(accuracy: str, tables: Iterable[Any], what: str) -> None:
    """The export's assertion of ``require_nc_exact``: refuse an LC or NLC
    export from any seed table over blocks that are not exact in N_c
    (``SeedTable.formal_nc`` false), so no stored plan carries such a
    projection."""

    if accuracy == "full":
        return
    if any(not table.formal_nc for table in tables):
        raise LoopRequestUnsupported(
            "colour-accuracy-nc-exact",
            f"the {what} of an {accuracy.upper()} request reads seed tables over "
            "tree blocks that mix U(1)-subtraction grades",
        )


# ------------------------------------------------- declared colour coverage


def _covered(continuation: Any, rule: Any) -> bool:
    keys = continuation.kernel_keys
    return all(entry.key in keys for entry in rule.entries)


def _rule_label(name: str, rule: Any) -> str:
    entry = rule.entries[0]
    return f"{name} [{' '.join(entry.particle_names)}]"


def require_declared_colour_rules(
    continuation: Any,
    rules: Mapping[str, Any],
    reachability: Callable[[], Callable[[Any], bool]] | None,
    process: str,
) -> None:
    """Refuse (``colour-declaration-coverage``) an LC or NLC request
    (``continuation`` its ``SourceColorContinuation``; None at full colour)
    whose counterterm ``rules`` (name -> prepared rule) hold a kernel the
    continuation does not cover and whose vertex can enter a tree of the
    request: ``reachability`` builds the ``rule -> bool`` test
    (``ct_closure.InsertionReachability.admits``), only when a rule is
    uncovered; None means every rule can."""

    if continuation is None:
        return
    uncovered = {
        name: rule for name, rule in rules.items() if not _covered(continuation, rule)
    }
    if not uncovered:
        return
    admits = None if reachability is None else reachability()
    reached = sorted(
        name for name, rule in uncovered.items() if admits is None or admits(rule)
    )
    if reached:
        shown = "; ".join(_rule_label(name, uncovered[name]) for name in reached[:6])
        more = f"; and {len(reached) - 6} more" if len(reached) > 6 else ""
        raise LoopRequestUnsupported(
            "colour-declaration-coverage",
            f"{process!r}: its colour declaration states no N_c continuation of "
            f"the counterterm vertices {shown}{more}, which can enter a tree of "
            "the request",
        )


def require_declared_colour_fields(
    continuation: Any, insertions: Iterable[Any], process: str
) -> None:
    """Refuse (``colour-declaration-coverage``) an LC or NLC request
    (``continuation`` its ``SourceColorContinuation``; None at full colour)
    whose external-field ``insertions`` (``wavefunction.ExternalFieldInsertion``)
    take a field factor the continuation does not cover."""

    if continuation is None:
        return
    keys = continuation.field_keys
    missing = sorted(
        {
            (rule.particle_id, rule.correction, rule.loop_group_index)
            for rule in (insertion.rule for insertion in insertions)
        }
        - set(keys)
    )
    if missing:
        raise LoopRequestUnsupported(
            "colour-declaration-coverage",
            f"{process!r}: its colour declaration states no N_c continuation of "
            "the external-field factors (particle record, correction, loop "
            f"group) {missing}, which its external legs take",
        )


# ------------------------------------------------- fermion-line representation


def _sign_ownership(record: Any) -> bool | None:
    """Whether the colour-singlet (Wick) convention signs the fermion of a
    compiled particle record in a loop-request DAG (True), the colour flow
    (False), or None: not a fermion, or a colour representation the fermion
    ordering refuses on its own. These are the predicates both DAG lanes use
    (``CompiledUFOModel.is_fermion`` and ``color_rep`` read the same record
    fields; ``generation.fermion_ordering._fermion_rep_requires_exterior_sign``)."""

    from pyamplicol.generation.fermion_ordering import (
        _fermion_rep_requires_exterior_sign,
    )

    if record.statistics != "fermion":
        return None
    try:
        return _fermion_rep_requires_exterior_sign(record.color, False)
    except ValueError:
        return None


def representation_changing_vertices(
    compiled: Any,
    process: Any,
    *,
    order_limits: Iterable[Mapping[str, int]],
    ignored_particle_ids: Iterable[int] = (),
    loop_species: Iterable[int] = (),
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The vertices (UFO name, UFO particles) of ``compiled`` whose fermions have
    different sign ownership (``_sign_ownership``) and that a tree or a loop
    of ``process`` can use: an oriented kernel admitted as the DAG admits it
    (no particle in ``ignored_particle_ids``, its coupling orders within one
    of ``order_limits``, ``dag_reachability._coupling_orders_within_limits``)
    whose inputs are both currents some tree of the external legs can
    produce (a tree vertex), or one such current and a loop species whose
    result is a loop species too (a loop vertex; contact auxiliaries count as
    loop species). The producible currents are the fixed point of the admitted
    kernels from the legs' particles, which over-approximates the trees
    (subsets are not kept disjoint), so a vertex any tree or loop uses is
    always found."""

    from pyamplicol.generation.dag_reachability import (
        _coupling_orders_within_limits,
        _normalize_coupling_order_limits,
    )

    record_of = {record.pdg_code: record for record in compiled.ir.particles}
    pdg_of = {record.name: record.pdg_code for record in compiled.ir.particles}
    ignored = {int(pdg) for pdg in ignored_particle_ids}
    limits = [_normalize_coupling_order_limits(limit) for limit in order_limits]
    kernels = []
    for kernel in compiled.ir.oriented_kernels:
        pdgs = tuple(pdg_of[name] for name in kernel.particles)
        if ignored & set(pdgs):
            continue
        if not any(
            _coupling_orders_within_limits(kernel.coupling_orders, limit)
            for limit in limits
        ):
            continue
        kernels.append((kernel, pdgs))
    producible = {
        int(pdg)
        for leg in process.legs
        for pdg in (leg.outgoing_pdg, leg.pdg)
        if pdg is not None
    }
    grown = True
    while grown:
        grown = False
        for _kernel, (left, right, result) in kernels:
            if left in producible and right in producible and result not in producible:
                producible.add(result)
                grown = True
    # A loop vertex takes the loop line in at one input and out at its result
    # (a loop species, or a contact auxiliary of a decomposed vertex).
    auxiliaries = {
        record.pdg_code for record in compiled.ir.particles if record.auxiliary_kind
    }
    loop = ({int(pdg) for pdg in loop_species} | auxiliaries) - ignored
    # A vertex is named by its UFO name and particles (its oriented kernels
    # are its orientations and contact decompositions).
    particles_of: dict[str, tuple[str, ...]] = {}
    for term in compiled.ir.vertex_terms:
        particles_of.setdefault(term.vertex, tuple(term.particles))
    found: dict[tuple[str, tuple[str, ...]], None] = {}
    for kernel, pdgs in kernels:
        left, right, result = pdgs
        used = (left in producible and right in producible) or (
            result in loop
            and (
                (left in loop and right in producible)
                or (right in loop and left in producible)
            )
        )
        if not used:
            continue
        owners = {_sign_ownership(record_of[pdg]) for pdg in pdgs}
        owners.discard(None)
        if len(owners) > 1:
            found.setdefault((kernel.vertex, particles_of[kernel.vertex]), None)
    return tuple(found)


def require_colour_preserving_fermion_lines(
    compiled: Any,
    process: Any,
    *,
    order_limits: Iterable[Mapping[str, int]],
    ignored_particle_ids: Iterable[int] = (),
    loop_species: Iterable[int] = (),
) -> None:
    """Refuse (``fermion-line-representation``) a request that can use a
    vertex joining fermions of different sign ownership
    (``representation_changing_vertices``), naming the vertices."""

    found = representation_changing_vertices(
        compiled,
        process,
        order_limits=order_limits,
        ignored_particle_ids=ignored_particle_ids,
        loop_species=loop_species,
    )
    if found:
        names = ", ".join(
            f"{vertex} ({' '.join(particles)})" for vertex, particles in found[:4]
        )
        more = f", and {len(found) - 4} more" if len(found) > 4 else ""
        raise LoopRequestUnsupported(
            "fermion-line-representation",
            f"{process.process!r}: the vertex {names}{more} joins a coloured "
            "and a colour-singlet fermion on one fermion line",
        )


# ------------------------------------------- veto of unlabelled counterterms


def unlabelled_counterterm_vertices(
    kernels: Iterable[Any], corrections: Iterable[str]
) -> tuple[str, ...]:
    """The UFO counterterm vertices (names, sorted) of a loop model's
    ``kernels`` (``LoopModelIR.kernels``: every vertex of the model) with a
    term of a correction in ``corrections`` (its coupling orders hold one)
    that declares no loop content: its UFO ``loop_particles`` group is empty
    (``()``, the counterterm admission's wildcard)."""

    corrections = frozenset(corrections)
    return tuple(
        sorted(
            {
                entry.vertex_name
                for entry in kernels
                if entry.kind != "tree"
                and entry.loop_particles is not None
                and () in entry.loop_particles
                and corrections & {name for name, power in entry.order if power}
            }
        )
    )


def require_labelled_counterterms_for_veto(
    kernels: Iterable[Any],
    process: str,
    veto: Iterable[int],
    corrections: Iterable[str],
    species_orders: Mapping[int, Iterable[str]],
) -> None:
    """Refuse (``veto-unlabelled-counterterms``) a request for ``process``
    (its expression) whose ``veto`` (|PDG| codes) removes a species
    perturbative in one of its ``corrections`` (``species_orders``: |PDG| ->
    orders, the prepared structure's ``perturbative_species_orders``) while
    the model's counterterm terms of those corrections are not all labelled
    with their loop content (``unlabelled_counterterm_vertices``)."""

    corrections = frozenset(corrections)
    vetoed = sorted(
        pdg
        for pdg in {abs(int(code)) for code in veto}
        if corrections & set(species_orders.get(pdg, ()))
    )
    if not vetoed:
        return
    unlabelled = unlabelled_counterterm_vertices(kernels, corrections)
    if unlabelled:
        shown = ", ".join(unlabelled[:4])
        more = f", and {len(unlabelled) - 4} more" if len(unlabelled) > 4 else ""
        raise LoopRequestUnsupported(
            "veto-unlabelled-counterterms",
            f"{process!r}: particle_veto removes the species {vetoed}, "
            f"perturbative in the corrections {sorted(corrections)}, and the "
            f"model's counterterm vertices {shown}{more} write terms of those "
            "corrections without loop content (loop_particles = []), so the "
            "veto cannot remove the vetoed loops from its counterterms",
        )


__all__ = [
    "LOOP_ADMISSION_REFUSALS",
    "LOOP_CONFIGURATION_REFUSALS",
    "LOOP_INTERNAL_REFUSALS",
    "ListedBlocker",
    "LoopAdmissionRefusal",
    "LoopEngineRefusal",
    "LoopRequestUnsupported",
    "LoopRequestUnsupportedValue",
    "NativeExportRefused",
    "keyed_error",
    "listed_configuration_keys",
    "listed_internal_keys",
    "listed_refusal_keys",
    "mixed_grade_blocks",
    "refusal_text",
    "representation_changing_vertices",
    "require_colour_preserving_fermion_lines",
    "require_declared_colour_fields",
    "require_declared_colour_rules",
    "require_formal_nc_tables",
    "require_labelled_counterterms_for_veto",
    "require_nc_exact",
    "unlabelled_counterterm_vertices",
]
