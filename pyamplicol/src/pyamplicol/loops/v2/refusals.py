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
  any tree, naming the vetoed species and those vertices.
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
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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
        "WP3.2 EWI-7 ([U2] veto guard; C1 UV certifier)",
        "vetoed loops removed from unlabelled counterterm couplings, or the "
        "veto certified against the loops",
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
        "a request whose tree currents propagate a coloured particle with a "
        "nonzero width (e.g. the top at a restriction with DECAY 6 1.4915)",
        "v2 virtuals are real on-shell: loop propagators carry no width and the "
        "model counterterms renormalise the real mass, so a width in the tree "
        "currents makes the Born, its colour correlations and the virtual "
        "inconsistent (a resonant coloured particle needs the complex-mass "
        "scheme)",
        "a restriction with every width at zero (model.restriction = "
        '"no_widths", MG5_aMC loop_sm-no_widths), or the width fixed to zero '
        'in [loop.preparation] fixed_parameters (e.g. WT = "0")',
        "M-SM (complex-mass scheme; user ruling 2026-10-28 ~15:25 (3))",
        "widths of coloured particles in the complex-mass scheme",
    ),
    LoopAdmissionRefusal(
        "dirac-weyl-closure",
        "a fermion line that joins massive and massless fermions through a "
        "chiral vertex and closes a massive (Dirac) current on a massless "
        "(Weyl) one (e.g. b b~ > w+ w- with a massless b and the t channel)",
        "models install no Dirac-Weyl closure contraction, so such a closure "
        "would silently drop its diagrams",
        "keep the light fermion massive in the tree model (the default "
        "restriction with the mass fixed to zero in [loop.preparation] "
        'fixed_parameters, e.g. MB = "0" and ymb = "0", instead of a '
        "restriction that removes the mass)",
        "MQ-5F",
        "Dirac-Weyl closures in loop Born admission",
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
)
_REFUSALS = {refusal.key: refusal for refusal in LOOP_ADMISSION_REFUSALS}


class LoopRequestUnsupported(GenerationError):
    """A listed structure v2 generation refuses (``LOOP_ADMISSION_REFUSALS``);
    ``key`` is its key. The message starts with the key and names the rule
    and the remedy."""

    def __init__(self, key: str, message: str) -> None:
        refusal = _REFUSALS.get(key)
        if refusal is None:
            raise ValueError(f"unknown loop admission refusal {key!r}")
        super().__init__(
            f"{key}: {message}. Rule: {refusal.rule}. Instead: {refusal.remedy}"
        )
        self.key = key


class LoopRequestUnsupportedValue(LoopRequestUnsupported, ValueError):
    """A listed refusal (``LoopRequestUnsupported``) raised from code that
    tree requests share, whose callers handle the untyped ``ValueError`` it
    replaces (the Dirac-Weyl closure of the tree DAG)."""


def listed_refusal_keys() -> tuple[str, ...]:
    return tuple(_REFUSALS)


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
    "LoopAdmissionRefusal",
    "LoopRequestUnsupported",
    "listed_refusal_keys",
    "mixed_grade_blocks",
    "representation_changing_vertices",
    "require_colour_preserving_fermion_lines",
    "require_formal_nc_tables",
    "require_labelled_counterterms_for_veto",
    "require_nc_exact",
    "unlabelled_counterterm_vertices",
]
