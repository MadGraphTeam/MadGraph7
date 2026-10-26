# SPDX-License-Identifier: 0BSD
"""Generation-time external-fermion ordering signs.

Numerical current kernels commute their already-evaluated inputs.  Wick
contractions do not: every binary current merge must first restore the
canonical source order of its external fermions, and a two-fermion kernel has
one additional sign when its oriented inputs are particle--antiparticle rather
than antiparticle--particle.  Folding this exact ``+/-1`` into the generated
coefficient realizes the exterior algebra without adding runtime state or
splitting currents by factorially many Wick lineages.

Colour-singlet fermions always take these signs. Coloured (fundamental) open
lines have two conventions:

* **Wick** (``coloured_lines=True``, tree programs with amplitude roots). The
  merges sign coloured fermions as well. The Born colour metric multiplies a
  closed open-line sector by the parity of its fundamental--antifundamental
  pairing (``color/correlator_matrices.py``); that parity is the Wick sign, up
  to ``(-1)^(lines-1)``, only when every link between quark lines is a gluon,
  because a gluon transposes the colour ends of the two lines it joins. A
  colour-singlet link (photon, Z, W, or the U(1) part of a gluon) leaves them
  in place. Each amplitude root therefore divides out the colour parity of its
  sector times ``(-1)^(lines-1)`` (``open_line_closure_sign``), and a U(1)
  subtraction current, whose ``-1/N_c`` sign the colour parity used to supply,
  carries that ``-1`` where it is created. Roots of gluon-joined lines keep
  their sign, and QCD x electroweak interference between quark lines gets its
  Wick sign. The Rust recurrence builder applies the same factors
  (``rusticol-core`` ``recurrence/fermion_ordering.rs``) except the global
  ``gluon_joined_phase``, so the two lanes' amplitudes may differ by an
  overall sign; squared results and their per-order partials agree.
* **Colour flow** (``coloured_lines=False``, loop-request blocks). The colour
  flow owns the ordering of coloured lines; ``loops/v2/blocks.py`` documents
  how the loop engine turns those currents into Feynman-rule currents.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ..color.plan_types import LCColorSector
from ..models.base import Model
from ..models.contracts import U1_SUBTRACTION_AUXILIARY_KIND
from ..processes.ir import CanonicalProcessIR, ProcessLegIR
from .dag_types import AmplitudeRoot, CurrentIndex, GenericDAG, InteractionNode


def _external_fermion_requires_exterior_sign(
    statistics: str,
    color_role: str,
    coloured_lines: bool = False,
) -> bool:
    """Classify one canonical source without guessing unsupported color roles."""

    if statistics != "fermion":
        return False
    if color_role == "singlet":
        return True
    if color_role in {"fundamental", "antifundamental"}:
        return coloured_lines
    raise ValueError(
        f"external-fermion ordering does not support fermion color role {color_role!r}"
    )


def _fermion_rep_requires_exterior_sign(
    color_representation: int, coloured_lines: bool = False
) -> bool:
    """Return sign ownership for a model fermion in the supported UFO scope."""

    color_representation = int(color_representation)
    if color_representation == 1:
        return True
    if abs(color_representation) == 3:
        return coloured_lines
    raise ValueError(
        "external-fermion ordering does not support fermion color "
        f"representation {color_representation}"
    )


def _external_fermion_support_sign(
    left_labels: tuple[int, ...],
    right_labels: tuple[int, ...],
    fermion_source_ranks: dict[int, int],
) -> int:
    """Return the Koszul sign for concatenating two disjoint supports."""

    left_ranks = tuple(
        fermion_source_ranks[label]
        for label in left_labels
        if label in fermion_source_ranks
    )
    right_ranks = tuple(
        fermion_source_ranks[label]
        for label in right_labels
        if label in fermion_source_ranks
    )
    inversions = sum(
        left_rank > right_rank for left_rank in left_ranks for right_rank in right_ranks
    )
    return -1 if inversions % 2 else 1


def _fermion_input_orientation_sign(
    left_is_fermion: bool,
    left_orientation: str,
    left_requires_exterior_sign: bool,
    right_is_fermion: bool,
    right_orientation: str,
    right_requires_exterior_sign: bool,
) -> int:
    """Return the field-order sign of one oriented two-fermion kernel."""

    if not (left_is_fermion and right_is_fermion):
        return 1
    if left_requires_exterior_sign != right_requires_exterior_sign:
        raise ValueError(
            "external-fermion ordering encountered mixed color-sign ownership"
        )
    if not left_requires_exterior_sign:
        return 1
    supported = {"particle", "antiparticle"}
    if left_orientation not in supported or right_orientation not in supported:
        raise ValueError(
            "external-fermion ordering requires non-self-conjugate model orientations"
        )
    pair = (left_orientation, right_orientation)
    if pair == ("antiparticle", "particle"):
        return 1
    if pair == ("particle", "antiparticle"):
        return -1
    raise ValueError("external-fermion ordering requires opposite Dirac orientations")


def open_line_closure_sign(sector: LCColorSector | None) -> int:
    """The sign that divides the colour parity of a closed open-line sector out.

    For ``k`` open lines it is the parity of the fundamental-to-antifundamental
    pairing (lines listed by ascending fundamental label, as the Born colour
    metric lists them) times ``(-1)^(k-1)``, the colour parity of a tree whose
    lines are all joined by gluons. Other sector kinds have no open line.
    """

    if sector is None or sector.kind != "open-lines":
        return 1
    antifundamentals = tuple(
        line.antifundamental_label
        for line in sorted(
            sector.open_color_lines, key=lambda line: line.fundamental_label
        )
    )
    inversions = sum(
        first > second
        for index, first in enumerate(antifundamentals)
        for second in antifundamentals[index + 1 :]
    )
    return -1 if (inversions + len(antifundamentals) - 1) % 2 else 1


class FermionOrderingSign:
    """Cached model/process contract for cold DAG construction."""

    __slots__ = (
        "_coloured_lines",
        "_merge_cache",
        "_model",
        "_particle_contract_cache",
        "_source_ranks",
    )

    def __init__(
        self,
        process: CanonicalProcessIR,
        model: Model,
        *,
        coloured_lines: bool = False,
    ) -> None:
        self._model = model
        self._coloured_lines = bool(coloured_lines)
        source_ranks: dict[int, int] = {}
        for source_slot, leg in enumerate(process.legs):
            if _external_fermion_requires_exterior_sign(
                leg.statistics,
                leg.color_role,
                self._coloured_lines,
            ):
                source_ranks[int(leg.label)] = source_slot
        self._source_ranks = source_ranks
        self._particle_contract_cache: dict[int, tuple[bool, str, bool]] = {}
        self._merge_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...], int | None, int | None], int
        ] = {}

    @property
    def active(self) -> bool:
        """Whether this process has any source whose sign is owned here."""

        return bool(self._source_ranks)

    def tree_sign(self, lines: Sequence[tuple[ProcessLegIR, ProcessLegIR]]) -> int:
        """Product of the merge signs of one tree with these fermion lines.

        Each line closes its two external ends at a two-fermion merge, and the
        closed lines join one by one through bosons. The exterior algebra
        makes the product independent of the tree's shape.
        """

        sign = 1
        joined: tuple[int, ...] = ()
        for first, second in lines:
            ends = (int(first.label),), (int(second.label),)
            sign *= self._merge_sign(
                *ends, int(first.outgoing_pdg or 0), int(second.outgoing_pdg or 0)
            )
            line = tuple(sorted((*ends[0], *ends[1])))
            sign *= self._merge_sign(joined, line, None, None)
            joined = tuple(sorted((*joined, *line)))
        return sign

    def _particle_contract(self, particle_id: int) -> tuple[bool, str, bool]:
        particle_id = int(particle_id)
        cached = self._particle_contract_cache.get(particle_id)
        if cached is not None:
            return cached
        is_fermion = bool(self._model.is_fermion(particle_id))
        orientation = (
            str(self._model.source_orientation(particle_id))
            if is_fermion
            else "self-conjugate"
        )
        requires_exterior_sign = (
            _fermion_rep_requires_exterior_sign(
                self._model.color_rep(particle_id), self._coloured_lines
            )
            if is_fermion
            else False
        )
        result = (is_fermion, orientation, requires_exterior_sign)
        self._particle_contract_cache[particle_id] = result
        return result

    def merge_sign(self, left: CurrentIndex, right: CurrentIndex) -> int:
        return self._merge_sign(
            left.external_labels,
            right.external_labels,
            int(left.particle_id),
            int(right.particle_id),
        )

    def _merge_sign(
        self,
        left_labels: tuple[int, ...],
        right_labels: tuple[int, ...],
        left_particle_id: int | None,
        right_particle_id: int | None,
    ) -> int:
        """Sign of one merge; a ``None`` particle id is a boson current."""

        key = (left_labels, right_labels, left_particle_id, right_particle_id)
        cached = self._merge_cache.get(key)
        if cached is not None:
            return cached
        boson = (False, "self-conjugate", False)
        (
            left_is_fermion,
            left_orientation,
            left_requires_exterior_sign,
        ) = (
            boson
            if left_particle_id is None
            else self._particle_contract(left_particle_id)
        )
        (
            right_is_fermion,
            right_orientation,
            right_requires_exterior_sign,
        ) = (
            boson
            if right_particle_id is None
            else self._particle_contract(right_particle_id)
        )
        sign = _external_fermion_support_sign(
            left_labels,
            right_labels,
            self._source_ranks,
        )
        sign *= _fermion_input_orientation_sign(
            left_is_fermion,
            left_orientation,
            left_requires_exterior_sign,
            right_is_fermion,
            right_orientation,
            right_requires_exterior_sign,
        )
        self._merge_cache[key] = sign
        return sign


def gluon_joined_phase(process: CanonicalProcessIR, model: Model) -> int:
    """The constant that keeps gluon-joined roots unchanged by the Wick signs.

    For a tree whose coloured lines are joined by gluons, the Wick sign times
    ``open_line_closure_sign`` of its sector is the same for every tree of a
    process; this is that constant times the colour-flow convention's sign of
    the same tree, so multiplying every root by it leaves those roots exactly
    as the colour-flow convention builds them. It is evaluated on a reference
    tree: fundamental ends paired with antifundamental ends in label order
    (the identity colour pairing), and colour-singlet fermions paired
    particle with antiparticle in label order.
    """

    lines: list[tuple[ProcessLegIR, ProcessLegIR]] = []
    fermions = [leg for leg in process.legs if leg.statistics == "fermion"]
    for first_role, second_role in (
        ("fundamental", "antifundamental"),
        ("particle", "antiparticle"),
    ):
        if first_role == "fundamental":
            first = [leg for leg in fermions if leg.color_role == first_role]
            second = [leg for leg in fermions if leg.color_role == second_role]
        else:
            singlets = [leg for leg in fermions if leg.color_role == "singlet"]
            first, second = (
                [
                    leg
                    for leg in singlets
                    if model.source_orientation(int(leg.outgoing_pdg or 0)) == role
                ]
                for role in (first_role, second_role)
            )
        if len(first) != len(second):
            raise ValueError(
                "external-fermion ordering cannot pair the external fermions"
            )
        lines.extend(
            zip(
                sorted(first, key=lambda leg: leg.label),
                sorted(second, key=lambda leg: leg.label),
                strict=True,
            )
        )
    wick = FermionOrderingSign(process, model, coloured_lines=True)
    flow = FermionOrderingSign(process, model, coloured_lines=False)
    return wick.tree_sign(lines) * flow.tree_sign(lines)


def apply_fermion_ordering_signs(
    dag: GenericDAG, model: Model, *, coloured_lines: bool = False
) -> GenericDAG:
    """Fold every merge sign into existing DAG constants exactly once.

    ``coloured_lines`` selects the Wick convention for coloured fermions
    (module docstring); it is for programs that close amplitude roots.
    """

    if not dag.interactions and not dag.amplitude_roots:
        return dag
    ordering = FermionOrderingSign(dag.process, model, coloured_lines=coloured_lines)
    if not ordering.active:
        return dag
    currents = dag.currents
    color_plan = dag.color_plan
    phase = gluon_joined_phase(dag.process, model) if coloured_lines else 1

    def signed_weight(
        weight: tuple[float, float], left_id: int, right_id: int, factor: int = 1
    ) -> tuple[float, float]:
        sign = factor * ordering.merge_sign(
            currents[left_id].index,
            currents[right_id].index,
        )
        return weight if sign == 1 else (-weight[0], -weight[1])

    def creation_sign(result_id: int) -> int:
        if (
            coloured_lines
            and currents[result_id].index.auxiliary_kind
            == U1_SUBTRACTION_AUXILIARY_KIND
        ):
            return -1
        return 1

    def closure_sign(root: AmplitudeRoot) -> int:
        if not coloured_lines:
            return 1
        sector_id = root.color_sector_id
        if sector_id is None:
            sector_id = currents[root.left_id].index.color_state.sector_id
        return phase * open_line_closure_sign(
            None if sector_id is None else color_plan.sector(sector_id)
        )

    def signed_interaction(interaction: InteractionNode) -> InteractionNode:
        color_weight = signed_weight(
            interaction.color_weight,
            interaction.left_id,
            interaction.right_id,
            creation_sign(interaction.result_id),
        )
        if color_weight is interaction.color_weight:
            return interaction
        return replace(interaction, color_weight=color_weight)

    def signed_root(root: AmplitudeRoot) -> AmplitudeRoot:
        color_weight = signed_weight(
            root.color_weight,
            root.left_id,
            root.right_id,
            closure_sign(root),
        )
        if color_weight is root.color_weight:
            return root
        return replace(root, color_weight=color_weight)

    return replace(
        dag,
        interactions=tuple(signed_interaction(row) for row in dag.interactions),
        amplitude_roots=tuple(signed_root(row) for row in dag.amplitude_roots),
    )


def _coloured_fermion_endpoints(dag: GenericDAG, model: Model) -> frozenset[int]:
    """Labels of the coloured fermion legs (``|colour rep| == 3``)."""

    return frozenset(
        leg.label
        for leg in dag.process.legs
        if leg.outgoing_pdg is not None
        and abs(model.color_rep(int(leg.outgoing_pdg))) == 3
        and model.is_fermion(int(leg.outgoing_pdg))
    )


def colour_connection_parities(
    dag: GenericDAG, model: Model
) -> tuple[list[int | None], frozenset[int]]:
    """The colour-connection parity of every current of a loop-request DAG
    (colour-flow convention) and the ids of the interactions whose
    derivation has the other parity.

    A merge connects two coloured fermion lines when both inputs hold
    coloured fermion endpoints in their open words and one of them is a
    colour-octet current (a gluon, contact or U(1)-subtraction auxiliary);
    a colour singlet merged into a current that is no coloured fermion
    connects nothing (its U(1)-subtraction sign is a colour weight of the
    stage that absorbs it). Against Feynman rules a tree of the colour-flow
    convention carries ``(-1)`` per connection (``loops/v2/blocks.py``).

    A current whose trees pair its coloured fermion endpoints in more than
    one way can be derived with both parities: with identical-flavour
    lines, a counterterm stage closing the gluons of the pairing
    (a, b~)(b, a~) and one taking the U(1)-subtraction auxiliary of the
    pairing (a, a~)(b, b~) give the same closed lines. Every current of one
    block (its index without the helicity ancestry) then takes the parity
    of the block's first derivation (interactions by result size, in DAG
    order), and the derivations with the other parity are returned:
    ``apply_colour_connection_signs`` folds their ``-1`` into their colour
    weights.
    """

    currents = dag.currents
    endpoints = _coloured_fermion_endpoints(dag, model)

    def octet(index: CurrentIndex) -> bool:
        return (
            model.color_rep(index.particle_id) == 8
            or index.auxiliary_kind == U1_SUBTRACTION_AUXILIARY_KIND
        )

    def singlet(index: CurrentIndex) -> bool:
        return (
            model.color_rep(index.particle_id) == 1
            or index.auxiliary_kind == U1_SUBTRACTION_AUXILIARY_KIND
        )

    def coloured_fermion(particle_id: int) -> bool:
        return model.is_fermion(particle_id) and abs(model.color_rep(particle_id)) == 3

    def has_endpoints(index: CurrentIndex) -> bool:
        return any(label in endpoints for label in index.ordered_external_labels)

    def block(index: CurrentIndex) -> tuple[object, ...]:
        return (
            index.particle_id,
            index.external_mask,
            index.chirality,
            index.spin_state,
            index.flavour_flow,
            index.quantum_number_flow,
            index.color_state,
            index.coupling_orders,
            index.auxiliary_kind,
            index.ordered_external_labels,
        )

    parity: list[int | None] = [
        0 if current.is_source else None for current in currents
    ]
    block_parity: dict[tuple[object, ...], int] = {}
    deviating: set[int] = set()
    for interaction in sorted(
        dag.interactions,
        key=lambda row: currents[row.result_id].index.external_mask.bit_count(),
    ):
        left = currents[interaction.left_id].index
        right = currents[interaction.right_id].index
        connects = (
            has_endpoints(left)
            and has_endpoints(right)
            and (octet(left) or octet(right))
            and (
                coloured_fermion(currents[interaction.result_id].index.particle_id)
                or not (singlet(left) or singlet(right))
            )
        )
        left_parity = parity[interaction.left_id]
        right_parity = parity[interaction.right_id]
        if left_parity is None or right_parity is None:
            raise ValueError(
                f"interaction {interaction.id} reads a current without a derivation"
            )
        value = left_parity ^ right_parity ^ int(connects)
        known = block_parity.setdefault(
            block(currents[interaction.result_id].index), value
        )
        parity[interaction.result_id] = known
        if known != value:
            deviating.add(interaction.id)
    return parity, frozenset(deviating)


def apply_colour_connection_signs(dag: GenericDAG, model: Model) -> GenericDAG:
    """Fold ``-1`` into the colour weight of every derivation whose
    colour-connection parity differs from its current's
    (``colour_connection_parities``), so each current of a loop-request DAG
    has one parity and its blocks one connection sign. Without coloured
    fermion legs no merge connects and nothing is folded."""

    if not dag.interactions or not _coloured_fermion_endpoints(dag, model):
        return dag
    _, deviating = colour_connection_parities(dag, model)
    if not deviating:
        return dag
    return replace(
        dag,
        interactions=tuple(
            replace(row, color_weight=(-row.color_weight[0], -row.color_weight[1]))
            if row.id in deviating
            else row
            for row in dag.interactions
        ),
    )


__all__ = [
    "FermionOrderingSign",
    "apply_colour_connection_signs",
    "apply_fermion_ordering_signs",
    "colour_connection_parities",
    "gluon_joined_phase",
    "open_line_closure_sign",
]
