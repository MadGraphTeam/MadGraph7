# SPDX-License-Identifier: 0BSD
"""Direct exact connected matrices in the generated tree colour basis.

The adapter adds the endpoint-pairing phase used by AmpliCol's ordinary metric
to otherwise literal colour tensors. Every stored entry is directed and means
``<bra-connected left sector | ket-connected right sector>``; no upper-triangle
doubling or assumption of Hermiticity is implicit. Runtime contraction therefore
uses ``conjugate(a[left]) * entry * a[right]``.

Only complete full-colour plans are accepted. The ordinary matrix's zero pattern
is deliberately not consulted: a connection can turn a zero overlap into a
nonzero one. Planning is direct and quadratic in the selected basis size, with
each connected tensor reused across all requested matrix entries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..processes.ir import CanonicalProcessIR
from .connections import (
    COLOR_CONNECTION_CONVENTION,
    MAX_CONNECTION_ORDER,
    ColorConnection,
    ColorEmission,
    ColorLeg,
    ColorTensor,
    ConnectedColorTensor,
    EmitGluon,
    ExactColorCoefficient,
    OpenColorString,
    SplitGluon,
    TensorTerm,
    _validate_tensor_legs,
    apply_color_connection,
    contract_connected_tensors,
)
from .plan_types import GenericColorPlan, LCColorSector

_COLOR_ROLE_REPRESENTATIONS = {
    "singlet": 1,
    "fundamental": 3,
    "antifundamental": -3,
    "adjoint": 8,
}


def _emission_json(emission: ColorEmission) -> dict[str, object]:
    if isinstance(emission, EmitGluon):
        return {
            "kind": "emit-gluon",
            "emitter": emission.emitter_label,
            "emitted": emission.emitted_label,
        }
    return {
        "kind": "split-gluon",
        "parent": emission.parent_label,
        "quark": emission.quark_label,
        "antiquark": emission.antiquark_label,
    }


@dataclass(frozen=True)
class ColorCorrelator:
    """A named pair of ordered colour connections on the same Born amplitude.

    Labels refer to canonical external process labels (positive integers).
    Fresh negative labels are convenient for auxiliary emitted particles. The
    bra and ket must finish with identical labelled colour representations.
    """

    id: str
    bra: tuple[ColorEmission, ...] = ()
    ket: tuple[ColorEmission, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("colour correlator ID must be a nonempty string")
        object.__setattr__(self, "bra", tuple(self.bra))
        object.__setattr__(self, "ket", tuple(self.ket))
        if len(self.bra) != len(self.ket):
            raise ValueError("bra and ket must have the same number of emissions")
        if len(self.bra) > MAX_CONNECTION_ORDER:
            raise ValueError("colour correlators support at most three emissions")
        if any(
            not isinstance(op, (EmitGluon, SplitGluon)) for op in (*self.bra, *self.ket)
        ):
            raise TypeError(
                "colour correlator operations must be typed emission records"
            )

    @property
    def order(self) -> int:
        return len(self.bra)

    @classmethod
    def dipole(cls, id: str, left_leg: int, right_leg: int) -> ColorCorrelator:
        """Request ``<M|T_left . T_right|M>`` using auxiliary gluon label -1."""

        return cls(id, (EmitGluon(left_leg, -1),), (EmitGluon(right_leg, -1),))

    def to_json_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "bra": [_emission_json(op) for op in self.bra],
            "ket": [_emission_json(op) for op in self.ket],
        }


def process_color_legs(process: CanonicalProcessIR) -> tuple[ColorLeg, ...]:
    """Read all-outgoing roles from process IR; do not cross them a second time."""

    if not isinstance(process, CanonicalProcessIR):
        raise TypeError("colour correlators require a CanonicalProcessIR")
    legs = []
    for leg in process.legs:
        if type(leg.label) is not int or leg.label <= 0:
            raise ValueError(
                "canonical external colour labels must be positive integers"
            )
        try:
            representation = _COLOR_ROLE_REPRESENTATIONS[leg.color_role]
        except KeyError as exc:
            raise ValueError(
                "colour correlators require concrete resolved colour roles"
            ) from exc
        legs.append(ColorLeg(leg.label, representation))
    return ColorConnection(tuple(legs)).initial_legs


def _sector_color_tensor(
    sector: LCColorSector, legs: tuple[ColorLeg, ...]
) -> TensorTerm:
    if not isinstance(sector, LCColorSector):
        raise TypeError("colour correlator sectors must be LCColorSector records")
    if sector.kind == "open-lines":
        strings = tuple(
            OpenColorString(
                line.fundamental_label, line.adjoint_labels, line.antifundamental_label
            )
            for line in sector.open_color_lines
        )
        tensor = ColorTensor(strings)
        # sign(pairing_left) * sign(pairing_right) is exactly the permutation
        # parity (-1)^(line_count - alternating_cycle_count) in the legacy metric.
        paired_antiquarks = tuple(
            string.antifundamental_label
            for string in sorted(strings, key=lambda item: item.fundamental_label)
        )
        inversions = sum(
            first > second
            for index, first in enumerate(paired_antiquarks)
            for second in paired_antiquarks[index + 1 :]
        )
        coefficient = ExactColorCoefficient(-1 if inversions % 2 else 1)
    elif sector.kind == "single-trace":
        tensor = ColorTensor(traces=(sector.trace_labels,))
        coefficient = ExactColorCoefficient(1)
    elif sector.kind == "singlet":
        tensor = ColorTensor()
        coefficient = ExactColorCoefficient(1)
    else:
        raise ValueError(f"unsupported colour sector kind {sector.kind!r}")
    _validate_tensor_legs(tensor, legs)
    return TensorTerm(coefficient, tensor)


def sector_color_tensor(
    sector: LCColorSector, process: CanonicalProcessIR
) -> TensorTerm:
    """Convert one generated sector to a phase-weighted literal tensor."""

    return _sector_color_tensor(sector, process_color_legs(process))


@dataclass(frozen=True)
class ColorCorrelatorMatrixEntry:
    left_sector_id: int
    right_sector_id: int
    weight: ExactColorCoefficient

    def to_json_dict(self) -> dict[str, object]:
        return {
            "left_sector_id": self.left_sector_id,
            "right_sector_id": self.right_sector_id,
            "weight": {
                "real": [
                    str(self.weight.real.numerator),
                    str(self.weight.real.denominator),
                ],
                "imag": [
                    str(self.weight.imag.numerator),
                    str(self.weight.imag.denominator),
                ],
            },
        }


@dataclass(frozen=True)
class ColorCorrelatorMatrix:
    id: str
    order: int
    sector_ids: tuple[int, ...]
    output_legs: tuple[ColorLeg, ...]
    entries: tuple[ColorCorrelatorMatrixEntry, ...]
    bra: tuple[ColorEmission, ...] = ()
    ket: tuple[ColorEmission, ...] = ()

    def to_json_dict(self) -> dict[str, object]:
        request = ColorCorrelator(self.id, self.bra, self.ket)
        return {
            "id": self.id,
            "order": self.order,
            "convention": COLOR_CONNECTION_CONVENTION,
            "color_accuracy": "full",
            "storage": "sparse-directed",
            "includes_color_factor": True,
            "sector_ids": list(self.sector_ids),
            "output_legs": [
                {"label": leg.label, "representation": leg.representation}
                for leg in self.output_legs
            ],
            "bra": request.to_json_dict()["bra"],
            "ket": request.to_json_dict()["ket"],
            "entries": [entry.to_json_dict() for entry in self.entries],
        }


def _selected_sectors(
    plan: GenericColorPlan, sector_ids: Sequence[int] | None
) -> tuple[LCColorSector, ...]:
    if not isinstance(plan, GenericColorPlan):
        raise TypeError("colour correlator planning requires a GenericColorPlan")
    if plan.color_accuracy != "full" or plan.process.color_accuracy != "full":
        raise ValueError("colour correlators require a full-colour plan")
    if plan.truncated or plan.trace_reflections_folded or not plan.sectors:
        raise ValueError("colour correlators require a complete unfolded nonempty plan")
    by_id = {sector.id: sector for sector in plan.sectors}
    if len(by_id) != len(plan.sectors) or any(
        type(index) is not int for index in by_id
    ):
        raise ValueError("colour sector IDs must be unique integers")
    requested = tuple(by_id) if sector_ids is None else tuple(sector_ids)
    if not requested or any(type(index) is not int for index in requested):
        raise ValueError("selected colour sector IDs must be nonempty integers")
    if len(set(requested)) != len(requested) or any(
        index not in by_id for index in requested
    ):
        raise ValueError(
            "selected colour sector IDs must be unique and belong to the plan"
        )
    return tuple(by_id[index] for index in requested)


def build_color_correlator_matrices(
    color_plan: GenericColorPlan,
    correlators: Sequence[ColorCorrelator],
    *,
    sector_ids: Sequence[int] | None = None,
) -> tuple[ColorCorrelatorMatrix, ...]:
    """Build exact directed matrices, sharing graph images within this request.

    ``sector_ids`` may select canonical owners from a complete generated plan;
    it does not reinterpret a truncated colour plan as a complete amplitude.
    An empty request is a no-op and leaves the ordinary colour path untouched.
    """

    requests = tuple(correlators)
    if not requests:
        return ()
    if any(not isinstance(request, ColorCorrelator) for request in requests):
        raise TypeError("colour correlators must be ColorCorrelator requests")
    if len({request.id for request in requests}) != len(requests):
        raise ValueError("colour correlator IDs must be unique")
    sectors = _selected_sectors(color_plan, sector_ids)
    ids = tuple(sector.id for sector in sectors)
    legs = process_color_legs(color_plan.process)
    terms = tuple(_sector_color_tensor(sector, legs) for sector in sectors)
    images: dict[tuple[ColorEmission, ...], tuple[ConnectedColorTensor, ...]] = {}
    connections: dict[tuple[ColorEmission, ...], ColorConnection] = {}
    for request in requests:
        for emissions in (request.bra, request.ket):
            if emissions not in connections:
                connections[emissions] = ColorConnection(legs, emissions)
        if connections[request.bra].output_legs != connections[request.ket].output_legs:
            raise ValueError(
                f"colour correlator {request.id!r} has incompatible "
                "typed bra/ket outputs"
            )
    result = []
    for request in requests:
        for emissions in (request.bra, request.ket):
            if emissions not in images:
                images[emissions] = tuple(
                    apply_color_connection(
                        term.tensor,
                        connections[emissions],
                        coefficient=term.coefficient,
                    )
                    for term in terms
                )
        entries = []
        for left_id, left in zip(ids, images[request.bra], strict=True):
            for right_id, right in zip(ids, images[request.ket], strict=True):
                weight = contract_connected_tensors(left, right)
                if weight:
                    entries.append(
                        ColorCorrelatorMatrixEntry(left_id, right_id, weight)
                    )
        result.append(
            ColorCorrelatorMatrix(
                request.id,
                request.order,
                ids,
                connections[request.bra].output_legs,
                tuple(entries),
                request.bra,
                request.ket,
            )
        )
    return tuple(result)


def build_color_correlator_matrix(
    color_plan: GenericColorPlan,
    correlator: ColorCorrelator,
    *,
    sector_ids: Sequence[int] | None = None,
) -> ColorCorrelatorMatrix:
    return build_color_correlator_matrices(
        color_plan, (correlator,), sector_ids=sector_ids
    )[0]


__all__ = [
    "ColorCorrelator",
    "ColorCorrelatorMatrix",
    "ColorCorrelatorMatrixEntry",
    "build_color_correlator_matrices",
    "build_color_correlator_matrix",
    "process_color_legs",
    "sector_color_tensor",
]
