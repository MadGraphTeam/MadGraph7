# SPDX-License-Identifier: 0BSD
"""Internal complex directed-colour bare interference on shared native currents.

These diagnostics contain N4 only. They do not complete UV, R2, external-field
or spin-correlated virtual observables and are not a public process dispatcher.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyamplicol.color.connections import ColorConnection, ColorLeg

from .kernels import require_finite_batch
from .orders import canonical_order

# Empty colour histories use the already projected common-degree model metrics.
PROJECTED_IDENTITY_CONVENTION = "sun-graded-identity-all-outgoing-v1"


def directed_color_convention(bare, histories):
    """Admit exact ordinary metrics or an actual projected identity observable."""
    from pyamplicol.color.connections import COLOR_CONNECTION_CONVENTION

    from .physical_color import color_projection_metadata

    projection = color_projection_metadata(bare)
    if projection is not None:
        if not histories or any(
            connection.emissions for pair in histories for connection in pair
        ):
            raise NotImplementedError(
                "projected colour correlations currently require identity histories; "
                "ordered LC/NLC operator banks remain unsupported"
            )
        return PROJECTED_IDENTITY_CONVENTION
    if getattr(bare, "_color_metrics_revision", 0):
        raise NotImplementedError("directed diagnostics require unmodified full colour")
    return COLOR_CONNECTION_CONVENTION


def _physical_color_ports(bare):
    from symbolica.community import tensor as t

    representations = (
        (t.Representation.cof(3), 3),
        (t.Representation.cof(3).dual(), -3),
        (t.Representation.coad(8), 8),
    )
    legs = {index: 1 for index in range(len(bare.state_kinds))}
    groups = []
    for tensors in (bare.born_color_tensors, bare.loop_color_tensors):
        renamed = []
        for expression in tensors:
            mapping = {}
            for axis in expression.axes:
                match = re.fullmatch(
                    r"physical_color_leg_(\d+)",
                    axis.index.get_name().rsplit("::", 1)[-1],
                )
                if match is None or int(match[1]) not in legs:
                    raise ValueError("unknown native physical colour port")
                label = int(match[1])
                representation = next(
                    (
                        value
                        for native, value in representations
                        if axis.representation == native
                    ),
                    None,
                )
                if representation is None or legs[label] not in (1, representation):
                    raise ValueError("native physical colour representations disagree")
                legs[label] = representation
                mapping[axis.index] = label
            renamed.append(expression.rename_indices(mapping))
        groups.append(tuple(renamed))
    return tuple(ColorLeg(label, value) for label, value in legs.items()), *groups


def directed_color_legs(bare):
    """Actual native colour ports, labelled by zero-based physical momentum leg.

    Their signed representations come from the native tensor interface, including
    its crossing convention, rather than from physical particle names alone.
    """
    return _physical_color_ports(bare)[0]


def _rename_physical_color_tensors(tensors):
    """Use the same zero-based physical ports on extra CT colour tensors."""
    result = []
    for expression in tensors:
        mapping = {}
        for axis in expression.axes:
            match = re.fullmatch(
                r"physical_color_leg_(\d+)",
                axis.index.get_name().rsplit("::", 1)[-1],
            )
            if match is None:
                raise ValueError("unknown native physical colour port")
            mapping[axis.index] = int(match[1])
        result.append(expression.rename_indices(mapping))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class NativeDirectedBareEvaluation:
    batch: Any
    born: Any
    virtual: Any
    first: Any
    second: Any
    statuses: Any


@dataclass(frozen=True, slots=True)
class DirectedBareResult:
    """Decimal complex pairs on point, history, order, optional pole axes.

    Status entries contain original physical point IDs. Their parallel channel
    catalogue supplies direction, order and family; IDs never encode those axes.
    """

    born: Any
    virtual: Any
    first: Any
    second: Any
    orders: Any
    histories: tuple[tuple[ColorConnection, ColorConnection], ...]
    channels: tuple[tuple[int, int, int], ...]
    statuses: Any
    arithmetic: str
    precision: int | None
    epsilon_powers: tuple[int, ...] = (0, -1, -2)
    convention: str = "MadLoop-Cepsilon-directed-bare-N4-no-UV-no-R2"
    spin_legs: tuple[int, ...] = ()

    def __post_init__(self):
        for values in (self.born, self.virtual, self.first, self.second):
            values.setflags(write=False)

    def born_order(self, order):
        wanted = canonical_order(order)
        available = tuple(sector.order for sector in self.orders.born)
        if wanted not in available:
            raise ValueError(f"Born squared order {wanted!r} is absent")
        return self.born[:, :, available.index(wanted)]

    def virtual_order(self, order):
        wanted = canonical_order(order)
        available = tuple(sector.order for sector in self.orders.virtual)
        if wanted not in available:
            raise ValueError(f"virtual squared order {wanted!r} is absent")
        return self.virtual[:, :, available.index(wanted)]


class NativeDirectedBareProgram:
    """Reduce both ordered complex cross terms with one shared current batch.

    Directions 2*h and 2*h+1 represent a history pair and its swap. Both use
    the same Feynman prescription. Only after reduction is the latter amplitude,
    including i/(16*pi**2), conjugated. Born contractions retain their original
    ordered history and are never coerced to a real number.
    """

    def __init__(self, bare, histories, *, spin_layout=None, _prepared=None):
        histories = tuple(tuple(pair) for pair in histories)
        self.bare = bare
        self.histories = histories
        self.spin_layout = spin_layout
        self.helicities = (
            bare.helicities if spin_layout is None else spin_layout.helicities
        )
        self._helicity_options = (
            {} if spin_layout is None else {"helicities": self.helicities}
        )
        self._check_scope()
        self._metrics = (bare.born_metric, bare.loop_metric)
        from .prepared_correlations import PreparedDirectedBare, prepare_directed_bare

        prepared = _prepared or prepare_directed_bare(
            bare, histories, spin_layout=spin_layout
        )
        if not isinstance(prepared, PreparedDirectedBare):
            raise TypeError("expected immutable directed bare preparation")
        prepared.require_owner(bare)
        if (
            prepared.histories != tuple(tuple(pair) for pair in histories)
            or prepared.spin_layout != spin_layout
        ):
            raise ValueError("directed history/source layout differs from preparation")
        self.prepared = prepared
        self.histories, self.color_legs = prepared.histories, prepared.color_legs
        self.born_projector = bare.backend.CoefficientEvaluator(
            prepared.born_state, **bare.options
        )
        self.projectors = tuple(family.bind(bare) for family in prepared.families)
        self.family_weight_columns = tuple(
            family.weight_columns for family in prepared.families
        )
        self.dual_born_weights = prepared.weight_plan.bind(bare)
        self.directions, self.channels = 2 * len(self.histories), prepared.channels
        self.postprocess = bare.backend.CoefficientEvaluator(
            prepared.postprocess_state, **bare.options
        )
        # A model wrapper may not later reinterpret these weights as complete CT
        # metrics; ordinary colour mutation would also invalidate the attachment.
        bare._directed_color_metrics_bound = True

    def _check_scope(self):
        if getattr(self.bare, "_model_color_metrics_bound", False):
            raise ValueError(
                "directed bare diagnostics cannot use a model-bound program"
            )
        self._check_metric_scope()

    def _check_metric_scope(self):
        bare = self.bare
        if bare.dimensional_numerators:
            raise NotImplementedError("directed diagnostics require N4 without R2")
        directed_color_convention(bare, self.histories)
        if (
            hasattr(self, "_metrics")
            and (bare.born_metric, bare.loop_metric) != self._metrics
        ):
            raise ValueError("bare colour metrics changed after directed preparation")

    def _prepare_currents_native(
        self,
        momenta,
        scales_squared,
        *,
        model_values=None,
        spin_sources=None,
        point_ids=None,
    ):
        options = self._helicity_options
        expected = () if self.spin_layout is None else self.spin_layout.legs
        if tuple(sorted(spin_sources or {})) != expected:
            raise ValueError("spin sources differ from the compiled joint layout")
        if spin_sources:
            options = {**options, "spin_sources": spin_sources}
        return self.bare.prepare_currents_native(
            momenta,
            scales_squared,
            model_values=model_values,
            point_ids=point_ids,
            **options,
        )

    def evaluate_native(
        self,
        momenta,
        scales_squared,
        *,
        model_values=None,
        spin_sources=None,
        point_ids=None,
    ):
        self._check_scope()
        currents = self._prepare_currents_native(
            momenta,
            scales_squared,
            model_values=model_values,
            spin_sources=spin_sources,
            point_ids=point_ids,
        )
        return self.evaluate_currents_native(currents)

    def evaluate_currents_native(self, currents):
        """Contract already prepared currents without repeating state work."""
        self._check_scope()
        bare = self.bare
        if (
            currents.batch.helicity_count != len(self.helicities)
            or currents.batch.rows != currents.batch.size * len(self.helicities)
            or currents.batch.spin_legs
            != (() if self.spin_layout is None else self.spin_layout.legs)
        ):
            raise ValueError("physical currents use a different helicity/source layout")
        batch = bare.backend.NativeBatch
        size, point_ids = currents.batch.size, currents.batch.point_ids
        order_count = len(bare.order_layout.virtual)
        born = self.born_projector.evaluate_batch(
            currents.born.reshape_rows(
                size, len(self.helicities) * currents.born.columns, point_ids
            )
        )
        require_finite_batch(born, "directed Born contraction")
        weights = self.dual_born_weights.evaluate_batch(currents.born)
        require_finite_batch(
            weights,
            "directed dual Born weights",
            row_points=currents.batch.point_by_lane,
        )
        integrated, statuses = [], []
        for family, weight_columns, projector in zip(
            bare.families, self.family_weight_columns, self.projectors, strict=True
        ):
            channels = tuple(
                (direction, sector)
                for direction in range(self.directions)
                for sector in family.sector_indices
            )
            count = len(channels)
            row_ids = list(range(size * count))
            point_rows = np.repeat(np.arange(size), count).tolist()
            coefficients = projector.evaluate_batch(
                bare.family_projection_arguments(
                    currents,
                    family,
                    weights=weights,
                    weight_columns=weight_columns,
                )
            ).reshape_rows(size * count, len(family.powers), row_ids)
            shifts, masses, matrix = bare.family_kinematics_native(currents, family)
            reduced = family.reducer.evaluate_batch(
                shifts.take_rows(point_rows, row_ids=row_ids),
                masses.take_rows(
                    point_rows if masses.rows == size else [0] * (size * count),
                    row_ids=row_ids,
                ),
                currents.batch.scales_squared.take_rows(point_rows, row_ids=row_ids),
                coefficients,
                s_matrix=matrix.take_rows(point_rows, row_ids=row_ids),
            )
            failures = [
                (
                    point_ids[row // count],
                    *channels[row % count],
                    family.index,
                    status,
                )
                for row, status in reduced.statuses
                if status != "success"
            ]
            if failures:
                raise ArithmeticError(
                    "directed reduction failed "
                    "(point, direction, order, family, status): "
                    f"{failures}"
                )
            integrated.append(reduced.values.reshape_rows(size, count * 3, point_ids))
            by_id = dict(reduced.statuses)
            statuses.extend(
                tuple(
                    (
                        point,
                        by_id[position * count + channel],
                    )
                    for position, point in enumerate(point_ids)
                )
                for channel in range(count)
            )
        result = self.postprocess.evaluate_batch(batch.concat_columns(integrated))
        require_finite_batch(result, "directed bare cross terms")
        width = len(self.histories) * order_count * 3
        return NativeDirectedBareEvaluation(
            currents.batch,
            born,
            result.take_columns(list(range(width))),
            result.take_columns(list(range(width, 2 * width))),
            result.take_columns(list(range(2 * width, 3 * width))),
            tuple(
                tuple(channel[point] for channel in statuses) for point in range(size)
            ),
        )

    def evaluate_decimal(
        self,
        momenta,
        scales_squared,
        *,
        model_values=None,
        spin_sources=None,
        point_ids=None,
    ):
        native = self.evaluate_native(
            momenta,
            scales_squared,
            model_values=model_values,
            spin_sources=spin_sources,
            point_ids=point_ids,
        )
        size, history_count = native.batch.size, len(self.histories)

        def values(batch, *axes):
            return np.asarray(batch.to_decimal(), dtype=object).reshape(
                size, history_count, *axes, 2
            )

        return DirectedBareResult(
            values(native.born, len(self.bare.order_layout.born)),
            *(
                values(value, len(self.bare.order_layout.virtual), 3)
                for value in (native.virtual, native.first, native.second)
            ),
            self.bare.order_layout,
            self.histories,
            self.channels,
            native.statuses,
            self.bare.arithmetic,
            self.bare.precision,
            spin_legs=() if self.spin_layout is None else self.spin_layout.legs,
        )
