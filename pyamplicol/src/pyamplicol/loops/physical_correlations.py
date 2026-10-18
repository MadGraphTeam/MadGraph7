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

from .color_bilinears import SUNColorSpace
from .color_connections import directed_color_bilinears
from .kernels import require_finite_batch
from .orders import canonical_order
from .physical import _saved


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
    """Decimal complex pairs on point, history, order, optional pole axes."""

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

    def __init__(self, bare, histories):
        from symbolica import E, S

        self.bare = bare
        self._check_scope()
        self._metrics = (bare.born_metric, bare.loop_metric)
        histories = tuple(tuple(pair) for pair in histories)
        if not histories or any(
            len(pair) != 2 or any(not isinstance(x, ColorConnection) for x in pair)
            for pair in histories
        ):
            raise ValueError("expected nonempty ordered colour-history pairs")
        self.histories = histories
        self.color_legs, born, loop = _physical_color_ports(bare)
        if any(
            connection.initial_legs != self.color_legs
            for pair in histories
            for connection in pair
        ):
            raise ValueError(
                "history legs differ from the native physical colour ports"
            )
        nc = S("loop_directed_physical::Nc", is_real=True)
        space = SUNColorSpace(nc)

        def fixed(matrix):
            return tuple(
                tuple(
                    (value.replace(nc, 3) / bare.initial_average).together()
                    for value in row
                )
                for row in matrix
            )

        born_metrics, loop_metrics = [], []
        for bra, ket in histories:
            kernels = directed_color_bilinears(space, born, loop, bra, ket)
            born_metrics.append(
                fixed(directed_color_bilinears(space, born, born, bra, ket).born_loop)
            )
            loop_metrics.extend(
                (
                    fixed(kernels.born_loop),
                    fixed(
                        tuple(
                            tuple(
                                kernels.loop_born[j][i].conj() for j in range(len(loop))
                            )
                            for i in range(len(born))
                        )
                    ),
                )
            )
        self.born_projector = bare._make_born_projector(born_metrics)
        self.projectors = tuple(
            bare._compile_family_projection(
                family.graph_indices,
                family.powers,
                family.sector_indices,
                metrics=loop_metrics,
            )
            for family in bare.families
        )
        self.directions = 2 * len(histories)
        self.channels = tuple(
            (direction, sector, family.index)
            for family in bare.families
            for direction in range(self.directions)
            for sector in family.sector_indices
        )
        poles = {
            channel: tuple(
                S(f"loop_directed_physical::pole_{index}_{pole}") for pole in range(3)
            )
            for index, channel in enumerate(self.channels)
        }
        first, second = [], []
        for history in range(len(histories)):
            for sector in range(len(bare.order_layout.virtual)):
                for pole in range(3):

                    def amplitude(direction, sector=sector, pole=pole):
                        return (
                            E("1i")
                            * sum(
                                (
                                    values[pole]
                                    for (d, s, _), values in poles.items()
                                    if d == direction and s == sector
                                ),
                                E("0"),
                            )
                            / (16 * E("pi") ** 2)
                        )

                    first.append(amplitude(2 * history))
                    second.append(amplitude(2 * history + 1).conj())
        self.postprocess = bare.backend.CoefficientEvaluator(
            _saved(
                [a + b for a, b in zip(first, second, strict=True)] + first + second,
                [value for values in poles.values() for value in values],
            ),
            **bare.options,
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
        if getattr(bare, "color_accuracy", "full") != "full" or getattr(
            bare, "_color_metrics_revision", 0
        ):
            raise NotImplementedError(
                "directed diagnostics require unmodified full colour"
            )
        if (
            hasattr(self, "_metrics")
            and (bare.born_metric, bare.loop_metric) != self._metrics
        ):
            raise ValueError("bare colour metrics changed after directed preparation")

    def evaluate_native(self, momenta, scales_squared, *, model_values):
        self._check_scope()
        bare = self.bare
        currents = bare.prepare_currents_native(
            momenta, scales_squared, model_values=model_values
        )
        return self.evaluate_currents_native(currents)

    def evaluate_currents_native(self, currents):
        """Contract already prepared currents without repeating state work."""
        self._check_scope()
        bare = self.bare
        batch = bare.backend.NativeBatch
        size, point_ids = currents.batch.size, currents.batch.point_ids
        order_count = len(bare.order_layout.virtual)
        born = self.born_projector.evaluate_batch(
            currents.born.reshape_rows(
                size, len(bare.helicities) * currents.born.columns, point_ids
            )
        )
        require_finite_batch(born, "directed Born contraction")
        integrated, statuses = [], []
        for family, projector in zip(bare.families, self.projectors, strict=True):
            channels = tuple(
                (direction, sector)
                for direction in range(self.directions)
                for sector in family.sector_indices
            )
            count = len(channels)
            row_ids = [
                point * self.directions * order_count + direction * order_count + sector
                for point in range(size)
                for direction, sector in channels
            ]
            point_rows = np.repeat(np.arange(size), count).tolist()
            coefficients = projector.evaluate_batch(
                bare.family_projection_arguments(currents, family)
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
                    row // (self.directions * order_count),
                    (row // order_count) % self.directions,
                    row % order_count,
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
                        point * self.directions * order_count
                        + direction * order_count
                        + sector,
                        by_id[
                            point * self.directions * order_count
                            + direction * order_count
                            + sector
                        ],
                    )
                    for point in range(size)
                )
                for direction, sector in channels
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

    def evaluate_decimal(self, momenta, scales_squared, *, model_values):
        native = self.evaluate_native(
            momenta, scales_squared, model_values=model_values
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
        )
