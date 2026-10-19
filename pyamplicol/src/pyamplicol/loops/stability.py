# SPDX-License-Identifier: 0BSD
"""Optional empirical cross-precision verification of complete native controls.

Agreement between evaluations is evidence under explicit componentwise bounds,
not an independent certification of NinjaRS or the physical model. Ordinary
physical evaluators do not enable this additional work implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .numerical_lanes import require_partial_domain


@dataclass(frozen=True, slots=True)
class ArithmeticSpec:
    arithmetic: str
    precision: int | None = None

    @property
    def options(self):
        return {"arithmetic": self.arithmetic, "precision": self.precision}


def _validate_precision_sequence(
    backend, specs, relative_tolerance, absolute_tolerance
):
    previous_bits = 0
    for spec in specs:
        empty = backend.NativeBatch.from_decimal(
            [], 0, 1, arithmetic=spec.arithmetic, precision=spec.precision
        )
        if empty.precision_bits <= previous_bits:
            raise ValueError("verification arithmetic precision must strictly increase")
        previous_bits = empty.precision_bits
        empty.agreement_rows(empty, relative_tolerance, absolute_tolerance)


def validate_source_stability(backend, arithmetic, precision, policy):
    """Validate the existing public policy before a producer source is bound."""
    if policy is None:
        return
    from pyamplicol.api.virtual_results import VirtualStabilityPolicy

    if not isinstance(policy, VirtualStabilityPolicy):
        raise TypeError("source stability requires a VirtualStabilityPolicy")
    _validate_precision_sequence(
        backend,
        (ArithmeticSpec(arithmetic, precision), *policy.higher),
        policy.relative_tolerance,
        policy.absolute_tolerance,
    )


@dataclass(frozen=True, slots=True)
class NativeStabilityDiagnostic:
    point_id: int
    status: str
    accuracy: str
    output_arithmetic: ArithmeticSpec
    evaluated_arithmetic: ArithmeticSpec | None
    verification_pair: tuple[ArithmeticSpec, ArithmeticSpec] | None
    attempts: tuple[ArithmeticSpec, ...]
    failures: tuple[Any, ...]
    reduction_statuses: tuple[Any, ...]
    detail: str = ""


@dataclass(frozen=True, slots=True)
class NativeStableModelResult:
    requested_point_ids: tuple[int, ...]
    point_ids: tuple[int, ...]
    born: Any
    virtual: Any
    bare: Any
    uv: Any
    r2: Any
    uv_local: Any
    uv_external_fields: Any
    totals: Any
    orders: Any
    channels: Any
    r2_strategy: str
    diagnostics: tuple[NativeStabilityDiagnostic, ...]

    @property
    def component_batches(self):
        return (
            self.virtual,
            self.bare,
            self.uv,
            self.r2,
            self.uv_local,
            self.uv_external_fields,
        )


class NumericalVerificationError(ArithmeticError):
    def __init__(self, result):
        self.result = result
        ids = [item.point_id for item in result.diagnostics if item.status == "failed"]
        super().__init__(f"native virtual evaluation has unrecovered points: {ids}")


def _select(values, ids):
    positions = {point: row for row, point in enumerate(values.row_ids)}
    return values.take_rows([positions[point] for point in ids])


class NativeModelStabilityProgram:
    """Explicit verification/rescue view over one fixed R2 physical plan.

    Higher views are compiled lazily from the same exact cold process/model data.
    Each call copies its original decimal inputs once. No rounded intermediate is
    used to reconstruct a higher-precision physical evaluation. Source owners
    retain one exact snapshot across attempts. The certified direct-R2 gg control
    preserves its strategy and dimensional sectors. Projected-colour, CMS and
    correlated views remain outside this boundary.
    Malformed original inputs and metadata errors still propagate. Numerical
    source-realization and external-state failures retain original point IDs.
    """

    def __init__(
        self, program, higher_arithmetics, *, relative_tolerance, absolute_tolerance
    ):
        require_partial_domain(program.bare)
        if (
            program.r2_strategy not in {"model", "direct"}
            or program.color_projection is not None
        ):
            raise NotImplementedError(
                "verification requires a certified full-colour R2 strategy"
            )
        self.r2_strategy = program.r2_strategy
        self.dimensional_numerators = program.bare.dimensional_numerators
        self.program = program
        self.backend = program.bare.backend
        self.requested = ArithmeticSpec(program.bare.arithmetic, program.bare.precision)
        self.higher_arithmetics = tuple(higher_arithmetics)
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance
        self._views = {self.requested: program}
        for spec in (self.requested, *self.higher_arithmetics):
            if not isinstance(spec, ArithmeticSpec):
                raise TypeError("verification arithmetic must be an ArithmeticSpec")
        _validate_precision_sequence(
            self.backend,
            (self.requested, *self.higher_arithmetics),
            relative_tolerance,
            absolute_tolerance,
        )

    def _view(self, spec, source_snapshot=None):
        if spec not in self._views and self.program.bare.model_state is None:
            self._views[spec] = self.program.rebind(
                **spec.options,
                reduction_engine="generic",
                coefficient_engine="interpreted",
            )
        if spec not in self._views:
            original = self.program.bare.model_state
            original.require_snapshot(source_snapshot)
            # Bind the exact retained source record, including its guards, rather
            # than reconstructing it from optional producer inspection metadata.
            self._views[spec] = self.program.prepared.bind(
                backend=self.backend,
                parameters=source_snapshot.overrides,
                reduction_engine="generic",
                coefficient_engine="interpreted",
                **spec.options,
            )
        return self._views[spec]

    def evaluate_native(
        self,
        momenta,
        scales_squared,
        *,
        model_values=None,
        verify=False,
        allow_partial=False,
        point_ids=None,
    ):
        """Optionally verify all components and return only accepted native rows.

        A failed or disagreeing primary needs two successful higher evaluations
        to agree before rescue. The final requested-carrier result must itself
        satisfy the same bound against the verified higher result.
        """
        if verify and len(self.higher_arithmetics) < 2:
            raise ValueError(
                "verification/rescue requires at least two higher arithmetics"
            )
        require_partial_domain(self.program.bare)
        if (
            self.program.r2_strategy != self.r2_strategy
            or self.program.bare.dimensional_numerators != self.dimensional_numerators
            or self.program.color_projection is not None
        ):
            raise ValueError("verification plan strategy or colour scope changed")
        source_state = self.program.bare.model_state
        if source_state is not None:
            if model_values is not None:
                raise ValueError(
                    "update the source model state through its named inputs"
                )
            source_snapshot = source_state.require_snapshot(source_state.snapshot())
        else:
            source_snapshot = None
            model_values = tuple(model_values) if model_values is not None else None
        momenta = np.array(momenta, dtype=object, copy=True)
        scales = np.array(scales_squared, dtype=object, copy=True)
        from .physical_process import _physical_point_ids

        ids = tuple(
            _physical_point_ids(
                momenta, scales, len(self.program.bare.state_kinds), point_ids
            )
        )
        positions = {point: row for row, point in enumerate(ids)}
        attempts = {point: [] for point in ids}
        failures = {point: [] for point in ids}
        reduction_statuses = {point: [] for point in ids}
        selected_specs, pairs, details = {}, {}, {}
        accepted = []
        primary = None

        def attempt(spec, selected):
            from .numerical_lanes import NumericalLaneFailure
            from .scalar_bindings import SourceBindingFailure

            rows = [positions[point] for point in selected]
            try:
                view = self._view(spec, source_snapshot)
                snapshot = source_snapshot
                if source_state is not None and spec != self.requested:
                    snapshot = view.bare.model_state.reconstruct_snapshot(
                        source_state, source_snapshot
                    )
            except SourceBindingFailure as error:
                if primary is None:
                    raise
                for point in selected:
                    attempts[point].append(spec)
                    failures[point].append(
                        (spec, NumericalLaneFailure(point, error.stage, error.detail))
                    )
                return primary.take_rows([]).convert_arithmetic(**spec.options)
            result = view.evaluate_ordered_partial_native(
                momenta[rows],
                scales[rows],
                model_values=model_values,
                point_ids=selected,
                model_snapshot=snapshot,
            )
            if result.r2_strategy != self.r2_strategy:
                raise ValueError("native attempt changed the captured R2 strategy")
            for point in selected:
                attempts[point].append(spec)
            for point, statuses in zip(
                result.batch.point_ids, result.statuses, strict=True
            ):
                reduction_statuses[point].append((spec, statuses))
            for failure in result.failures:
                failures[failure.point_id].append((spec, failure))
            return self.backend.NativeBatch.concat_columns(
                [result.born, *result.component_batches, result.totals]
            )

        def agree(left, right):
            return [
                point
                for point, good in left.agreement_rows(
                    right, self.relative_tolerance, self.absolute_tolerance
                )
                if good
            ]

        primary = attempt(self.requested, ids)
        if not verify:
            accepted.append(primary)
            selected_specs.update((point, self.requested) for point in primary.row_ids)
        else:
            pending = list(ids)
            previous_groups = []
            for level, spec in enumerate(self.higher_arithmetics):
                if not pending:
                    break
                current = attempt(spec, pending)
                current_ids = set(current.row_ids)
                if level == 0:
                    common = [
                        point
                        for point in pending
                        if point in current_ids and point in primary.row_ids
                    ]
                    matching = agree(_select(primary, common), _select(current, common))
                    if matching:
                        accepted.append(_select(primary, matching))
                        for point in matching:
                            selected_specs[point] = self.requested
                            pairs[point] = (self.requested, spec)
                else:
                    for previous_spec, previous_high in previous_groups:
                        common = [
                            point
                            for point in pending
                            if point in current_ids and point in previous_high.row_ids
                        ]
                        matching = agree(
                            _select(previous_high, common), _select(current, common)
                        )
                        high = _select(current, matching)
                        converted = high.convert_arithmetic(**self.requested.options)
                        output_matches = agree(converted, high)
                        if output_matches:
                            accepted.append(_select(converted, output_matches))
                            for point in output_matches:
                                selected_specs[point] = spec
                                pairs[point] = (previous_spec, spec)
                                details.pop(point, None)
                        for point in set(matching).difference(output_matches):
                            details[point] = (
                                "verified higher results exceed the requested-carrier "
                                "output tolerance"
                            )
                pending = [point for point in pending if point not in selected_specs]
                available = [point for point in pending if point in current_ids]
                # Retain each previous successful carrier without conversion:
                # sparse DoubleFloat words may extend beyond its nominal bits.
                retained_groups = []
                for previous_spec, previous_high in previous_groups:
                    retained = [
                        point
                        for point in pending
                        if point not in current_ids and point in previous_high.row_ids
                    ]
                    if retained:
                        retained_groups.append(
                            (previous_spec, _select(previous_high, retained))
                        )
                previous_groups = [
                    (spec, _select(current, available)),
                    *retained_groups,
                ]
        packet = self.backend.NativeBatch.merge_rows(
            ids, accepted or [primary.take_rows([])]
        )
        present = set(packet.row_ids)
        diagnostics = tuple(
            NativeStabilityDiagnostic(
                point,
                "failed"
                if point not in present
                else (
                    "success" if selected_specs[point] == self.requested else "rescued"
                ),
                "empirical-cross-precision-agreement"
                if point in present and verify
                else "not-established",
                self.requested,
                selected_specs.get(point),
                pairs.get(point),
                tuple(attempts[point]),
                tuple(failures[point]),
                tuple(reduction_statuses[point]),
                details.get(
                    point,
                    ""
                    if point in present
                    else "no accepted finite result within the configured "
                    "verification policy",
                ),
            )
            for point in ids
        )
        born_count = len(self.program.bare.order_layout.born)
        width = 3 * len(self.program.bare.order_layout.virtual)
        result = NativeStableModelResult(
            ids,
            tuple(packet.row_ids),
            packet.take_columns(range(born_count)),
            *[
                packet.take_columns(
                    range(born_count + i * width, born_count + (i + 1) * width)
                )
                for i in range(6)
            ],
            packet.take_columns(range(born_count + 6 * width, packet.columns)),
            self.program.bare.order_layout,
            self.program.bare.order_channels,
            self.r2_strategy,
            diagnostics,
        )
        if not allow_partial and len(result.point_ids) != len(ids):
            raise NumericalVerificationError(result)
        return result
