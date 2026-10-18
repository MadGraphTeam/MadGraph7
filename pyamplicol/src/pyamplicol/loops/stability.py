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
    """Explicit verification/rescue view over one fixed model-R2 physical plan.

    Higher views are compiled lazily from the same exact cold process/model data.
    Each call copies its original decimal inputs once. No rounded intermediate is
    used to reconstruct a higher-precision physical evaluation. Source-owned,
    direct-R2, projected-colour and correlated views remain outside this boundary.
    Malformed original inputs and global model formula errors still propagate.
    Numerical external-state failures retain sparse original point identities.
    """

    def __init__(
        self, program, higher_arithmetics, *, relative_tolerance, absolute_tolerance
    ):
        require_partial_domain(program.bare)
        if program.r2_strategy != "model" or program.color_projection is not None:
            raise NotImplementedError("verification requires full-colour model R2")
        self.program = program
        self.backend = program.bare.backend
        self.requested = ArithmeticSpec(program.bare.arithmetic, program.bare.precision)
        self.higher_arithmetics = tuple(higher_arithmetics)
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance
        self._views = {self.requested: program}
        previous_bits = 0
        for spec in (self.requested, *self.higher_arithmetics):
            if not isinstance(spec, ArithmeticSpec):
                raise TypeError("verification arithmetic must be an ArithmeticSpec")
            empty = self.backend.NativeBatch.from_decimal([], 0, 1, **spec.options)
            if empty.precision_bits <= previous_bits:
                raise ValueError(
                    "verification arithmetic precision must strictly increase"
                )
            previous_bits = empty.precision_bits
            # The existing native exact decimal parser validates the bound once.
            empty.agreement_rows(empty, relative_tolerance, absolute_tolerance)

    def _view(self, spec):
        if spec not in self._views:
            from .physical_counterterms import NativeModelVirtualProgram
            from .physical_process import NativeBareProcessProgram

            original = self.program.bare
            bare = NativeBareProcessProgram(
                original.process,
                state_kinds=original.state_kinds,
                initial_average=original.initial_average,
                model_expressions=original.model_expressions,
                model_inputs=original.model_inputs,
                vertex_spinor_frame=original.vertex_spinor_frame,
                backend=original.backend,
                reduction_engine="generic",
                coefficient_engine="interpreted",
                cache_coefficient_kernels=original.cache_coefficient_kernels,
                share_coefficient_schedules=original.share_coefficient_schedules,
                **spec.options,
            )
            self._views[spec] = NativeModelVirtualProgram(
                bare,
                self.program.matched_insertions,
                source_bindings=self.program.source_bindings,
                external_fields=self.program.external_fields,
                r2_strategy="model",
            )
        return self._views[spec]

    def evaluate_native(
        self,
        momenta,
        scales_squared,
        *,
        model_values,
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
            self.program.r2_strategy != "model"
            or self.program.color_projection is not None
        ):
            raise ValueError("verification plan strategy or colour scope changed")
        momenta = np.array(momenta, dtype=object, copy=True)
        scales = np.array(scales_squared, dtype=object, copy=True)
        model_values = tuple(model_values)
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

        def attempt(spec, selected):
            rows = [positions[point] for point in selected]
            result = self._view(spec).evaluate_ordered_partial_native(
                momenta[rows],
                scales[rows],
                model_values=model_values,
                point_ids=selected,
            )
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
            diagnostics,
        )
        if not allow_partial and len(result.point_ids) != len(ids):
            raise NumericalVerificationError(result)
        return result
