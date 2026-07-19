# SPDX-License-Identifier: 0BSD
"""Correctness-oriented exact execution of a validated eager DAG plan."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import cast

from pyamplicol.api.errors import ArtifactError
from pyamplicol.generation.eager_tables import MISSING_U32
from pyamplicol.models.prepared import PreparedKernelRecord
from pyamplicol.runtime.eager_exact._contracts import (
    _ZERO,
    _complex_add,
    _complex_mul,
    _complex_pair,
    _complex_zero,
    _direct_coefficients,
    _mapping,
    _sequence,
)
from pyamplicol.runtime.eager_exact._plan import _EagerExactPlan
from pyamplicol.runtime.symbolica_exact import (
    _ComplexDecimal,
    _decimal,
    _fill_momenta,
    _fill_sources,
)


def _evaluate_point(
    plan: _EagerExactPlan,
    point: tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...],
    model_parameters: tuple[Decimal, ...],
    precision: int,
) -> tuple[_ComplexDecimal, ...]:
    if len(model_parameters) != plan.parameter_count:
        raise ArtifactError(
            f"eager runtime has {len(model_parameters)} model parameters, "
            f"expected {plan.parameter_count}"
        )
    flattened = [
        _complex_zero()
        for _ in range(plan.value_component_count + plan.momentum_component_count)
    ]
    _fill_sources(flattened, point, plan.runtime_schema, model_parameters)
    _fill_momenta(flattened, point, plan.runtime_schema)
    values = flattened[: plan.value_component_count]
    momenta = flattened[plan.value_component_count :]
    currents = [_complex_zero() for _ in range(plan.current_component_count)]
    couplings = _resolve_couplings(plan, model_parameters)
    prepared_parameters = plan.project_model_parameters(model_parameters)

    for stage in plan.stages:
        for invocation in stage.invocations:
            kernel = plan.kernels[invocation.kernel_id]
            inputs = _gather_inputs(
                kernel.record,
                first_current=plan.value_slots[invocation.left_value_slot_id].read(
                    values
                ),
                second_current=plan.value_slots[invocation.right_value_slot_id].read(
                    values
                ),
                first_momentum=plan.momentum_slots[
                    invocation.left_momentum_slot_id
                ].read(momenta),
                second_momentum=plan.momentum_slots[
                    invocation.right_momentum_slot_id
                ].read(momenta),
                coupling=couplings[invocation.coupling_slot_id],
                prepared_parameters=prepared_parameters,
            )
            outputs = kernel.evaluate(inputs, precision)
            start = invocation.attachment_start
            stop = start + invocation.attachment_count
            for attachment in stage.attachments[start:stop]:
                target = plan.current_slots[attachment.result_current_id]
                factor = _complex_pair(
                    attachment.factor_real,
                    attachment.factor_imag,
                    "eager attachment factor",
                )
                accumulated = target.read(currents)
                target.write(
                    currents,
                    tuple(
                        _complex_add(current, _complex_mul(factor, output))
                        for current, output in zip(accumulated, outputs, strict=True)
                    ),
                    "eager attachment",
                )

        for finalization in stage.finalizations:
            current_slot = plan.current_slots[finalization.current_id]
            current = current_slot.read(currents)
            if finalization.unpropagated_value_slot_id != MISSING_U32:
                plan.value_slots[finalization.unpropagated_value_slot_id].write(
                    values, current, "unpropagated current finalization"
                )
            if finalization.kernel_id == MISSING_U32:
                continue
            kernel = plan.kernels[finalization.kernel_id]
            inputs = _gather_inputs(
                kernel.record,
                first_current=current,
                second_current=(),
                first_momentum=plan.momentum_slots[finalization.momentum_slot_id].read(
                    momenta
                ),
                second_momentum=(),
                coupling=None,
                prepared_parameters=prepared_parameters,
            )
            outputs = kernel.evaluate(inputs, precision)
            plan.value_slots[finalization.propagated_value_slot_id].write(
                values, outputs, "propagated current finalization"
            )

    amplitudes = [_complex_zero() for _ in range(plan.amplitude_count)]
    roots = _sequence(
        _mapping(plan.runtime_schema.get("amplitude_stage"), "amplitude_stage").get(
            "roots"
        ),
        "amplitude roots",
    )
    for index, (closure, raw_root) in enumerate(zip(plan.closures, roots, strict=True)):
        left = plan.value_slots[closure.left_value_slot_id].read(values)
        right = plan.value_slots[closure.right_value_slot_id].read(values)
        if closure.kernel_id == MISSING_U32:
            coefficients = _direct_coefficients(
                _mapping(raw_root, f"amplitude root {index}"), index
            )
            output = _complex_zero()
            for coefficient, left_value, right_value in zip(
                coefficients, left, right, strict=True
            ):
                output = _complex_add(
                    output,
                    _complex_mul(coefficient, _complex_mul(left_value, right_value)),
                )
        else:
            kernel = plan.kernels[closure.kernel_id]
            inputs = _gather_inputs(
                kernel.record,
                first_current=left,
                second_current=right,
                first_momentum=(),
                second_momentum=(),
                coupling=couplings[closure.coupling_slot_id],
                prepared_parameters=prepared_parameters,
            )
            output = kernel.evaluate(inputs, precision)[0]
        factor = _complex_pair(
            closure.factor_real,
            closure.factor_imag,
            "eager closure factor",
        )
        amplitude_id = closure.amplitude_index
        amplitudes[amplitude_id] = _complex_add(
            amplitudes[amplitude_id], _complex_mul(factor, output)
        )
    return tuple(amplitudes)


def _resolve_couplings(
    plan: _EagerExactPlan,
    parameters: Sequence[Decimal],
) -> tuple[_ComplexDecimal, ...]:
    result = []
    for index, row in enumerate(plan.couplings):
        real = (
            _decimal(row.constant_real, f"coupling {index} real constant")
            if row.real_parameter_id == MISSING_U32
            else parameters[row.real_parameter_id]
        )
        imaginary = (
            _decimal(row.constant_imag, f"coupling {index} imaginary constant")
            if row.imag_parameter_id == MISSING_U32
            else parameters[row.imag_parameter_id]
        )
        result.append((real, imaginary))
    return tuple(result)


def _gather_inputs(
    record: PreparedKernelRecord,
    *,
    first_current: Sequence[_ComplexDecimal],
    second_current: Sequence[_ComplexDecimal],
    first_momentum: Sequence[_ComplexDecimal],
    second_momentum: Sequence[_ComplexDecimal],
    coupling: _ComplexDecimal | None,
    prepared_parameters: Sequence[_ComplexDecimal],
) -> tuple[_ComplexDecimal, ...]:
    inputs = []
    for contract in record.input_contracts:
        role = str(contract["role"])
        component = int(cast(int, contract["component"]))
        if role == "left-current":
            value = first_current[component]
        elif role == "right-current":
            value = second_current[component]
        elif role == "current":
            value = first_current[component]
        elif role == "left-momentum":
            value = first_momentum[component]
        elif role == "right-momentum":
            value = second_momentum[component]
        elif role == "momentum":
            value = first_momentum[component]
        elif role == "coupling-real":
            if coupling is None:
                raise ArtifactError("eager kernel requires an unavailable coupling")
            value = (coupling[0], _ZERO)
        elif role == "coupling-imag":
            if coupling is None:
                raise ArtifactError("eager kernel requires an unavailable coupling")
            value = (coupling[1], _ZERO)
        else:
            parameter_id = cast(int, contract["model_parameter_index"])
            value = prepared_parameters[parameter_id]
        inputs.append(value)
    return tuple(inputs)
