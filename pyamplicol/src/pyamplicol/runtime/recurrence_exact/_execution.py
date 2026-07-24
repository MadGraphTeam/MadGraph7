# SPDX-License-Identifier: 0BSD
"""Decimal/Symbolica execution of one authenticated Direct-Arena plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

from pyamplicol.api.errors import ArtifactError, CompatibilityError, EvaluationError
from pyamplicol.runtime.eager_exact._contracts import (
    _complex_add,
    _complex_mul,
    _complex_zero,
)
from pyamplicol.runtime.symbolica_exact import (
    _antiquark_dirac,
    _antiquark_weyl,
    _ComplexDecimal,
    _massive_vector,
    _massless_vector,
    _quark_dirac,
    _quark_weyl,
    _spin2,
)

from ._plan import _RecurrenceExactPlan, _SourceTemplate
from ._plan_v2 import (
    DIRECT_NONE_U32,
    _Closure,
    _Contribution,
    _Executor,
    _Finalization,
    _ReplayTarget,
    _ResolvedHelicity,
    _SourceDispatchVariant,
)

_ZERO = Decimal(0)
_ONE = Decimal(1)
_ROLE_SOURCE = 0
_ROLE_CONTRIBUTION = 1
_ROLE_FINALIZATION = 2
_ROLE_CLOSURE = 3
_NODE_CURRENT = 1

_Point = Sequence[tuple[Decimal, Decimal, Decimal, Decimal]]


def _evaluate_replay_point(
    plan: _RecurrenceExactPlan,
    point: _Point,
    target: _ReplayTarget,
    prepared_parameters: Sequence[_ComplexDecimal],
    precision: int,
) -> tuple[_ComplexDecimal, ...]:
    sections = plan.sections
    permutation = sections.source_permutations[
        target.source_permutation_start : target.source_permutation_start
        + target.source_permutation_count
    ]
    if len(permutation) != sections.external_source_count:
        raise ArtifactError("recurrence replay permutation has invalid width")
    momenta = _momentum_forms(plan, point, permutation)
    amplitudes = list(
        _execute_schedule(
            plan,
            momenta,
            prepared_parameters,
            precision,
            selected_source_variants=None,
        )
    )

    replay_factor = _complex_mul(
        _factor(plan, target.phase_factor_id),
        (Decimal(target.multiplicity), _ZERO),
    )
    for destination in sections.amplitude_destinations:
        if destination.target_sector_id == target.representative_id:
            amplitudes[destination.destination_id] = _complex_mul(
                amplitudes[destination.destination_id],
                replay_factor,
            )
    return tuple(amplitudes)


def _evaluate_union_point(
    plan: _RecurrenceExactPlan,
    point: _Point,
    helicity: _ResolvedHelicity,
    prepared_parameters: Sequence[_ComplexDecimal],
    precision: int,
) -> tuple[_ComplexDecimal, ...]:
    """Execute one all-flow union once for one runtime-selected helicity."""

    sections = plan.sections
    if sections.strategy != "all-flow-union":
        raise ArtifactError("union exact execution requires an all-flow-union plan")
    selections = sections.resolved_source_selections[
        helicity.source_selection_start : helicity.source_selection_start
        + helicity.source_selection_count
    ]
    if len(selections) != sections.external_source_count:
        raise ArtifactError(
            "all-flow-union helicity does not select every external source"
        )
    selected_source_variants = {}
    for source_slot, selection in enumerate(selections):
        if selection.source_slot != source_slot:
            raise ArtifactError(
                "all-flow-union source selections are not in source-slot order"
            )
        try:
            variant = sections.source_dispatch_variants[
                selection.dispatch_variant_id
            ]
        except IndexError as exc:
            raise ArtifactError(
                "all-flow-union source selection references an absent variant"
            ) from exc
        try:
            source = sections.sources[variant.source_row_id]
        except IndexError as exc:
            raise ArtifactError(
                "all-flow-union source variant references an absent source row"
            ) from exc
        if source.source_slot != source_slot:
            raise ArtifactError(
                "all-flow-union source variant selects the wrong external source"
            )
        if variant.source_row_id in selected_source_variants:
            raise ArtifactError(
                "all-flow-union helicity selects one source row more than once"
            )
        selected_source_variants[variant.source_row_id] = variant
    momenta = _momentum_forms(
        plan,
        point,
        tuple(range(sections.external_source_count)),
    )
    return _execute_schedule(
        plan,
        momenta,
        prepared_parameters,
        precision,
        selected_source_variants=selected_source_variants,
    )


def _execute_schedule(
    plan: _RecurrenceExactPlan,
    momenta: Sequence[Sequence[Decimal]],
    prepared_parameters: Sequence[_ComplexDecimal],
    precision: int,
    *,
    selected_source_variants: Mapping[int, _SourceDispatchVariant] | None,
) -> tuple[_ComplexDecimal, ...]:
    sections = plan.sections
    arena = [_complex_zero() for _ in range(sections.current_arena_components)]
    amplitudes = [_complex_zero() for _ in range(sections.amplitude_destination_count)]
    initialized_contribution_stage: int | None = None

    for group in sections.row_groups:
        start = group.row_start
        stop = start + group.row_count
        if group.role == _ROLE_SOURCE:
            if selected_source_variants is not None:
                if group.executor_id != DIRECT_NONE_U32:
                    raise ArtifactError(
                        "all-flow-union source row group must use runtime dispatch"
                    )
            else:
                executor = plan.executors.get(group.executor_id)
                if executor is None or _role_index(executor.role) != group.role:
                    raise ArtifactError(
                        "recurrence source row group references invalid executor "
                        f"{group.executor_id}"
                    )
            for source_row_id in range(start, stop):
                row = sections.sources[source_row_id]
                if selected_source_variants is None:
                    _execute_source(
                        plan,
                        row,
                        momenta,
                        prepared_parameters,
                        arena,
                    )
                    continue
                variant = selected_source_variants.get(source_row_id)
                if variant is not None:
                    _execute_union_source(
                        plan,
                        row,
                        variant,
                        momenta,
                        prepared_parameters,
                        arena,
                    )
            continue

        executor = plan.executors.get(group.executor_id)
        if executor is None or _role_index(executor.role) != group.role:
            raise ArtifactError(
                f"recurrence row group references invalid executor {group.executor_id}"
            )
        if group.role == _ROLE_CONTRIBUTION:
            if initialized_contribution_stage != group.stage:
                _clear_stage(plan, arena, group.stage)
                initialized_contribution_stage = group.stage
            for row in sections.contributions[start:stop]:
                _execute_prepared_row(
                    plan,
                    executor,
                    row,
                    momenta,
                    prepared_parameters,
                    arena,
                    amplitudes,
                    precision,
                )
        elif group.role == _ROLE_FINALIZATION:
            for row in sections.finalizations[start:stop]:
                _execute_finalization(
                    plan,
                    executor,
                    row,
                    momenta,
                    prepared_parameters,
                    arena,
                    amplitudes,
                    precision,
                )
        elif group.role == _ROLE_CLOSURE:
            for row in sections.closures[start:stop]:
                if executor.runtime_template is not None:
                    _execute_intrinsic_closure(
                        plan,
                        row,
                        arena,
                        amplitudes,
                    )
                else:
                    _execute_prepared_row(
                        plan,
                        executor,
                        row,
                        momenta,
                        prepared_parameters,
                        arena,
                        amplitudes,
                        precision,
                    )
        else:  # pragma: no cover - native plan validation rejects this
            raise ArtifactError(f"unsupported recurrence row role {group.role}")
    return tuple(amplitudes)


def _momentum_forms(
    plan: _RecurrenceExactPlan,
    point: _Point,
    permutation: Sequence[int],
) -> tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...]:
    result = []
    terms = plan.sections.momentum_terms
    for form in plan.sections.momentum_forms:
        values = [_ZERO, _ZERO, _ZERO, _ZERO]
        for term in terms[form.term_start : form.term_start + form.term_count]:
            try:
                external_slot = permutation[term.source_slot]
                source = point[external_slot]
            except IndexError as exc:
                raise ArtifactError(
                    "recurrence momentum form references an absent external source"
                ) from exc
            coefficient = Decimal(term.coefficient)
            for component in range(4):
                values[component] += coefficient * source[component]
        result.append((values[0], values[1], values[2], values[3]))
    return tuple(result)


def _clear_stage(
    plan: _RecurrenceExactPlan,
    arena: list[_ComplexDecimal],
    stage: int,
) -> None:
    for current in plan.sections.currents:
        if current.node_kind != _NODE_CURRENT or current.stage != stage:
            continue
        stop = current.component_base + current.component_count
        arena[current.component_base : stop] = [
            _complex_zero() for _ in range(current.component_count)
        ]


def _execute_source(
    plan: _RecurrenceExactPlan,
    row: object,
    momenta: Sequence[Sequence[Decimal]],
    prepared_parameters: Sequence[_ComplexDecimal],
    arena: list[_ComplexDecimal],
) -> None:
    source = row
    template = plan.source_templates[source.source_template_or_dispatch_domain]
    initial = source.source_slot in plan.initial_source_slots
    helicity = template.helicity * (template.crossing_helicity_factor if initial else 1)
    chirality = template.chirality * (
        template.crossing_chirality_factor if initial else 1
    )
    spin_state = template.spin_state * (
        template.crossing_spin_state_factor if initial else 1
    )
    if spin_state != source.spin_state_class:
        raise ArtifactError("recurrence source spin-state contract is inconsistent")
    try:
        momentum = momenta[source.momentum_form_id]
    except IndexError as exc:
        raise ArtifactError("recurrence source momentum form is absent") from exc
    mass = _ZERO
    if template.mass_prepared_parameter_id is not None:
        try:
            mass_value = prepared_parameters[template.mass_prepared_parameter_id]
        except IndexError as exc:
            raise ArtifactError(
                "recurrence source mass parameter is out of range"
            ) from exc
        if mass_value[1] != _ZERO:
            raise EvaluationError("recurrence source mass must be real")
        mass = mass_value[0]
    wave = _source_wavefunction(template, momentum, helicity, chirality, mass)
    if len(wave) != template.dimension:
        raise ArtifactError("recurrence source wavefunction has the wrong dimension")
    factor = _factor(plan, source.exact_factor_id)
    values = tuple(_complex_mul(value, factor) for value in wave)
    _write(arena, source.destination_base, values, replace=True)


def _execute_union_source(
    plan: _RecurrenceExactPlan,
    source: object,
    variant: _SourceDispatchVariant,
    momenta: Sequence[Sequence[Decimal]],
    prepared_parameters: Sequence[_ComplexDecimal],
    arena: list[_ComplexDecimal],
) -> None:
    if (
        source.source_template_or_dispatch_domain != variant.dispatch_domain_id
        or variant.source_row_id >= len(plan.sections.sources)
        or plan.sections.sources[variant.source_row_id] != source
    ):
        raise ArtifactError(
            "all-flow-union source variant does not match its dispatch row"
        )
    try:
        template = plan.source_templates[variant.source_template_id]
    except KeyError as exc:
        raise ArtifactError(
            "all-flow-union source variant references an absent source template"
        ) from exc
    try:
        momentum = momenta[source.momentum_form_id]
    except IndexError as exc:
        raise ArtifactError("recurrence source momentum form is absent") from exc

    initial = source.source_slot in plan.initial_source_slots
    helicity = template.helicity * (
        template.crossing_helicity_factor if initial else 1
    )
    chirality = template.chirality * (
        template.crossing_chirality_factor if initial else 1
    )
    spin_state = template.spin_state * (
        template.crossing_spin_state_factor if initial else 1
    )
    if spin_state != variant.crossed_spin_state_class:
        raise ArtifactError(
            "all-flow-union source variant has inconsistent crossed spin state"
        )

    mass = _ZERO
    if template.mass_prepared_parameter_id is not None:
        try:
            mass_value = prepared_parameters[template.mass_prepared_parameter_id]
        except IndexError as exc:
            raise ArtifactError(
                "recurrence source mass parameter is out of range"
            ) from exc
        if mass_value[1] != _ZERO:
            raise EvaluationError("recurrence source mass must be real")
        mass = mass_value[0]
    wave = _source_wavefunction(template, momentum, helicity, chirality, mass)
    if len(wave) != variant.projection_count:
        raise ArtifactError(
            "all-flow-union source wavefunction has the wrong projected dimension"
        )

    embeddings = plan.sections.source_embeddings[
        variant.embedding_start : variant.embedding_start + variant.embedding_count
    ]
    if len(embeddings) != variant.embedding_count:
        raise ArtifactError("all-flow-union source embedding is out of bounds")
    source_factor = _complex_mul(
        _factor(plan, source.exact_factor_id),
        _factor(plan, variant.crossing_exact_factor_id),
    )
    values = []
    for full_component, embedding in enumerate(embeddings):
        if embedding.full_component != full_component:
            raise ArtifactError(
                "all-flow-union source embedding is not in component order"
            )
        if embedding.source_component == DIRECT_NONE_U32:
            values.append(_complex_zero())
            continue
        try:
            value = wave[embedding.source_component]
        except IndexError as exc:
            raise ArtifactError(
                "all-flow-union source embedding references an absent component"
            ) from exc
        values.append(
            _complex_mul(
                _complex_mul(value, source_factor),
                _factor(plan, embedding.exact_factor_id),
            )
        )
    _write(arena, source.destination_base, values, replace=True)


def _source_wavefunction(
    template: _SourceTemplate,
    momentum: Sequence[Decimal],
    helicity: int,
    chirality: int,
    mass: Decimal,
) -> tuple[_ComplexDecimal, ...]:
    if template.dimension == 1 and template.family == "scalar":
        return ((_ONE, _ZERO),)
    if template.family == "fermion" and template.orientation == "self-conjugate":
        raise CompatibilityError(
            "self-conjugate fermion source wavefunctions are unsupported"
        )
    if template.dimension == 2 and template.family == "fermion":
        return (
            _antiquark_weyl(momentum, helicity, chirality)
            if template.orientation == "antiparticle"
            else _quark_weyl(momentum, helicity, chirality)
        )
    if template.dimension == 4 and template.family == "fermion":
        return (
            _antiquark_dirac(momentum, helicity, mass)
            if template.orientation == "antiparticle"
            else _quark_dirac(momentum, helicity, mass)
        )
    if template.dimension == 4 and template.family == "vector":
        return (
            _massless_vector(momentum, helicity)
            if mass == _ZERO
            else _massive_vector(momentum, helicity, mass)
        )
    if template.dimension == 16 and template.family == "spin2":
        return _spin2(momentum, helicity, mass)
    raise CompatibilityError(
        f"exact recurrence source {template.family!r} with dimension "
        f"{template.dimension} is unsupported"
    )


def _execute_finalization(
    plan: _RecurrenceExactPlan,
    executor: _Executor,
    row: _Finalization,
    momenta: Sequence[Sequence[Decimal]],
    prepared_parameters: Sequence[_ComplexDecimal],
    arena: list[_ComplexDecimal],
    amplitudes: list[_ComplexDecimal],
    precision: int,
) -> None:
    if executor.runtime_template is not None:
        if executor.runtime_template != "rusticol.identity-finalize-in-place.v1":
            raise CompatibilityError(
                f"unsupported exact recurrence finalization intrinsic "
                f"{executor.runtime_template!r}"
            )
        factor = _factor(plan, row.exact_factor_id)
        stop = row.component_base + row.component_count
        arena[row.component_base : stop] = [
            _complex_mul(value, factor) for value in arena[row.component_base : stop]
        ]
        return
    _execute_prepared_row(
        plan,
        executor,
        row,
        momenta,
        prepared_parameters,
        arena,
        amplitudes,
        precision,
    )


def _execute_prepared_row(
    plan: _RecurrenceExactPlan,
    executor: _Executor,
    row: _Contribution | _Finalization | _Closure,
    momenta: Sequence[Sequence[Decimal]],
    prepared_parameters: Sequence[_ComplexDecimal],
    arena: list[_ComplexDecimal],
    amplitudes: list[_ComplexDecimal],
    precision: int,
) -> None:
    if executor.prepared_kernel_id is None:
        raise ArtifactError(
            f"direct executor {executor.executor_id} has no prepared exact kernel"
        )
    kernel = plan.kernels.get(executor.prepared_kernel_id)
    if kernel is None:
        raise ArtifactError(
            f"prepared exact kernel {executor.prepared_kernel_id} is absent"
        )
    inputs = _kernel_inputs(
        plan,
        kernel.record.input_contracts,
        executor,
        row,
        momenta,
        prepared_parameters,
        arena,
    )
    outputs = kernel.evaluate(inputs, precision)
    factor = _factor(plan, row.exact_factor_id)
    scaled = tuple(_complex_mul(value, factor) for value in outputs)
    if executor.role == "contribution":
        _write(arena, row.destination_base, scaled, replace=False)
    elif executor.role == "finalization":
        _write(arena, row.component_base, scaled, replace=True)
    elif executor.role == "closure":
        if len(scaled) != executor.destination_component_count:
            raise ArtifactError("recurrence closure output width is inconsistent")
        _write(
            amplitudes,
            row.amplitude_destination_id,
            scaled,
            replace=False,
        )
    else:
        raise ArtifactError(
            f"prepared direct executor has unsupported role {executor.role!r}"
        )


def _kernel_inputs(
    plan: _RecurrenceExactPlan,
    contracts: Sequence[Mapping[str, object]],
    executor: _Executor,
    row: _Contribution | _Finalization | _Closure,
    momenta: Sequence[Sequence[Decimal]],
    prepared_parameters: Sequence[_ComplexDecimal],
    arena: Sequence[_ComplexDecimal],
) -> tuple[_ComplexDecimal, ...]:
    result = []
    for index, contract in enumerate(contracts):
        role = contract.get("role")
        component = contract.get("component")
        if (
            isinstance(component, bool)
            or not isinstance(component, int)
            or component < 0
        ):
            raise ArtifactError(
                f"prepared kernel input {index} has an invalid component"
            )
        if role in {"left-current", "current"}:
            result.append(
                _arena_value(
                    arena,
                    _parent_component_base(executor.role, row, 0),
                    component,
                )
            )
        elif role == "right-current":
            result.append(
                _arena_value(
                    arena,
                    _parent_component_base(executor.role, row, 1),
                    component,
                )
            )
        elif role in {"left-momentum", "momentum"}:
            result.append(
                _momentum_value(
                    momenta,
                    _momentum_form_id(executor.role, row, 0),
                    component,
                )
            )
        elif role == "right-momentum":
            result.append(
                _momentum_value(
                    momenta,
                    _momentum_form_id(executor.role, row, 1),
                    component,
                )
            )
        elif role == "coupling-real":
            coupling = plan.executor_couplings.get(executor.executor_id)
            if coupling is None:
                raise ArtifactError(
                    f"direct executor {executor.executor_id} has no exact coupling"
                )
            result.append((coupling[0], _ZERO))
        elif role == "coupling-imag":
            coupling = plan.executor_couplings.get(executor.executor_id)
            if coupling is None:
                raise ArtifactError(
                    f"direct executor {executor.executor_id} has no exact coupling"
                )
            result.append((coupling[1], _ZERO))
        elif role == "model-parameter":
            parameter_id = contract.get("model_parameter_index")
            if (
                isinstance(parameter_id, bool)
                or not isinstance(parameter_id, int)
                or parameter_id < 0
            ):
                raise ArtifactError(
                    f"prepared kernel input {index} has no model-parameter index"
                )
            result.append(_parameter(prepared_parameters, parameter_id))
        else:
            raise CompatibilityError(
                f"unsupported exact recurrence kernel input role {role!r}"
            )
    return tuple(result)


def _parent_component_base(
    role: str,
    row: _Contribution | _Finalization | _Closure,
    parent: int,
) -> int:
    if role == "finalization":
        if parent == 0 and isinstance(row, _Finalization):
            return row.component_base
    elif role in {"contribution", "closure"} and not isinstance(row, _Finalization):
        if parent == 0:
            return row.parent0_base
        if parent == 1 and row.parent1_base != DIRECT_NONE_U32:
            return row.parent1_base
    raise ArtifactError(f"recurrence {role} row has no parent current {parent}")


def _momentum_form_id(
    role: str,
    row: _Contribution | _Finalization | _Closure,
    operand: int,
) -> int:
    if role == "finalization":
        if operand == 0 and isinstance(row, _Finalization):
            return row.momentum_form_id
    elif role in {"contribution", "closure"} and not isinstance(row, _Finalization):
        if operand == 0:
            return row.parent0_momentum
        if operand == 1 and row.parent1_momentum != DIRECT_NONE_U32:
            return row.parent1_momentum
    raise ArtifactError(f"recurrence {role} row has no momentum operand {operand}")


def _execute_intrinsic_closure(
    plan: _RecurrenceExactPlan,
    row: object,
    arena: Sequence[_ComplexDecimal],
    amplitudes: list[_ComplexDecimal],
) -> None:
    if row.parent1_base == DIRECT_NONE_U32 or row.component_count == 0:
        raise ArtifactError("recurrence intrinsic closure has invalid parents")
    row_factor = _factor(plan, row.exact_factor_id)
    value = _complex_zero()
    for component in range(row.component_count):
        coefficient = _factor(plan, row.component_factor_start + component)
        left = _arena_value(arena, row.parent0_base, component)
        right = _arena_value(arena, row.parent1_base, component)
        value = _complex_add(
            value,
            _complex_mul(_complex_mul(left, right), coefficient),
        )
    value = _complex_mul(value, row_factor)
    amplitudes[row.amplitude_destination_id] = _complex_add(
        amplitudes[row.amplitude_destination_id],
        value,
    )


def _factor(plan: _RecurrenceExactPlan, factor_id: int) -> _ComplexDecimal:
    try:
        value = plan.sections.exact_factors[factor_id]
    except IndexError as exc:
        raise ArtifactError("recurrence exact factor is out of range") from exc
    return (
        Decimal(value.real_numerator) / Decimal(value.real_denominator),
        Decimal(value.imaginary_numerator) / Decimal(value.imaginary_denominator),
    )


def _parameter(
    parameters: Sequence[_ComplexDecimal],
    parameter_id: int,
) -> _ComplexDecimal:
    try:
        return parameters[parameter_id]
    except IndexError as exc:
        raise ArtifactError("recurrence prepared parameter is out of range") from exc


def _arena_value(
    values: Sequence[_ComplexDecimal],
    base: int,
    component: int,
) -> _ComplexDecimal:
    try:
        return values[base + component]
    except IndexError as exc:
        raise ArtifactError("recurrence component reference is out of range") from exc


def _momentum_value(
    momenta: Sequence[Sequence[Decimal]],
    form_id: int,
    component: int,
) -> _ComplexDecimal:
    try:
        return momenta[form_id][component], _ZERO
    except IndexError as exc:
        raise ArtifactError("recurrence momentum reference is out of range") from exc


def _write(
    values: list[_ComplexDecimal],
    start: int,
    entries: Sequence[_ComplexDecimal],
    *,
    replace: bool,
) -> None:
    stop = start + len(entries)
    if start < 0 or stop > len(values):
        raise ArtifactError("recurrence destination range is out of bounds")
    if replace:
        values[start:stop] = entries
    else:
        values[start:stop] = [
            _complex_add(previous, value)
            for previous, value in zip(values[start:stop], entries, strict=True)
        ]


def _role_index(role: str) -> int:
    try:
        return {
            "source": _ROLE_SOURCE,
            "contribution": _ROLE_CONTRIBUTION,
            "finalization": _ROLE_FINALIZATION,
            "closure": _ROLE_CLOSURE,
        }[role]
    except KeyError as exc:
        raise ArtifactError(f"unsupported direct executor role {role!r}") from exc


__all__ = ["_evaluate_replay_point", "_evaluate_union_point"]
