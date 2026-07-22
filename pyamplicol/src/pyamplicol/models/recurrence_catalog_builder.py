# SPDX-License-Identifier: 0BSD
"""Build the process-independent recurrence semantic catalog.

The builder deliberately consumes only model contracts and prepared-kernel
records.  It does not inspect a process and it does not classify a model by
name or Python type.  Any semantic relation that is not represented by those
contracts is rejected rather than inferred from a familiar particle inventory.

Prepared source-fill kernels are not part of the pre-existing eager kernel
catalog.  Recurrence companions therefore bind source semantics to generic
Rusticol source-fill templates, while vertex, propagator, closure, and model
parameter callables retain their prepared-kernel bindings.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from .base import Model, Vertex
from .prepared_catalog import (
    PREPARED_HOMOGENEOUS_LINEAR_CURRENT_PROOF,
    PreparedClosureBinding,
    PreparedKernelCatalog,
    PreparedKernelCatalogError,
    PreparedKernelInput,
    PreparedParticleState,
    PreparedPropagatorBinding,
    PreparedVertexBinding,
)
from .recurrence_template import (
    ClosureTemplateV1,
    ColorContractionTemplateV1,
    CurrentStateTemplateV1,
    EvaluatorBindingV1,
    ExactComplexRationalV1,
    ParameterTemplateV1,
    PropagatorTemplateV1,
    QuantumFlowTemplateV1,
    RecurrenceTemplateCatalog,
    RecurrenceTemplateError,
    SourceTemplateV1,
    SymmetryProofV1,
    TransitionTemplateV1,
)

_PREPARED_CATALOG_ABI = "pyamplicol-prepared-kernel-catalog-v1"
_SUPPORTED_COLOR_RULES = frozenset(
    {
        "singlet",
        "color-identity",
        "fundamental-generator",
        "adjoint-structure-constant",
        "adjoint-structure-constant-product",
    }
)


def build_recurrence_template_catalog(
    model: Model,
    prepared_catalog: PreparedKernelCatalog,
    *,
    compiled_model_digest: str,
) -> RecurrenceTemplateCatalog:
    """Project one model and its exact prepared kernels into recurrence v1.

    The construction is deterministic and process independent.  Callback
    results that affect recurrence semantics are sampled twice and compared in
    canonical form; a stateful or nondeterministic model therefore fails
    closed.  All prepared callables are independently signature-checked before
    their IDs are admitted into the recurrence catalog.
    """

    _require_digest(compiled_model_digest, "compiled model digest")
    _validate_prepared_catalog_identity(model, prepared_catalog)
    kernels = _validated_kernels(prepared_catalog)
    recurrence_catalog = _recurrence_prepared_catalog(model, prepared_catalog)

    prepared_states = _collect_prepared_states(recurrence_catalog)
    parameters = _build_parameters(model, recurrence_catalog, prepared_states)
    parameter_ids = {parameter.name: parameter.template_id for parameter in parameters}
    current_states = _build_current_states(model, prepared_states, parameter_ids)
    state_ids = {
        (state.particle_id, state.chirality): state.template_id
        for state in current_states
    }
    state_by_id = {state.template_id: state for state in current_states}

    evaluator_builders: list[_EvaluatorRequest] = []
    sources, source_evaluator_bindings = _build_sources(
        model,
        prepared_states,
        state_ids,
        parameter_ids,
    )
    (
        quantum_flows,
        transitions,
        transition_colors,
    ) = _build_transitions(
        model,
        recurrence_catalog.vertex_bindings,
        state_ids,
        parameter_ids,
        kernels,
        evaluator_builders,
    )
    propagators, propagator_proofs = _build_propagators(
        model,
        recurrence_catalog.propagator_bindings,
        state_ids,
        parameter_ids,
        kernels,
        evaluator_builders,
    )
    prepared_closures, prepared_closure_colors = _build_closures(
        model,
        recurrence_catalog.closure_bindings,
        state_ids,
        parameter_ids,
        kernels,
        evaluator_builders,
    )
    direct_closures, direct_closure_colors, direct_closure_bindings = (
        _build_direct_closures(
            model,
            state_ids,
            state_by_id,
        )
    )
    closures = (*prepared_closures, *direct_closures)
    closure_colors = (*prepared_closure_colors, *direct_closure_colors)
    _add_model_parameter_evaluator(
        recurrence_catalog,
        parameters,
        kernels,
        evaluator_builders,
    )
    evaluator_bindings = (
        *source_evaluator_bindings,
        *direct_closure_bindings,
        *_coalesce_evaluator_requests(evaluator_builders),
    )

    colors_by_id = {
        color.template_id: color for color in (*transition_colors, *closure_colors)
    }
    return RecurrenceTemplateCatalog.create(
        compiled_model_digest=compiled_model_digest,
        parameters=parameters,
        current_states=current_states,
        sources=sources,
        quantum_flows=quantum_flows,
        transitions=transitions,
        propagators=propagators,
        closures=closures,
        color_contractions=tuple(colors_by_id.values()),
        symmetry_proofs=propagator_proofs,
        evaluator_bindings=evaluator_bindings,
    )


class _EvaluatorRequest:
    __slots__ = (
        "contract_kind",
        "input_state_template_ids",
        "kernel",
        "output_state_template_id",
        "semantic_template_id",
    )

    def __init__(
        self,
        *,
        kernel: Any,
        contract_kind: str,
        input_state_template_ids: tuple[str, ...],
        output_state_template_id: str | None,
        semantic_template_id: str,
    ) -> None:
        self.kernel = kernel
        self.contract_kind = contract_kind
        self.input_state_template_ids = input_state_template_ids
        self.output_state_template_id = output_state_template_id
        self.semantic_template_id = semantic_template_id


def _canonical_json(payload: object) -> str:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RecurrenceTemplateError(
            "recurrence model callback returned noncanonical metadata"
        ) from exc


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _expression_digest(expression: str) -> str:
    if not isinstance(expression, str) or not expression:
        raise PreparedKernelCatalogError(
            "prepared kernel exact expressions must be nonempty strings"
        )
    return hashlib.sha256(expression.encode("utf-8")).hexdigest()


def _require_digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RecurrenceTemplateError(f"{context} must be a lowercase SHA-256")
    return value


def _token(prefix: str, payload: object) -> str:
    return f"{prefix}:{_digest(payload)[:24]}"


def _exact_pair(value: object, context: str) -> ExactComplexRationalV1:
    if isinstance(value, complex):
        pair = (float(value.real), float(value.imag))
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        ):
            raise RecurrenceTemplateError(
                f"{context} must contain two real binary64 components"
            )
        pair = (float(value[0]), float(value[1]))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        pair = (float(value), 0.0)
    else:
        raise RecurrenceTemplateError(
            f"{context} must be a real or complex binary64 value"
        )
    if not all(math.isfinite(component) for component in pair):
        raise RecurrenceTemplateError(f"{context} must be finite")
    return ExactComplexRationalV1.from_binary64(*pair)


def _multiply_exact(
    left: ExactComplexRationalV1,
    right: ExactComplexRationalV1,
) -> ExactComplexRationalV1:
    real = left.real * right.real - left.imag * right.imag
    imag = left.real * right.imag + left.imag * right.real
    return ExactComplexRationalV1.from_fractions(real, imag)


def _validate_prepared_catalog_identity(
    model: Model,
    prepared_catalog: PreparedKernelCatalog,
) -> None:
    model_name = getattr(model, "name", None)
    if not isinstance(model_name, str) or not model_name:
        raise RecurrenceTemplateError("recurrence model name must be nonempty")
    if getattr(prepared_catalog, "model_name", None) != model_name:
        raise PreparedKernelCatalogError(
            "prepared kernel catalog model identity does not match the recurrence "
            f"model: {getattr(prepared_catalog, 'model_name', None)!r} != "
            f"{model_name!r}"
        )
    gaps = tuple(getattr(prepared_catalog, "unsupported_variants", ()))
    if gaps:
        details = "; ".join(
            f"{gap.contract_kind}:{gap.context}:{gap.reason}" for gap in gaps
        )
        raise PreparedKernelCatalogError(
            "prepared kernel catalog has unsupported recurrence semantics: " + details
        )


def _validated_kernels(
    prepared_catalog: PreparedKernelCatalog,
) -> Mapping[int, Any]:
    kernels = tuple(getattr(prepared_catalog, "kernels", ()))
    expected_ids = tuple(range(len(kernels)))
    actual_ids = tuple(getattr(kernel, "kernel_id", None) for kernel in kernels)
    if actual_ids != expected_ids:
        raise PreparedKernelCatalogError(
            "prepared kernel IDs must be contiguous before recurrence catalog build"
        )
    by_id: dict[int, Any] = {}
    signatures: list[str] = []
    for kernel in kernels:
        payload = {
            "abi": _PREPARED_CATALOG_ABI,
            "contract_kind": kernel.contract_kind,
            "inputs": [item.to_dict() for item in kernel.inputs],
            "outputs": list(kernel.exact_expressions),
            "output_layout": list(kernel.output_layout),
        }
        expected = _digest(payload)
        if kernel.canonical_signature != expected:
            raise PreparedKernelCatalogError(
                f"prepared kernel {kernel.kernel_id} has a stale canonical signature"
            )
        if len(kernel.exact_expressions) != len(kernel.output_layout):
            raise PreparedKernelCatalogError(
                f"prepared kernel {kernel.kernel_id} has misaligned outputs"
            )
        by_id[int(kernel.kernel_id)] = kernel
        signatures.append(str(kernel.canonical_signature))
    if tuple(signatures) != tuple(sorted(signatures)) or len(set(signatures)) != len(
        signatures
    ):
        raise PreparedKernelCatalogError(
            "prepared kernel signatures must be sorted and unique"
        )
    return by_id


def _collect_prepared_states(
    prepared_catalog: PreparedKernelCatalog,
) -> tuple[PreparedParticleState, ...]:
    states: dict[tuple[int, int], PreparedParticleState] = {}
    bindings: Sequence[Any] = (
        *prepared_catalog.vertex_bindings,
        *prepared_catalog.propagator_bindings,
        *prepared_catalog.closure_bindings,
    )
    for binding in bindings:
        candidates = tuple(
            state
            for name in ("left_state", "right_state", "result_state", "state")
            for state in (getattr(binding, name, None),)
            if state is not None
        )
        for state in candidates:
            key = (int(state.particle_id), int(state.chirality))
            previous = states.setdefault(key, state)
            if previous != state:
                raise PreparedKernelCatalogError(
                    "prepared bindings disagree on current-state semantics for "
                    f"particle={key[0]}, chirality={key[1]}"
                )
    return tuple(states[key] for key in sorted(states))


def _recurrence_prepared_catalog(
    model: Model,
    prepared_catalog: PreparedKernelCatalog,
) -> PreparedKernelCatalog:
    """Return the process-independent subset admitted by recurrence semantics.

    Prepared catalogs intentionally retain every exact model kernel useful to
    eager execution, including UFO ghost-interaction kernels.  Tree-level
    recurrence execution has no ghost source/current contract in v1.  Filter
    such bindings from the recurrence companion using the typed model source
    contract, without changing the prepared catalog or recognizing a model by
    name, particle ID, or Python implementation type.

    A future model that gives ghosts explicit recurrence semantics can extend
    the public source/runtime template ABI.  Until then, a process whose only
    construction needs an excluded binding will fail during process lowering
    because the semantic transition is absent; no unsupported identity is
    silently introduced.
    """

    states = _collect_prepared_states(prepared_catalog)
    admitted = {
        (state.particle_id, state.chirality)
        for state in states
        if _state_is_admitted_by_recurrence(model, state)
    }

    def state_key(state: PreparedParticleState) -> tuple[int, int]:
        return int(state.particle_id), int(state.chirality)

    vertex_bindings = tuple(
        binding
        for binding in prepared_catalog.vertex_bindings
        if all(
            state_key(state) in admitted
            for state in (
                binding.left_state,
                binding.right_state,
                binding.result_state,
            )
        )
    )
    propagator_bindings = tuple(
        binding
        for binding in prepared_catalog.propagator_bindings
        if state_key(binding.state) in admitted
    )
    closure_bindings = tuple(
        binding
        for binding in prepared_catalog.closure_bindings
        if all(
            state_key(state) in admitted
            for state in (
                binding.left_state,
                binding.right_state,
                binding.result_state,
            )
        )
    )
    return PreparedKernelCatalog(
        model_name=prepared_catalog.model_name,
        kernels=prepared_catalog.kernels,
        vertex_bindings=vertex_bindings,
        propagator_bindings=propagator_bindings,
        closure_bindings=closure_bindings,
        model_parameter_kernel_id=prepared_catalog.model_parameter_kernel_id,
        unsupported_variants=prepared_catalog.unsupported_variants,
    )


def _state_is_admitted_by_recurrence(
    model: Model,
    state: PreparedParticleState,
) -> bool:
    if model.auxiliary_kind(state.particle_id) is not None:
        return True
    wavefunction_family = _stable_callback(
        f"wavefunction family for particle {state.particle_id}",
        lambda: model.source_wavefunction_kind(state.particle_id),
        serializer=str,
    )
    if wavefunction_family == "ghost":
        return False
    if (
        _effective_auxiliary_kind(
            model,
            state.particle_id,
            state.chirality,
        )
        is not None
    ):
        return True
    source_ir = _stable_callback(
        f"source contract for particle {state.particle_id}",
        lambda: model._source_ir(state.particle_id),
        serializer=_canonical_source_ir,
    )
    if source_ir.statistics == "ghost" or source_ir.wavefunction_family == "ghost":
        return False
    if source_ir.statistics not in {"boson", "fermion", "auxiliary"}:
        raise RecurrenceTemplateError(
            "recurrence-template-v1 cannot classify source statistics "
            f"{source_ir.statistics!r} for particle={state.particle_id}"
        )
    # Require the corresponding finalization contract now.  This keeps missing
    # physical-state semantics distinct from deliberately unsupported ghosts.
    _stable_callback(
        f"propagator contract for particle {state.particle_id}",
        lambda: model._propagator_ir(state.particle_id, state.chirality),
        serializer=lambda value: _canonical_json(value.to_json_dict()),
    )
    return True


def _build_parameters(
    model: Model,
    prepared_catalog: PreparedKernelCatalog,
    prepared_states: Sequence[PreparedParticleState],
) -> tuple[ParameterTemplateV1, ...]:
    defaults: dict[str, ExactComplexRationalV1] = {}
    kinds: dict[str, str] = {}
    value_types: dict[str, str] = {}
    prepared_parameter_ids: dict[str, int] = {}

    for kernel in prepared_catalog.kernels:
        for item in kernel.inputs:
            if item.model_parameter_name is None:
                continue
            if item.model_parameter_index is None:
                raise RecurrenceTemplateError(
                    "prepared model-parameter input lacks its stable index"
                )
            name = str(item.model_parameter_name)
            index = int(item.model_parameter_index)
            previous = prepared_parameter_ids.setdefault(name, index)
            if previous != index:
                raise RecurrenceTemplateError(
                    f"prepared model parameter {name!r} has conflicting indices"
                )
    names_by_prepared_id: dict[int, str] = {}
    for name, index in prepared_parameter_ids.items():
        previous = names_by_prepared_id.setdefault(index, name)
        if previous != name:
            raise RecurrenceTemplateError(
                f"prepared parameter index {index} names both {previous!r} and "
                f"{name!r}"
            )

    def add_default(name: object, value: object, *, kind: str) -> None:
        parameter_name = str(name)
        exact = _exact_pair(value, f"model parameter {parameter_name!r} default")
        previous = defaults.setdefault(parameter_name, exact)
        if previous != exact:
            raise RecurrenceTemplateError(
                f"model parameter {parameter_name!r} has conflicting defaults"
            )
        kinds.setdefault(parameter_name, kind)
        value_types.setdefault(
            parameter_name,
            "real" if exact.imag_numerator == 0 else "complex",
        )

    defaults_provider = getattr(model, "runtime_parameter_defaults", None)
    if callable(defaults_provider):
        first = dict(defaults_provider())
        second = dict(defaults_provider())
        if _canonical_binary64_mapping(first) != _canonical_binary64_mapping(second):
            raise RecurrenceTemplateError(
                "runtime_parameter_defaults is nondeterministic"
            )
        for name, value in first.items():
            add_default(name, value, kind="external")
            type_provider = getattr(model, "runtime_parameter_type", None)
            if callable(type_provider):
                declared = str(type_provider(str(name))).lower()
                if declared not in {"real", "complex"}:
                    raise RecurrenceTemplateError(
                        f"unsupported parameter type {declared!r} for {name!r}"
                    )
                value_types[str(name)] = declared

    normalization_provider = getattr(
        model, "runtime_normalization_parameter_defaults", None
    )
    if callable(normalization_provider):
        first = dict(normalization_provider())
        second = dict(normalization_provider())
        if _canonical_binary64_mapping(first) != _canonical_binary64_mapping(second):
            raise RecurrenceTemplateError(
                "runtime_normalization_parameter_defaults is nondeterministic"
            )
        for name, value in first.items():
            add_default(name, value, kind="external")

    definitions_provider = getattr(model, "runtime_derived_parameter_definitions", None)
    definitions = (
        {str(name): str(value) for name, value in definitions_provider().items()}
        if callable(definitions_provider)
        else {}
    )
    if callable(definitions_provider):
        repeated = {
            str(name): str(value) for name, value in definitions_provider().items()
        }
        if definitions != repeated:
            raise RecurrenceTemplateError(
                "runtime_derived_parameter_definitions is nondeterministic"
            )

    required_names = {
        str(item.model_parameter_name)
        for kernel in prepared_catalog.kernels
        for item in kernel.inputs
        if item.model_parameter_name is not None
    }
    required_names.update(
        str(name)
        for binding in prepared_catalog.propagator_bindings
        for name in binding.model_parameters
    )
    for state in prepared_states:
        if (
            _effective_auxiliary_kind(model, state.particle_id, state.chirality)
            is not None
        ):
            continue
        source_ir = _stable_callback(
            f"source contract for particle {state.particle_id}",
            lambda state=state: model._source_ir(state.particle_id),
            serializer=lambda value: _canonical_source_ir(value),
        )
        for name in (source_ir.mass_parameter, source_ir.width_parameter):
            if name is not None:
                required_names.add(str(name))
        if (
            source_ir.mass_parameter is not None
            and source_ir.mass_parameter not in defaults
            and source_ir.mass_parameter not in definitions
        ):
            add_default(
                source_ir.mass_parameter,
                model.mass(state.particle_id),
                kind="external",
            )
        if (
            source_ir.width_parameter is not None
            and source_ir.width_parameter not in defaults
            and source_ir.width_parameter not in definitions
        ):
            add_default(
                source_ir.width_parameter,
                model.width(state.particle_id),
                kind="external",
            )

        # Match the model-generic fallback naming contract used by the
        # prepared-kernel builder when a model does not expose named particle
        # parameters itself.
        particle = model.particle(state.particle_id)
        fallback_values = {
            f"particle.{int(particle.pdg)}.mass": model.mass(state.particle_id),
            f"particle.{int(particle.pdg)}.width": model.width(state.particle_id),
        }
        for name, value in fallback_values.items():
            if (
                name in required_names
                and name not in defaults
                and name not in definitions
            ):
                add_default(name, value, kind="external")

    derived_defaults_provider = getattr(
        model, "runtime_derived_parameter_defaults", None
    )
    derived_defaults = (
        dict(derived_defaults_provider()) if callable(derived_defaults_provider) else {}
    )
    for name, value in derived_defaults.items():
        add_default(name, value, kind="derived")

    missing = sorted(required_names - defaults.keys() - definitions.keys())
    if missing:
        raise RecurrenceTemplateError(
            "recurrence catalog cannot resolve defaults or exact definitions for "
            "prepared model parameters: " + ", ".join(missing)
        )

    all_names = tuple(sorted(defaults.keys() | definitions.keys()))
    ids = {name: _token("parameter", {"name": name}) for name in all_names}
    external_ids = tuple(
        sorted(ids[name] for name in all_names if name not in definitions)
    )
    records: list[ParameterTemplateV1] = []
    for name in all_names:
        derived = name in definitions
        default = defaults.get(name)
        declared = value_types.get(
            name,
            "complex"
            if default is not None and default.imag_numerator != 0
            else "real",
        )
        if declared == "real" and default is not None and default.imag_numerator != 0:
            raise RecurrenceTemplateError(
                f"real model parameter {name!r} has a complex default"
            )
        records.append(
            ParameterTemplateV1(
                template_id=ids[name],
                name=name,
                parameter_kind="derived" if derived else "external",
                value_type=declared,  # type: ignore[arg-type]
                mutable=not derived,
                default_value=None if derived else default,
                exact_expression_digest=(
                    _expression_digest(definitions[name]) if derived else None
                ),
                # A superset is conservative and avoids parsing model syntax as
                # proof. Rust may use the evaluator's actual input layout.
                dependency_parameter_ids=external_ids if derived else (),
                prepared_parameter_id=prepared_parameter_ids.get(name),
            )
        )
    return tuple(records)


def _canonical_binary64_mapping(values: Mapping[object, object]) -> str:
    return _canonical_json(
        {
            str(name): _exact_pair(value, f"parameter {name!r}").to_dict()
            for name, value in values.items()
        }
    )


def _build_current_states(
    model: Model,
    prepared_states: Sequence[PreparedParticleState],
    parameter_ids: Mapping[str, str],
) -> tuple[CurrentStateTemplateV1, ...]:
    records: list[CurrentStateTemplateV1] = []
    for state in prepared_states:
        auxiliary_kind = _effective_auxiliary_kind(
            model, state.particle_id, state.chirality
        )
        if auxiliary_kind is None:
            source_ir = _stable_callback(
                f"source contract for particle {state.particle_id}",
                lambda state=state: model._source_ir(state.particle_id),
                serializer=lambda value: _canonical_source_ir(value),
            )
            identity = source_ir.identity
            basis = model._current_basis(state.particle_id, state.chirality)
            statistics = source_ir.statistics
            mass_parameter = source_ir.mass_parameter
            width_parameter = source_ir.width_parameter
        else:
            identity = _stable_callback(
                f"auxiliary identity for particle {state.particle_id}",
                lambda state=state: model._particle_identity_ir(state.particle_id),
                serializer=lambda value: _canonical_json(value.to_json_dict()),
            )
            basis = model._current_basis(state.particle_id, state.chirality)
            statistics = "fermion" if model.is_fermion(state.particle_id) else "boson"
            propagator = model._propagator_ir(state.particle_id, state.chirality)
            mass_parameter = propagator.mass_parameter
            width_parameter = propagator.width_parameter
        if (
            identity.canonical_id != state.identity
            or identity.orientation != state.orientation
            or basis != state.basis
            or int(model.current_dimension(state.particle_id, state.chirality))
            != state.dimension
        ):
            raise PreparedKernelCatalogError(
                "prepared current-state metadata does not match the live model "
                f"contract for particle={state.particle_id}, "
                f"chirality={state.chirality}"
            )
        if statistics == "ghost":
            raise RecurrenceTemplateError(
                f"recurrence-template-v1 cannot encode ghost current {state.identity!r}"
            )
        public_statistics = "fermion" if statistics == "fermion" else "boson"
        mass_id = _parameter_reference(
            mass_parameter, parameter_ids, "current mass parameter"
        )
        width_id = _parameter_reference(
            width_parameter, parameter_ids, "current width parameter"
        )
        records.append(
            CurrentStateTemplateV1(
                template_id=_state_template_id(state),
                particle_id=state.particle_id,
                anti_particle_id=identity.anti_pdg_label,
                species_id=identity.species_id,
                orientation=state.orientation,
                statistics=public_statistics,
                color_representation=int(model.color_rep(state.particle_id)),
                basis=state.basis,
                tensor_ordering=tuple(
                    f"{state.basis}:c{component}"
                    for component in range(state.dimension)
                ),
                dimension=state.dimension,
                chirality=state.chirality,
                auxiliary_kind=(
                    None if auxiliary_kind is None else str(auxiliary_kind)
                ),
                mass_parameter_id=mass_id,
                width_parameter_id=width_id,
            )
        )
    return tuple(records)


def _state_template_id(state: PreparedParticleState) -> str:
    return _token(
        "state",
        {
            "particle_id": state.particle_id,
            "identity": state.identity,
            "orientation": state.orientation,
            "basis": state.basis,
            "chirality": state.chirality,
            "dimension": state.dimension,
        },
    )


def _parameter_reference(
    name: str | None,
    parameter_ids: Mapping[str, str],
    context: str,
) -> str | None:
    if name is None:
        return None
    try:
        return parameter_ids[str(name)]
    except KeyError as exc:
        raise RecurrenceTemplateError(
            f"{context} {name!r} is absent from the recurrence parameter catalog"
        ) from exc


def _build_sources(
    model: Model,
    prepared_states: Sequence[PreparedParticleState],
    state_ids: Mapping[tuple[int, int], str],
    parameter_ids: Mapping[str, str],
) -> tuple[tuple[SourceTemplateV1, ...], tuple[EvaluatorBindingV1, ...]]:
    records: list[SourceTemplateV1] = []
    bindings: list[EvaluatorBindingV1] = []
    for prepared_state in prepared_states:
        if (
            _effective_auxiliary_kind(
                model,
                prepared_state.particle_id,
                prepared_state.chirality,
            )
            is not None
        ):
            continue
        source_ir = model._source_ir(prepared_state.particle_id)
        states = tuple(
            state
            for state in source_ir.states
            if int(state.chirality) == prepared_state.chirality
        )
        if not states:
            # Full internal fermion states may coexist with chiral external
            # source states. They are recurrence currents, not source-fill
            # entry points.
            continue
        for state in states:
            if not isinstance(state.spin_state, int) or isinstance(
                state.spin_state, bool
            ):
                raise RecurrenceTemplateError(
                    "recurrence-template-v1 cannot encode structured source spin "
                    f"state {state.spin_state!r}"
                )
            state_key = (
                prepared_state.particle_id,
                prepared_state.chirality,
            )
            mass_parameter_id = _parameter_reference(
                source_ir.mass_parameter,
                parameter_ids,
                "source mass parameter",
            )
            width_parameter_id = _parameter_reference(
                source_ir.width_parameter,
                parameter_ids,
                "source width parameter",
            )
            source_semantics = {
                "abi": "rusticol-source-fill-v1",
                "state": state_ids[state_key],
                "statistics": source_ir.statistics,
                "wavefunction_family": source_ir.wavefunction_family,
                "basis": source_ir.basis,
                "dimension": prepared_state.dimension,
                "crossing": _canonical_crossing(source_ir.crossing),
                "helicity": int(state.helicity),
                "chirality": int(state.chirality),
                "spin_state": int(state.spin_state),
                "mass_parameter_id": mass_parameter_id,
                "width_parameter_id": width_parameter_id,
            }
            callable_signature = _digest(source_semantics)
            runtime_template = (
                f"rusticol.source-fill.{source_ir.wavefunction_family}.v1:"
                f"{callable_signature[:24]}"
            )
            template_id = _token(
                "source",
                source_semantics,
            )
            resolver_key = _token("evaluator", source_semantics)
            record = SourceTemplateV1(
                template_id=template_id,
                state_template_id=state_ids[state_key],
                crossing=_canonical_crossing(source_ir.crossing),
                wavefunction_family=source_ir.wavefunction_family,
                helicity=int(state.helicity),
                spin_state=int(state.spin_state),
                wavefunction_expression_digest=callable_signature,
                evaluator_resolver_key=resolver_key,
                mass_parameter_id=mass_parameter_id,
                width_parameter_id=width_parameter_id,
            )
            records.append(record)
            input_layout = (
                "momentum:energy",
                "momentum:px",
                "momentum:py",
                "momentum:pz",
                *(() if mass_parameter_id is None else (mass_parameter_id,)),
                *(() if width_parameter_id is None else (width_parameter_id,)),
            )
            output_layout = tuple(
                f"source-component:{component}"
                for component in range(prepared_state.dimension)
            )
            bindings.append(
                EvaluatorBindingV1(
                    resolver_key=resolver_key,
                    prepared_kernel_id=None,
                    callable_kind="rusticol-template",
                    runtime_template=runtime_template,
                    contract_kind="source",
                    callable_signature=callable_signature,
                    input_state_template_ids=(),
                    output_state_template_id=record.state_template_id,
                    input_layout=input_layout,
                    output_layout=output_layout,
                    exact_expression_digests=tuple(
                        _digest({"source": source_semantics, "component": component})
                        for component in range(prepared_state.dimension)
                    ),
                    semantic_template_ids=(record.template_id,),
                )
            )
    return tuple(records), tuple(bindings)


def _canonical_crossing(crossing: Any) -> str:
    payload = {
        "momentum_transform": crossing.momentum_transform,
        "helicity_factor": int(crossing.helicity_factor),
        "chirality_factor": int(crossing.chirality_factor),
        "spin_state_factor": int(crossing.spin_state_factor),
        "phase": _exact_pair(crossing.phase, "source crossing phase").to_dict(),
    }
    return _canonical_json(payload)


def _effective_auxiliary_kind(
    model: Model,
    particle_id: int,
    chirality: int,
) -> str | None:
    declared = model.auxiliary_kind(particle_id)
    if declared is not None:
        return str(declared)
    particle = model.particle(particle_id)
    if int(particle.spin) >= 0:
        return None
    try:
        propagator = model._propagator_ir(particle_id, chirality)
    except Exception as exc:
        raise RecurrenceTemplateError(
            "recurrence-template-v1 cannot encode a negative-spin state without "
            f"an explicit auxiliary/propagator contract: particle={particle_id}, "
            f"chirality={chirality}: {exc}"
        ) from exc
    if not propagator.applies_propagator and propagator.auxiliary_policy:
        return str(propagator.auxiliary_policy)
    raise RecurrenceTemplateError(
        "recurrence-template-v1 cannot encode a negative-spin state without an "
        f"explicit auxiliary contract: particle={particle_id}, "
        f"chirality={chirality}"
    )


def _canonical_source_ir(source_ir: Any) -> str:
    payload = source_ir.to_json_dict()
    crossing = dict(payload["crossing"])
    crossing["phase"] = _exact_pair(
        crossing["phase"], "source crossing phase"
    ).to_dict()
    payload["crossing"] = crossing
    return _canonical_json(payload)


def _build_transitions(
    model: Model,
    bindings: Sequence[PreparedVertexBinding],
    state_ids: Mapping[tuple[int, int], str],
    parameter_ids: Mapping[str, str],
    kernels: Mapping[int, Any],
    evaluator_requests: list[_EvaluatorRequest],
) -> tuple[
    tuple[QuantumFlowTemplateV1, ...],
    tuple[TransitionTemplateV1, ...],
    tuple[ColorContractionTemplateV1, ...],
]:
    flows_by_id: dict[str, QuantumFlowTemplateV1] = {}
    transitions: list[TransitionTemplateV1] = []
    colors: dict[str, ColorContractionTemplateV1] = {}
    for binding in sorted(bindings, key=lambda item: item.key):
        vertex = Vertex(
            binding.key.kind,
            binding.key.particles,
            binding.key.coupling,
        )
        _validate_vertex_binding_against_model(model, binding, vertex)
        kernel = _kernel(kernels, binding.kernel_id, "vertex")
        if kernel.contract_kind != "vertex":
            raise PreparedKernelCatalogError(
                f"prepared vertex binding references a {kernel.contract_kind!r} kernel"
            )
        concrete_state_ids = (
            state_ids[(binding.left_state.particle_id, binding.left_state.chirality)],
            state_ids[(binding.right_state.particle_id, binding.right_state.chirality)],
        )
        result_state_id = state_ids[
            (binding.result_state.particle_id, binding.result_state.chirality)
        ]
        _validate_binary_kernel_layout(
            kernel,
            binding,
            output_dimension=binding.result_state.dimension,
            context=f"vertex kind {binding.key.kind}",
        )
        color = _color_template(model, vertex, closure=False)
        colors.setdefault(color.template_id, color)
        flow_variants = _probe_quantum_flows(model, vertex, binding)
        if not flow_variants:
            raise RecurrenceTemplateError(
                "prepared vertex binding is not admitted by the live quantum-flow "
                f"contract: kind={binding.key.kind}, key={binding.key!r}"
            )
        coupling_parameters = _kernel_parameter_ids(kernel, parameter_ids)
        coupling_orders = _canonical_coupling_orders(
            model.vertex_coupling_orders(vertex),
            f"vertex kind {vertex.kind} coupling orders",
        )
        for variant in flow_variants:
            flow_id = _token("quantum-flow", variant)
            flow = QuantumFlowTemplateV1(
                template_id=flow_id,
                input_state_template_ids=concrete_state_ids,
                input_spin_states=variant["input_spin_states"],
                input_flavour_flows=variant["input_flavour_flows"],
                input_quantum_number_flows=variant["input_quantum_number_flows"],
                coupling_orders=coupling_orders,
                result_state_template_id=result_state_id,
                result_flavour_flow=variant["result_flavour_flow"],
                result_quantum_number_flow=variant["result_quantum_number_flow"],
                predicate_digest=_digest(variant),
            )
            flows_by_id.setdefault(flow.template_id, flow)
            transition_id = _token(
                "transition",
                {
                    "binding": _vertex_binding_payload(binding),
                    "flow": flow.semantic_digest,
                    "color": color.semantic_digest,
                    "kernel": kernel.canonical_signature,
                },
            )
            resolver_key = _resolver_key(
                kernel,
                "vertex",
                concrete_state_ids,
                result_state_id,
            )
            transition = TransitionTemplateV1(
                template_id=transition_id,
                input_state_template_ids=concrete_state_ids,
                result_state_template_id=result_state_id,
                quantum_flow_template_id=flow.template_id,
                evaluator_resolver_key=resolver_key,
                canonical_input_order=tuple(binding.canonical_input_order),
                momentum_convention=("incoming-left", "incoming-right"),
                coupling_parameter_ids=coupling_parameters,
                coupling_orders=coupling_orders,
                color_contraction_template_id=color.template_id,
                exact_factor=_exact_pair(
                    binding.equivalence_factor,
                    f"vertex kind {vertex.kind} equivalence factor",
                ),
                output_projection=f"{binding.result_state.basis}:chirality={binding.result_state.chirality}",
            )
            transitions.append(transition)
            evaluator_requests.append(
                _EvaluatorRequest(
                    kernel=kernel,
                    contract_kind="vertex",
                    input_state_template_ids=concrete_state_ids,
                    output_state_template_id=result_state_id,
                    semantic_template_id=transition.template_id,
                )
            )
    return tuple(flows_by_id.values()), tuple(transitions), tuple(colors.values())


def _validate_vertex_binding_against_model(
    model: Model,
    binding: PreparedVertexBinding,
    vertex: Vertex,
) -> None:
    equivalence = _stable_callback(
        f"vertex equivalence for kind {vertex.kind}",
        lambda: model.vertex_evaluation_equivalence(vertex.kind),
        serializer=lambda value: _canonical_json(value.to_json_dict()),
    )
    expected = (
        tuple(equivalence.input_order),
        _exact_pair(equivalence.factor, "vertex equivalence factor"),
        None
        if equivalence.input_exchange_factor is None
        else _exact_pair(
            equivalence.input_exchange_factor,
            "vertex input-exchange factor",
        ),
        str(equivalence.class_id),
        bool(equivalence.verified),
    )
    actual = (
        tuple(binding.canonical_input_order),
        _exact_pair(binding.equivalence_factor, "prepared equivalence factor"),
        None
        if binding.input_exchange_factor is None
        else _exact_pair(
            binding.input_exchange_factor,
            "prepared input-exchange factor",
        ),
        str(binding.equivalence_class),
        True,
    )
    if expected != actual:
        raise PreparedKernelCatalogError(
            f"prepared vertex binding for kind {vertex.kind} has stale "
            "permutation/sign proof metadata"
        )


def _probe_quantum_flows(
    model: Model,
    vertex: Vertex,
    binding: PreparedVertexBinding,
) -> tuple[dict[str, Any], ...]:
    left_spins = _spin_states_for_chirality(
        model, binding.left_state.particle_id, binding.left_state.chirality
    )
    right_spins = _spin_states_for_chirality(
        model, binding.right_state.particle_id, binding.right_state.chirality
    )
    variants: dict[str, dict[str, Any]] = {}
    for left_spin in left_spins:
        for right_spin in right_spins:
            left = _flow_probe_index(model, binding.left_state, left_spin)
            right = _flow_probe_index(model, binding.right_state, right_spin)

            def evaluate(
                left: SimpleNamespace = left,
                right: SimpleNamespace = right,
                left_spin: int = left_spin,
                right_spin: int = right_spin,
            ) -> tuple[dict[str, Any], ...]:
                try:
                    flows = model.allowed_quantum_flows(vertex, left, right)
                except Exception as exc:
                    raise RecurrenceTemplateError(
                        "cannot evaluate model quantum-flow callback for "
                        f"vertex kind {vertex.kind}: {exc}"
                    ) from exc
                rows: list[dict[str, Any]] = []
                for flow in flows:
                    if int(flow.chirality) != binding.result_state.chirality:
                        continue
                    if not isinstance(flow.spin_state, int) or isinstance(
                        flow.spin_state, bool
                    ):
                        raise RecurrenceTemplateError(
                            "recurrence-template-v1 cannot encode structured "
                            f"quantum-flow spin state {flow.spin_state!r}"
                        )
                    rows.append(
                        {
                            "input_spin_states": (left_spin, right_spin),
                            "input_flavour_flows": (
                                _canonical_json(list(left.flavour_flow)),
                                _canonical_json(list(right.flavour_flow)),
                            ),
                            "input_quantum_number_flows": (
                                _canonical_json(list(left.quantum_number_flow)),
                                _canonical_json(list(right.quantum_number_flow)),
                            ),
                            "result_chirality": int(flow.chirality),
                            "result_spin_state": int(flow.spin_state),
                            "result_flavour_flow": _canonical_json(
                                list(flow.flavour_flow)
                            ),
                            "result_quantum_number_flow": _canonical_json(
                                list(flow.quantum_number_flow)
                            ),
                            "coupling": _exact_pair(
                                flow.coupling,
                                f"vertex kind {vertex.kind} quantum-flow coupling",
                            ).to_dict(),
                        }
                    )
                return tuple(sorted(rows, key=_canonical_json))

            first = evaluate()
            second = evaluate()
            if _canonical_json(first) != _canonical_json(second):
                raise RecurrenceTemplateError(
                    "allowed_quantum_flows is nondeterministic for "
                    f"vertex kind {vertex.kind}"
                )
            for row in first:
                variants.setdefault(_canonical_json(row), row)
    return tuple(variants[key] for key in sorted(variants))


def _spin_states_for_chirality(
    model: Model,
    particle_id: int,
    chirality: int,
) -> tuple[int, ...]:
    if _effective_auxiliary_kind(model, particle_id, chirality) is not None:
        result = model.result_spin_state(particle_id, chirality)
        if not isinstance(result, int) or isinstance(result, bool):
            raise RecurrenceTemplateError(
                "recurrence-template-v1 cannot encode structured auxiliary "
                "result spin state"
            )
        return (int(result),)
    values: set[int] = set()
    for state in model.source_spin_states(particle_id):
        if int(state.chirality) != chirality:
            continue
        if not isinstance(state.spin_state, int) or isinstance(state.spin_state, bool):
            raise RecurrenceTemplateError(
                "recurrence-template-v1 cannot encode structured source spin state "
                f"{state.spin_state!r}"
            )
        values.add(int(state.spin_state))
    if not values:
        result = model.result_spin_state(particle_id, chirality)
        if not isinstance(result, int) or isinstance(result, bool):
            raise RecurrenceTemplateError(
                "recurrence-template-v1 cannot encode structured result spin state"
            )
        values.add(int(result))
    return tuple(sorted(values))


def _flow_probe_index(
    model: Model,
    state: PreparedParticleState,
    spin_state: int,
) -> SimpleNamespace:
    quantum_flow = _stable_callback(
        f"quantum-number flow for particle {state.particle_id}",
        lambda: model.quantum_number_flow(state.particle_id),
        serializer=lambda value: _canonical_json(list(value)),
    )
    return SimpleNamespace(
        particle_id=state.particle_id,
        pdg=state.particle_id,
        chirality=state.chirality,
        spin_state=spin_state,
        flavour_flow=(state.particle_id,),
        quantum_number_flow=quantum_flow,
        coupling_orders=(),
    )


def _canonical_coupling_orders(
    value: Sequence[tuple[str, int]],
    context: str,
) -> tuple[tuple[str, int], ...]:
    result = tuple(sorted((str(name).upper(), int(power)) for name, power in value))
    if len({name for name, _ in result}) != len(result) or any(
        power < 0 for _, power in result
    ):
        raise RecurrenceTemplateError(f"{context} is not canonical")
    return result


def _color_template(
    model: Model,
    vertex: Vertex,
    *,
    closure: bool,
) -> ColorContractionTemplateV1:
    def evaluate() -> tuple[str, tuple[int, ...], ExactComplexRationalV1]:
        structure = str(model.vertex_color_structure(vertex))
        representations = tuple(
            int(model.color_rep(particle)) for particle in vertex.particles
        )
        if structure in {"model-defined", "generic-tensor"}:
            structure = _infer_model_defined_color_rule(representations)
        if structure not in _SUPPORTED_COLOR_RULES:
            raise RecurrenceTemplateError(
                f"recurrence-template-v1 cannot encode color rule {structure!r} "
                f"for vertex kind {vertex.kind}"
            )
        coefficient = _exact_pair(
            model.vertex_color_weight(vertex, color_accuracy="lc"),
            f"vertex kind {vertex.kind} LC color coefficient",
        )
        return structure, representations, coefficient

    first = evaluate()
    second = evaluate()
    if first != second:
        raise RecurrenceTemplateError(
            f"color callback is nondeterministic for vertex kind {vertex.kind}"
        )
    structure, representations, coefficient = first
    output_representation = None if closure else representations[2]
    payload = {
        "rule_kind": structure,
        "input_representations": representations[:2],
        "output_representation": output_representation,
        "coefficient": coefficient.to_dict(),
    }
    return ColorContractionTemplateV1(
        template_id=_token("color", payload),
        rule_kind=structure,
        input_representations=representations[:2],
        output_representation=output_representation,
        ordered_open_string_arity=(
            1 if any(abs(value) == 3 for value in representations) else 0
        ),
        exact_coefficient=coefficient,
        nc_polynomial=((0, ExactComplexRationalV1.one()),),
        expression_digest=_digest(payload),
    )


def _infer_model_defined_color_rule(representations: tuple[int, ...]) -> str:
    absolute = tuple(abs(value) for value in representations)
    if all(value == 1 for value in absolute):
        return "singlet"
    if sorted(absolute) == [3, 3, 8]:
        return "fundamental-generator"
    if sorted(absolute) == [1, 3, 3]:
        return "color-identity"
    if absolute == (8, 8, 8):
        return "adjoint-structure-constant"
    raise RecurrenceTemplateError(
        "model-defined color semantics are not reconstructible from generic "
        f"representations {representations!r}; the prepared recurrence companion "
        "must provide an explicit certified color rule"
    )


def _build_propagators(
    model: Model,
    bindings: Sequence[PreparedPropagatorBinding],
    state_ids: Mapping[tuple[int, int], str],
    parameter_ids: Mapping[str, str],
    kernels: Mapping[int, Any],
    evaluator_requests: list[_EvaluatorRequest],
) -> tuple[tuple[PropagatorTemplateV1, ...], tuple[SymmetryProofV1, ...]]:
    records: list[PropagatorTemplateV1] = []
    proofs: list[SymmetryProofV1] = []
    for binding in sorted(bindings, key=lambda item: item.key):
        metadata = _stable_callback(
            f"propagator contract for {binding.key!r}",
            lambda binding=binding: model._propagator_ir(
                binding.key.particle_id, binding.key.chirality
            ),
            serializer=lambda value: _canonical_json(value.to_json_dict()),
        )
        if (
            bool(metadata.applies_propagator) != binding.applies_propagator
            or str(metadata.kind) != binding.propagator_kind
            or str(metadata.mass_class) != binding.mass_class
            or metadata.gauge != binding.gauge
            or metadata.identity.canonical_id != binding.state.identity
            or metadata.basis != binding.state.basis
        ):
            raise PreparedKernelCatalogError(
                f"prepared propagator binding {binding.key!r} is stale"
            )
        state_id = state_ids[(binding.state.particle_id, binding.state.chirality)]
        template_id = _token(
            "propagator",
            {
                "state": state_id,
                "metadata": metadata.to_json_dict(),
                "kernel_id": binding.kernel_id,
            },
        )
        kernel = None
        resolver_key = None
        numerator_digest = None
        denominator_digest = None
        proof_id = None
        if binding.applies_propagator:
            if binding.kernel_id is None:
                raise PreparedKernelCatalogError(
                    f"active propagator {binding.key!r} has no prepared kernel"
                )
            kernel = _kernel(kernels, binding.kernel_id, "propagator")
            if kernel.contract_kind != "propagator":
                raise PreparedKernelCatalogError(
                    f"prepared propagator binding references a "
                    f"{kernel.contract_kind!r} kernel"
                )
            _validate_propagator_kernel_layout(kernel, binding)
            if metadata.numerator is None or metadata.denominator is None:
                raise RecurrenceTemplateError(
                    f"active propagator {binding.key!r} lacks exact formula metadata"
                )
            numerator_digest = _expression_digest(str(metadata.numerator))
            denominator_digest = _expression_digest(str(metadata.denominator))
            resolver_key = _resolver_key(kernel, "propagator", (state_id,), state_id)
            if PREPARED_HOMOGENEOUS_LINEAR_CURRENT_PROOF in kernel.proof_classes:
                proof_id = _token(
                    "proof",
                    {
                        "algorithm": PREPARED_HOMOGENEOUS_LINEAR_CURRENT_PROOF,
                        "subject": template_id,
                        "kernel": kernel.canonical_signature,
                    },
                )
                expression_digests = tuple(
                    _expression_digest(value) for value in kernel.exact_expressions
                )
                proofs.append(
                    SymmetryProofV1(
                        template_id=proof_id,
                        proof_algorithm=PREPARED_HOMOGENEOUS_LINEAR_CURRENT_PROOF,
                        subject_template_ids=(template_id,),
                        input_permutation=(0,),
                        exact_phase=ExactComplexRationalV1.one(),
                        expression_digests=expression_digests,
                        witness_digest=_digest(
                            {
                                "kernel": kernel.canonical_signature,
                                "proof_class": (
                                    PREPARED_HOMOGENEOUS_LINEAR_CURRENT_PROOF
                                ),
                            }
                        ),
                    )
                )
        record = PropagatorTemplateV1(
            template_id=template_id,
            state_template_id=state_id,
            applies_propagator=binding.applies_propagator,
            evaluator_resolver_key=resolver_key,
            numerator_expression_digest=numerator_digest,
            denominator_expression_digest=denominator_digest,
            mass_parameter_id=_parameter_reference(
                metadata.mass_parameter,
                parameter_ids,
                "propagator mass parameter",
            ),
            width_parameter_id=_parameter_reference(
                metadata.width_parameter,
                parameter_ids,
                "propagator width parameter",
            ),
            gauge=metadata.gauge,
            linearity_proof_template_id=proof_id,
        )
        records.append(record)
        if kernel is not None:
            evaluator_requests.append(
                _EvaluatorRequest(
                    kernel=kernel,
                    contract_kind="propagator",
                    input_state_template_ids=(state_id,),
                    output_state_template_id=state_id,
                    semantic_template_id=record.template_id,
                )
            )
    return tuple(records), tuple(proofs)


def _build_closures(
    model: Model,
    bindings: Sequence[PreparedClosureBinding],
    state_ids: Mapping[tuple[int, int], str],
    parameter_ids: Mapping[str, str],
    kernels: Mapping[int, Any],
    evaluator_requests: list[_EvaluatorRequest],
) -> tuple[
    tuple[ClosureTemplateV1, ...],
    tuple[ColorContractionTemplateV1, ...],
]:
    records: list[ClosureTemplateV1] = []
    colors: dict[str, ColorContractionTemplateV1] = {}
    for binding in sorted(bindings, key=lambda item: item.key):
        vertex = Vertex(
            binding.key.kind,
            binding.key.particles,
            binding.key.coupling,
        )
        kernel = _kernel(kernels, binding.kernel_id, "closure")
        if kernel.contract_kind != "closure":
            raise PreparedKernelCatalogError(
                f"prepared closure binding references a {kernel.contract_kind!r} kernel"
            )
        _validate_binary_kernel_layout(
            kernel,
            binding,
            output_dimension=1,
            context=f"closure kind {vertex.kind}",
        )
        contraction = model.closure_contraction_ir(
            binding.result_state.particle_id,
            binding.result_state.chirality,
        )
        if contraction is None or contraction.name != binding.projection:
            raise PreparedKernelCatalogError(
                f"prepared closure binding for kind {vertex.kind} does not match "
                "the live contraction contract"
            )
        color = _color_template(model, vertex, closure=True)
        colors.setdefault(color.template_id, color)
        input_state_ids = (
            state_ids[(binding.left_state.particle_id, binding.left_state.chirality)],
            state_ids[(binding.right_state.particle_id, binding.right_state.chirality)],
        )
        resolver_key = _resolver_key(kernel, "closure", input_state_ids, None)
        template_id = _token(
            "closure",
            {
                "binding": _closure_binding_payload(binding),
                "kernel": kernel.canonical_signature,
                "color": color.semantic_digest,
                "projection": contraction.to_json_dict(),
            },
        )
        record = ClosureTemplateV1(
            template_id=template_id,
            input_state_template_ids=input_state_ids,
            evaluator_resolver_key=resolver_key,
            canonical_input_order=tuple(binding.canonical_input_order),
            coupling_parameter_ids=_kernel_parameter_ids(kernel, parameter_ids),
            coupling_orders=_canonical_coupling_orders(
                model.vertex_coupling_orders(vertex),
                f"closure kind {vertex.kind} coupling orders",
            ),
            color_contraction_template_id=color.template_id,
            exact_factor=_exact_pair(
                binding.equivalence_factor,
                f"closure kind {vertex.kind} equivalence factor",
            ),
            projection=binding.projection,
        )
        records.append(record)
        evaluator_requests.append(
            _EvaluatorRequest(
                kernel=kernel,
                contract_kind="closure",
                input_state_template_ids=input_state_ids,
                output_state_template_id=None,
                semantic_template_id=record.template_id,
            )
        )
    return tuple(records), tuple(colors.values())


def _build_direct_closures(
    model: Model,
    state_ids: Mapping[tuple[int, int], str],
    state_by_id: Mapping[str, CurrentStateTemplateV1],
) -> tuple[
    tuple[ClosureTemplateV1, ...],
    tuple[ColorContractionTemplateV1, ...],
    tuple[EvaluatorBindingV1, ...],
]:
    """Bind exact model current-pair contractions to Rusticol templates."""

    direct = getattr(model, "_direct_contraction_ir_by_state", {})
    if not isinstance(direct, Mapping):
        raise RecurrenceTemplateError(
            "model direct-contraction inventory is not a mapping"
        )
    records: list[ClosureTemplateV1] = []
    colors: dict[str, ColorContractionTemplateV1] = {}
    evaluator_bindings: list[EvaluatorBindingV1] = []
    for raw_key in sorted(direct):
        if not isinstance(raw_key, tuple) or len(raw_key) != 4:
            raise RecurrenceTemplateError(
                f"direct-contraction key {raw_key!r} is not a four-state tuple"
            )
        key = tuple(int(value) for value in raw_key)
        left_key = key[0], key[1]
        right_key = key[2], key[3]
        if left_key not in state_ids or right_key not in state_ids:
            # Ghost-only and otherwise unadmitted states are intentionally not
            # part of recurrence v1.
            continue
        contraction = _stable_callback(
            f"direct contraction for states {key!r}",
            lambda key=key: model.direct_contraction_ir(
                key[0], key[2], key[1], key[3]
            ),
            serializer=lambda value: (
                "null"
                if value is None
                else _canonical_json(value.to_json_dict())
            ),
        )
        if contraction is None:
            continue
        left_state_id = state_ids[left_key]
        right_state_id = state_ids[right_key]
        left_state = state_by_id[left_state_id]
        right_state = state_by_id[right_state_id]
        if (
            contraction.left_basis != left_state.basis
            or contraction.right_basis != right_state.basis
        ):
            raise RecurrenceTemplateError(
                f"direct contraction {contraction.name!r} basis does not match "
                f"states {key!r}"
            )
        if left_state.dimension != right_state.dimension or len(
            contraction.coefficients
        ) != left_state.dimension:
            raise RecurrenceTemplateError(
                f"direct contraction {contraction.name!r} component count does "
                f"not match states {key!r}"
            )
        if contraction.chirality_relation == "equal" and key[1] != key[3]:
            raise RecurrenceTemplateError(
                f"direct contraction {contraction.name!r} violates equal chirality"
            )
        if contraction.chirality_relation == "opposite" and key[1] != -key[3]:
            raise RecurrenceTemplateError(
                f"direct contraction {contraction.name!r} violates opposite chirality"
            )
        exact_coefficients = tuple(
            _exact_pair(value, f"direct contraction {contraction.name!r} coefficient")
            for value in contraction.coefficients
        )
        callable_payload = {
            "abi": "rusticol-closure-reduce-v1",
            "left_basis": left_state.basis,
            "right_basis": right_state.basis,
            "dimension": left_state.dimension,
            "coefficients": [value.to_dict() for value in exact_coefficients],
            "chirality_relation": contraction.chirality_relation,
            "metric_signature": contraction.metric_signature,
        }
        callable_signature = _digest(callable_payload)
        runtime_template = f"rusticol.closure-reduce.v1:{callable_signature[:24]}"
        input_state_ids = left_state_id, right_state_id
        resolver_key = _token(
            "evaluator",
            {
                "runtime_template": runtime_template,
                "input_states": input_state_ids,
            },
        )
        color = _direct_closure_color_template(model, key[0], key[2])
        colors.setdefault(color.template_id, color)
        closure_payload = {
            "states": input_state_ids,
            "projection": contraction.to_json_dict(),
            "callable": callable_signature,
            "color": color.semantic_digest,
        }
        record = ClosureTemplateV1(
            template_id=_token("closure", closure_payload),
            input_state_template_ids=input_state_ids,
            evaluator_resolver_key=resolver_key,
            canonical_input_order=(0, 1),
            coupling_parameter_ids=(),
            coupling_orders=(),
            color_contraction_template_id=color.template_id,
            exact_factor=ExactComplexRationalV1.one(),
            projection=contraction.name,
            component_coefficients=exact_coefficients,
            chirality_relation=contraction.chirality_relation,
            metric_signature=contraction.metric_signature,
        )
        records.append(record)
        evaluator_bindings.append(
            EvaluatorBindingV1(
                resolver_key=resolver_key,
                prepared_kernel_id=None,
                callable_kind="rusticol-template",
                runtime_template=runtime_template,
                contract_kind="closure",
                callable_signature=callable_signature,
                input_state_template_ids=input_state_ids,
                output_state_template_id=None,
                input_layout=tuple(
                    f"left-current:{component}"
                    for component in range(left_state.dimension)
                )
                + tuple(
                    f"right-current:{component}"
                    for component in range(right_state.dimension)
                ),
                output_layout=("closure:amplitude",),
                exact_expression_digests=(_digest(callable_payload),),
                semantic_template_ids=(record.template_id,),
            )
        )
    return tuple(records), tuple(colors.values()), tuple(evaluator_bindings)


def _direct_closure_color_template(
    model: Model,
    left_particle_id: int,
    right_particle_id: int,
) -> ColorContractionTemplateV1:
    representations = (
        int(model.color_rep(left_particle_id)),
        int(model.color_rep(right_particle_id)),
    )
    if abs(representations[0]) != abs(representations[1]):
        raise RecurrenceTemplateError(
            "direct current contraction joins incompatible color "
            f"representations {representations!r}"
        )
    payload = {
        "rule_kind": "direct-pairing",
        "input_representations": representations,
    }
    return ColorContractionTemplateV1(
        template_id=_token("color", payload),
        rule_kind="direct-pairing",
        input_representations=representations,
        output_representation=None,
        ordered_open_string_arity=(
            1 if abs(representations[0]) == 3 else 0
        ),
        exact_coefficient=ExactComplexRationalV1.one(),
        nc_polynomial=((0, ExactComplexRationalV1.one()),),
        expression_digest=_digest(payload),
    )


def _add_model_parameter_evaluator(
    prepared_catalog: PreparedKernelCatalog,
    parameters: Sequence[ParameterTemplateV1],
    kernels: Mapping[int, Any],
    evaluator_requests: list[_EvaluatorRequest],
) -> None:
    kernel_id = prepared_catalog.model_parameter_kernel_id
    derived = {
        parameter.name: parameter
        for parameter in parameters
        if parameter.parameter_kind == "derived"
    }
    if kernel_id is None:
        if derived:
            raise PreparedKernelCatalogError(
                "derived recurrence parameters have no prepared model-parameter kernel"
            )
        return
    kernel = _kernel(kernels, kernel_id, "model-parameter")
    if kernel.contract_kind != "model-parameter":
        raise PreparedKernelCatalogError(
            "prepared model-parameter binding references the wrong kernel kind"
        )
    output_names = tuple(
        value.removeprefix("model-parameter:") for value in kernel.output_layout
    )
    if set(output_names) != set(derived):
        raise PreparedKernelCatalogError(
            "prepared model-parameter outputs do not match derived parameters"
        )
    for name in output_names:
        evaluator_requests.append(
            _EvaluatorRequest(
                kernel=kernel,
                contract_kind="model-parameter",
                input_state_template_ids=(),
                output_state_template_id=None,
                semantic_template_id=derived[name].template_id,
            )
        )


def _coalesce_evaluator_requests(
    requests: Sequence[_EvaluatorRequest],
) -> tuple[EvaluatorBindingV1, ...]:
    groups: dict[
        tuple[int, str, tuple[str, ...], str | None],
        list[_EvaluatorRequest],
    ] = {}
    for request in requests:
        key = (
            int(request.kernel.kernel_id),
            request.contract_kind,
            request.input_state_template_ids,
            request.output_state_template_id,
        )
        groups.setdefault(key, []).append(request)
    result: list[EvaluatorBindingV1] = []
    for key in sorted(groups, key=_canonical_json):
        group = groups[key]
        first = group[0]
        kernel = first.kernel
        input_layout = _kernel_input_layout(kernel.inputs)
        result.append(
            EvaluatorBindingV1(
                resolver_key=_resolver_key(
                    kernel,
                    first.contract_kind,
                    first.input_state_template_ids,
                    first.output_state_template_id,
                ),
                prepared_kernel_id=int(kernel.kernel_id),
                contract_kind=first.contract_kind,  # type: ignore[arg-type]
                callable_signature=kernel.canonical_signature,
                input_state_template_ids=first.input_state_template_ids,
                output_state_template_id=first.output_state_template_id,
                input_layout=input_layout,
                output_layout=tuple(str(value) for value in kernel.output_layout),
                exact_expression_digests=tuple(
                    _expression_digest(value) for value in kernel.exact_expressions
                ),
                semantic_template_ids=tuple(
                    sorted({item.semantic_template_id for item in group})
                ),
            )
        )
    return tuple(result)


def _kernel_input_layout(inputs: Sequence[PreparedKernelInput]) -> tuple[str, ...]:
    return tuple(
        _canonical_json(
            {
                "role": item.role,
                "component": item.component,
                "symbol": item.symbol,
                "model_parameter_name": item.model_parameter_name,
                "model_parameter_index": item.model_parameter_index,
            }
        )
        for item in inputs
    )


def _kernel_parameter_ids(
    kernel: Any,
    parameter_ids: Mapping[str, str],
) -> tuple[str, ...]:
    names = tuple(
        sorted(
            {
                str(item.model_parameter_name)
                for item in kernel.inputs
                if item.model_parameter_name is not None
            }
        )
    )
    missing = tuple(name for name in names if name not in parameter_ids)
    if missing:
        raise RecurrenceTemplateError(
            "prepared kernel references absent recurrence parameters: "
            + ", ".join(missing)
        )
    return tuple(sorted(parameter_ids[name] for name in names))


def _kernel(
    kernels: Mapping[int, Any],
    kernel_id: int,
    context: str,
) -> Any:
    try:
        return kernels[int(kernel_id)]
    except KeyError as exc:
        raise PreparedKernelCatalogError(
            f"prepared {context} binding references unknown kernel {kernel_id}"
        ) from exc


def _validate_binary_kernel_layout(
    kernel: Any,
    binding: PreparedVertexBinding | PreparedClosureBinding,
    *,
    output_dimension: int,
    context: str,
) -> None:
    canonical_states = tuple(
        (binding.left_state, binding.right_state)[index]
        for index in binding.canonical_input_order
    )
    expected = {
        "left-current": canonical_states[0].dimension,
        "right-current": canonical_states[1].dimension,
    }
    for role, dimension in expected.items():
        components = tuple(
            item.component for item in kernel.inputs if item.role == role
        )
        if len(set(components)) != len(components) or any(
            component < 0 or component >= dimension for component in components
        ):
            raise PreparedKernelCatalogError(
                f"prepared {context} {role} layout does not match its state dimension"
            )
    if len(kernel.output_layout) != output_dimension:
        raise PreparedKernelCatalogError(
            f"prepared {context} output layout has dimension "
            f"{len(kernel.output_layout)}, expected {output_dimension}"
        )


def _validate_propagator_kernel_layout(
    kernel: Any,
    binding: PreparedPropagatorBinding,
) -> None:
    components = tuple(
        item.component for item in kernel.inputs if item.role == "current"
    )
    if len(set(components)) != len(components) or any(
        component < 0 or component >= binding.state.dimension
        for component in components
    ):
        raise PreparedKernelCatalogError(
            f"prepared propagator {binding.key!r} current layout does not match "
            "its state dimension"
        )
    if len(kernel.output_layout) != binding.state.dimension:
        raise PreparedKernelCatalogError(
            f"prepared propagator {binding.key!r} output dimension is stale"
        )


def _resolver_key(
    kernel: Any,
    contract_kind: str,
    input_states: tuple[str, ...],
    output_state: str | None,
) -> str:
    return _token(
        "evaluator",
        {
            "kernel_id": int(kernel.kernel_id),
            "signature": kernel.canonical_signature,
            "contract_kind": contract_kind,
            "input_states": input_states,
            "output_state": output_state,
        },
    )


def _vertex_binding_payload(binding: PreparedVertexBinding) -> dict[str, object]:
    return {
        "kind": binding.key.kind,
        "particles": list(binding.key.particles),
        "chiralities": [
            binding.key.left_chirality,
            binding.key.right_chirality,
            binding.key.result_chirality,
        ],
        "coupling": _exact_pair(binding.key.coupling, "vertex coupling").to_dict(),
        "canonical_input_order": list(binding.canonical_input_order),
        "equivalence_factor": _exact_pair(
            binding.equivalence_factor, "vertex equivalence factor"
        ).to_dict(),
        "output_factor_source": binding.output_factor_source,
    }


def _closure_binding_payload(binding: PreparedClosureBinding) -> dict[str, object]:
    return {
        "kind": binding.key.kind,
        "particles": list(binding.key.particles),
        "chiralities": [binding.key.left_chirality, binding.key.right_chirality],
        "coupling": _exact_pair(binding.key.coupling, "closure coupling").to_dict(),
        "canonical_input_order": list(binding.canonical_input_order),
        "equivalence_factor": _exact_pair(
            binding.equivalence_factor, "closure equivalence factor"
        ).to_dict(),
        "projection": binding.projection,
        "output_factor_source": binding.output_factor_source,
    }


def _stable_callback(
    context: str,
    callback: Any,
    *,
    serializer: Any,
) -> Any:
    try:
        first = callback()
        first_payload = serializer(first)
        second = callback()
        second_payload = serializer(second)
    except (PreparedKernelCatalogError, RecurrenceTemplateError):
        raise
    except Exception as exc:
        raise RecurrenceTemplateError(f"cannot project {context}: {exc}") from exc
    if first_payload != second_payload:
        raise RecurrenceTemplateError(f"{context} is nondeterministic")
    return first


__all__ = ["build_recurrence_template_catalog"]
