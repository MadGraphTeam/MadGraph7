# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import pytest

from pyamplicol.models.recurrence_template import (
    ClosureTemplateV1,
    ColorContractionTemplateV1,
    CurrentStateTemplateV1,
    EvaluatorBindingV1,
    EvaluatorCallableKind,
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

_COMPILED_MODEL_DIGEST = "a" * 64
_EXPRESSION_A = "1" * 64
_EXPRESSION_B = "2" * 64
_EXPRESSION_C = "3" * 64
_PREDICATE = "4" * 64
_WITNESS = "5" * 64
_CALLABLE_A = "6" * 64
_CALLABLE_B = "7" * 64
_CALLABLE_C = "8" * 64
_CALLABLE_D = "9" * 64
_CALLABLE_E = "b" * 64


def _parameter_templates() -> tuple[ParameterTemplateV1, ...]:
    return (
        ParameterTemplateV1(
            template_id="parameter:coupling",
            name="model.coupling",
            parameter_kind="external",
            value_type="complex",
            mutable=True,
            default_value=ExactComplexRationalV1.from_fractions(
                Fraction(1, 3), Fraction(-1, 7)
            ),
            exact_expression_digest=None,
            dependency_parameter_ids=(),
        ),
        ParameterTemplateV1(
            template_id="parameter:mass",
            name="particle.mass",
            parameter_kind="external",
            value_type="real",
            mutable=True,
            default_value=ExactComplexRationalV1.from_binary64(172.5),
            exact_expression_digest=None,
            dependency_parameter_ids=(),
        ),
    )


def _state_templates() -> tuple[CurrentStateTemplateV1, ...]:
    return (
        CurrentStateTemplateV1(
            template_id="state:adjoint",
            particle_id=9000021,
            anti_particle_id=9000021,
            species_id="example:species:adjoint",
            orientation="self-conjugate",
            statistics="boson",
            color_representation=8,
            basis="lorentz-vector",
            tensor_ordering=("mu0", "mu1", "mu2", "mu3"),
            dimension=4,
            chirality=0,
            auxiliary_kind=None,
            mass_parameter_id=None,
            width_parameter_id=None,
        ),
        CurrentStateTemplateV1(
            template_id="state:matter",
            particle_id=9000001,
            anti_particle_id=-9000001,
            species_id="example:species:matter",
            orientation="particle",
            statistics="fermion",
            color_representation=3,
            basis="weyl-chiral",
            tensor_ordering=("spin0", "spin1"),
            dimension=2,
            chirality=1,
            auxiliary_kind=None,
            mass_parameter_id="parameter:mass",
            width_parameter_id=None,
        ),
    )


def _color_template() -> ColorContractionTemplateV1:
    return ColorContractionTemplateV1(
        template_id="color:matter-adjoint",
        rule_kind="ordered-open-string-append",
        input_representations=(3, 8),
        output_representation=3,
        ordered_open_string_arity=1,
        exact_coefficient=ExactComplexRationalV1.one(),
        nc_polynomial=((0, ExactComplexRationalV1.one()),),
        expression_digest=_EXPRESSION_A,
    )


def _proof() -> SymmetryProofV1:
    return SymmetryProofV1(
        template_id="proof:matter-linearity",
        proof_algorithm="prepared-kernel-homogeneous-complex-linear-current-v1",
        subject_template_ids=("state:matter",),
        input_permutation=(0,),
        exact_phase=ExactComplexRationalV1.one(),
        expression_digests=(_EXPRESSION_B,),
        witness_digest=_WITNESS,
    )


def _source_templates() -> tuple[SourceTemplateV1, ...]:
    return (
        SourceTemplateV1(
            template_id="source:adjoint:+1",
            state_template_id="state:adjoint",
            crossing="identity",
            wavefunction_family="vector",
            helicity=1,
            spin_state=1,
            wavefunction_expression_digest=_EXPRESSION_B,
            evaluator_resolver_key="evaluator:source:adjoint",
        ),
        SourceTemplateV1(
            template_id="source:matter:+1",
            state_template_id="state:matter",
            crossing="identity",
            wavefunction_family="fermion",
            helicity=1,
            spin_state=1,
            wavefunction_expression_digest=_EXPRESSION_A,
            evaluator_resolver_key="evaluator:source:matter",
            mass_parameter_id="parameter:mass",
        ),
    )


def _flow_template() -> QuantumFlowTemplateV1:
    return QuantumFlowTemplateV1(
        template_id="flow:matter-adjoint-to-matter",
        input_state_template_ids=("state:matter", "state:adjoint"),
        input_spin_states=(1, 1),
        input_flavour_flows=("matter", "neutral"),
        input_quantum_number_flows=("fundamental", "adjoint"),
        coupling_orders=(("QCD", 1),),
        result_state_template_id="state:matter",
        result_flavour_flow="matter",
        result_quantum_number_flow="fundamental",
        predicate_digest=_PREDICATE,
    )


def _transition() -> TransitionTemplateV1:
    return TransitionTemplateV1(
        template_id="transition:matter-adjoint-to-matter",
        input_state_template_ids=("state:matter", "state:adjoint"),
        result_state_template_id="state:matter",
        quantum_flow_template_id="flow:matter-adjoint-to-matter",
        evaluator_resolver_key="evaluator:vertex:matter-adjoint",
        canonical_input_order=(0, 1),
        momentum_convention=("incoming-left", "incoming-right"),
        coupling_parameter_ids=("parameter:coupling",),
        coupling_orders=(("QCD", 1),),
        color_contraction_template_id="color:matter-adjoint",
        exact_factor=ExactComplexRationalV1.one(),
        output_projection="weyl-chiral:+1",
    )


def _propagator() -> PropagatorTemplateV1:
    return PropagatorTemplateV1(
        template_id="propagator:matter",
        state_template_id="state:matter",
        applies_propagator=True,
        evaluator_resolver_key="evaluator:propagator:matter",
        numerator_expression_digest=_EXPRESSION_B,
        denominator_expression_digest=_EXPRESSION_C,
        mass_parameter_id="parameter:mass",
        width_parameter_id=None,
        gauge=None,
        linearity_proof_template_id="proof:matter-linearity",
    )


def _closure() -> ClosureTemplateV1:
    return ClosureTemplateV1(
        template_id="closure:matter-pair",
        input_state_template_ids=("state:matter", "state:matter"),
        evaluator_resolver_key="evaluator:closure:matter-pair",
        canonical_input_order=(0, 1),
        coupling_parameter_ids=(),
        coupling_orders=(),
        color_contraction_template_id="color:matter-adjoint",
        exact_factor=ExactComplexRationalV1.one(),
        projection="scalar",
    )


def _state_input_layout(*state_ids: str) -> tuple[str, ...]:
    dimensions = {"state:adjoint": 4, "state:matter": 2}
    return tuple(
        f"state:{state_id}:{component}"
        for state_id in state_ids
        for component in range(dimensions[state_id])
    )


def _evaluator_bindings() -> tuple[EvaluatorBindingV1, ...]:
    return (
        EvaluatorBindingV1(
            resolver_key="evaluator:closure:matter-pair",
            prepared_kernel_id=4,
            contract_kind="closure",
            callable_signature=_CALLABLE_E,
            input_state_template_ids=("state:matter", "state:matter"),
            output_state_template_id=None,
            input_layout=_state_input_layout("state:matter", "state:matter"),
            output_layout=("scalar",),
            exact_expression_digests=(_EXPRESSION_C,),
            semantic_template_ids=("closure:matter-pair",),
        ),
        EvaluatorBindingV1(
            resolver_key="evaluator:propagator:matter",
            prepared_kernel_id=3,
            contract_kind="propagator",
            callable_signature=_CALLABLE_D,
            input_state_template_ids=("state:matter",),
            output_state_template_id="state:matter",
            input_layout=_state_input_layout("state:matter"),
            output_layout=("spin0", "spin1"),
            exact_expression_digests=(_EXPRESSION_A, _EXPRESSION_B),
            semantic_template_ids=("propagator:matter",),
        ),
        EvaluatorBindingV1(
            resolver_key="evaluator:source:adjoint",
            prepared_kernel_id=1,
            contract_kind="source",
            callable_signature=_CALLABLE_B,
            input_state_template_ids=(),
            output_state_template_id="state:adjoint",
            input_layout=("momentum:E",),
            output_layout=("mu0", "mu1", "mu2", "mu3"),
            exact_expression_digests=(
                _EXPRESSION_A,
                _EXPRESSION_A,
                _EXPRESSION_A,
                _EXPRESSION_A,
            ),
            semantic_template_ids=("source:adjoint:+1",),
        ),
        EvaluatorBindingV1(
            resolver_key="evaluator:source:matter",
            prepared_kernel_id=0,
            contract_kind="source",
            callable_signature=_CALLABLE_A,
            input_state_template_ids=(),
            output_state_template_id="state:matter",
            input_layout=("momentum:E", "parameter:mass"),
            output_layout=("spin0", "spin1"),
            exact_expression_digests=(_EXPRESSION_A, _EXPRESSION_B),
            semantic_template_ids=("source:matter:+1",),
        ),
        EvaluatorBindingV1(
            resolver_key="evaluator:vertex:matter-adjoint",
            prepared_kernel_id=2,
            contract_kind="vertex",
            callable_signature=_CALLABLE_C,
            input_state_template_ids=("state:matter", "state:adjoint"),
            output_state_template_id="state:matter",
            input_layout=(
                *_state_input_layout("state:matter", "state:adjoint"),
                "parameter:coupling:real",
                "parameter:coupling:imag",
            ),
            output_layout=("spin0", "spin1"),
            exact_expression_digests=(_EXPRESSION_B, _EXPRESSION_C),
            semantic_template_ids=("transition:matter-adjoint-to-matter",),
        ),
    )


def _catalog(**overrides: object) -> RecurrenceTemplateCatalog:
    values: dict[str, object] = {
        "compiled_model_digest": _COMPILED_MODEL_DIGEST,
        "parameters": _parameter_templates(),
        "current_states": _state_templates(),
        "sources": _source_templates(),
        "quantum_flows": (_flow_template(),),
        "transitions": (_transition(),),
        "propagators": (_propagator(),),
        "closures": (_closure(),),
        "color_contractions": (_color_template(),),
        "symmetry_proofs": (_proof(),),
        "evaluator_bindings": _evaluator_bindings(),
    }
    values.update(overrides)
    return RecurrenceTemplateCatalog.create(**values)  # type: ignore[arg-type]


def test_binary64_conversion_is_exact_and_distinguishes_sub_ulp_terms() -> None:
    tenth = ExactComplexRationalV1.from_binary64(0.1)
    assert tenth.real == Fraction(3602879701896397, 36028797018963968)
    assert tenth.imag == 0

    one = ExactComplexRationalV1.one()
    one_plus_sub_ulp = ExactComplexRationalV1.from_fractions(
        Fraction(1, 1) + Fraction(1, 2**54)
    )
    assert one != one_plus_sub_ulp
    assert one_plus_sub_ulp.real == Fraction(2**54 + 1, 2**54)


@pytest.mark.parametrize(
    "arguments",
    [
        (2, 4, 0, 1),
        (0, 2, 0, 1),
        (1, -2, 0, 1),
        (1, 2, 0, 0),
    ],
)
def test_exact_rational_rejects_noncanonical_fractions(
    arguments: tuple[int, int, int, int],
) -> None:
    with pytest.raises(RecurrenceTemplateError):
        ExactComplexRationalV1(*arguments)


def test_exact_rational_json_requires_canonical_decimal_integers() -> None:
    payload = ExactComplexRationalV1.one().to_dict()
    payload["real_numerator"] = "+1"
    with pytest.raises(RecurrenceTemplateError, match=r"canonically|decimal"):
        ExactComplexRationalV1.from_dict(payload)


def test_catalog_round_trip_is_canonical_and_content_addressed() -> None:
    catalog = _catalog()
    loaded = RecurrenceTemplateCatalog.from_dict(json.loads(catalog.canonical_json))

    assert loaded == catalog
    assert loaded.canonical_json == catalog.canonical_json
    assert loaded.catalog_digest == catalog.catalog_digest
    assert catalog.current_states[0].template_id == "state:adjoint"
    with pytest.raises(FrozenInstanceError):
        catalog.current_states[0].dimension = 3  # type: ignore[misc]


def test_runtime_template_evaluator_binding_round_trips() -> None:
    bindings = list(_evaluator_bindings())
    source_index = next(
        index
        for index, binding in enumerate(bindings)
        if binding.contract_kind == "source"
    )
    bindings[source_index] = replace(
        bindings[source_index],
        prepared_kernel_id=None,
        callable_kind="rusticol-template",
        runtime_template="rusticol.source-fill.vector.v1",
        semantic_digest="",
    )

    catalog = _catalog(evaluator_bindings=tuple(bindings))
    restored = RecurrenceTemplateCatalog.from_dict(catalog.to_dict())
    binding = restored.evaluator_bindings[source_index]
    assert binding.callable_kind == "rusticol-template"
    assert binding.runtime_template == "rusticol.source-fill.vector.v1"
    assert binding.prepared_kernel_id is None


@pytest.mark.parametrize(
    ("prepared_kernel_id", "callable_kind", "runtime_template"),
    [
        (None, "prepared-kernel", None),
        (1, "rusticol-template", "rusticol.source-fill.vector.v1"),
        (None, "rusticol-template", None),
        (1, "prepared-kernel", "rusticol.source-fill.vector.v1"),
    ],
)
def test_evaluator_binding_requires_exactly_one_callable_lane(
    prepared_kernel_id: int | None,
    callable_kind: EvaluatorCallableKind,
    runtime_template: str | None,
) -> None:
    binding = _evaluator_bindings()[0]
    with pytest.raises(RecurrenceTemplateError):
        replace(
            binding,
            prepared_kernel_id=prepared_kernel_id,
            callable_kind=callable_kind,
            runtime_template=runtime_template,
            semantic_digest="",
        )


def test_catalog_creation_sorts_records_deterministically() -> None:
    forward = _catalog()
    reverse = _catalog(
        parameters=tuple(reversed(_parameter_templates())),
        current_states=tuple(reversed(_state_templates())),
        sources=tuple(reversed(_source_templates())),
        evaluator_bindings=tuple(reversed(_evaluator_bindings())),
    )
    assert reverse.canonical_json == forward.canonical_json
    assert reverse.catalog_digest == forward.catalog_digest


def test_record_rejects_stale_semantic_digest() -> None:
    source = _source_templates()[0]
    with pytest.raises(RecurrenceTemplateError, match="stale semantic digest"):
        replace(source, helicity=-1)


def test_catalog_rejects_stale_catalog_digest() -> None:
    catalog = _catalog()
    stale_header = replace(catalog.header, catalog_digest="f" * 64)
    with pytest.raises(RecurrenceTemplateError, match=r"stale.*catalog digest"):
        replace(catalog, header=stale_header)


def test_catalog_rejects_duplicate_semantic_identity() -> None:
    first = _parameter_templates()[0]
    duplicate = replace(
        first,
        name="another.parameter",
        semantic_digest="",
    )
    with pytest.raises(RecurrenceTemplateError, match="duplicate semantic identity"):
        _catalog(parameters=(first, duplicate, _parameter_templates()[1]))


def test_catalog_rejects_duplicate_evaluator_resolver_key() -> None:
    first = _evaluator_bindings()[0]
    duplicate = replace(
        first,
        prepared_kernel_id=99,
        callable_signature="c" * 64,
        semantic_digest="",
    )
    with pytest.raises(RecurrenceTemplateError, match="resolver keys must be unique"):
        _catalog(evaluator_bindings=(*_evaluator_bindings(), duplicate))


def test_unknown_proof_algorithm_fails_closed() -> None:
    with pytest.raises(RecurrenceTemplateError, match=r"unsupported.*algorithm"):
        replace(
            _proof(),
            proof_algorithm="trust-the-model-name-v1",
            semantic_digest="",
        )


def test_catalog_rejects_unknown_state_reference() -> None:
    transition = replace(
        _transition(),
        result_state_template_id="state:missing",
        semantic_digest="",
    )
    with pytest.raises(RecurrenceTemplateError, match="unknown 'state:missing'"):
        _catalog(transitions=(transition,))


def test_catalog_rejects_malformed_evaluator_state_contract() -> None:
    bindings = list(_evaluator_bindings())
    index = next(
        index
        for index, binding in enumerate(bindings)
        if binding.resolver_key == "evaluator:vertex:matter-adjoint"
    )
    bindings[index] = replace(
        bindings[index],
        input_state_template_ids=("state:adjoint", "state:matter"),
        semantic_digest="",
    )
    with pytest.raises(RecurrenceTemplateError, match="input states do not match"):
        _catalog(evaluator_bindings=tuple(bindings))


def test_catalog_rejects_malformed_evaluator_output_dimension() -> None:
    bindings = list(_evaluator_bindings())
    index = next(
        index
        for index, binding in enumerate(bindings)
        if binding.resolver_key == "evaluator:source:matter"
    )
    bindings[index] = replace(
        bindings[index],
        output_layout=("spin0",),
        exact_expression_digests=(_EXPRESSION_A,),
        semantic_digest="",
    )
    with pytest.raises(RecurrenceTemplateError, match="output layout"):
        _catalog(evaluator_bindings=tuple(bindings))


def test_catalog_loader_rejects_unknown_fields() -> None:
    payload = _catalog().to_dict()
    payload["model_name"] = "forbidden-model-specific-dispatch"
    with pytest.raises(RecurrenceTemplateError, match="unknown=model_name"):
        RecurrenceTemplateCatalog.from_dict(payload)


def test_catalog_loader_rejects_stale_nested_record_digest() -> None:
    payload = _catalog().to_dict()
    payload["sources"][0]["helicity"] = -1  # type: ignore[index]
    with pytest.raises(RecurrenceTemplateError, match="stale semantic digest"):
        RecurrenceTemplateCatalog.from_dict(payload)


def test_contract_is_model_generic() -> None:
    catalog = _catalog()
    serialized = catalog.canonical_json
    assert "built-in-sm" not in serialized
    assert "ufo" not in serialized.lower()
    assert "example:species:matter" in serialized
