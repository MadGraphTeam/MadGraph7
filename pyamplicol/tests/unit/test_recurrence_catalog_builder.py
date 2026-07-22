# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pyamplicol.models._physics_ir import ContractionIR
from pyamplicol.models.base import (
    Model,
    Particle,
    PropagatorLoweringRule,
    QuantumFlow,
    Vertex,
    VertexEvaluationEquivalence,
)
from pyamplicol.models.prepared_catalog import (
    PreparedKernelCatalog,
    PreparedKernelCatalogError,
    PreparedKernelInput,
    PreparedKernelSpec,
    PreparedParticleState,
    PreparedPropagatorBinding,
    PreparedVertexBinding,
    PropagatorKernelKey,
    VertexKernelKey,
)
from pyamplicol.models.recurrence_catalog_builder import (
    build_recurrence_template_catalog,
)
from pyamplicol.models.recurrence_template import (
    ExactComplexRationalV1,
    RecurrenceTemplateCatalog,
    RecurrenceTemplateError,
)

_MODEL_DIGEST = "a" * 64
_PREPARED_ABI = "pyamplicol-prepared-kernel-catalog-v1"


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _signature(
    contract_kind: str,
    inputs: tuple[PreparedKernelInput, ...],
    expressions: tuple[str, ...],
    output_layout: tuple[str, ...],
) -> str:
    payload = {
        "abi": _PREPARED_ABI,
        "contract_kind": contract_kind,
        "inputs": [item.to_dict() for item in inputs],
        "outputs": list(expressions),
        "output_layout": list(output_layout),
    }
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


class _ParameterModel(Model):
    def __init__(self) -> None:
        super().__init__(name="generic-parameter-model")

    def runtime_parameter_defaults(self):
        return {"alpha": (0.1, 0.0)}

    def runtime_parameter_type(self, name):
        assert name == "alpha"
        return "real"

    def runtime_derived_parameter_definitions(self):
        return {"beta": "2*alpha"}

    def runtime_derived_parameter_defaults(self):
        return {"beta": complex(0.2, 0.0)}

    def runtime_normalization_parameter_defaults(self):
        return {"normalization.scale": 1.0}


def _parameter_catalog(*, parameter_index: int = 0) -> PreparedKernelCatalog:
    inputs = (
        PreparedKernelInput(
            role="model-parameter",
            component=0,
            symbol="alpha",
            model_parameter_name="alpha",
            model_parameter_index=parameter_index,
        ),
    )
    expressions = ("2*alpha",)
    output_layout = ("model-parameter:beta",)
    kernel = PreparedKernelSpec(
        kernel_id=0,
        contract_kind="model-parameter",
        canonical_signature=_signature(
            "model-parameter", inputs, expressions, output_layout
        ),
        exact_expressions=expressions,
        inputs=inputs,
        output_layout=output_layout,
    )
    return PreparedKernelCatalog(
        model_name="generic-parameter-model",
        kernels=(kernel,),
        vertex_bindings=(),
        propagator_bindings=(),
        closure_bindings=(),
        model_parameter_kernel_id=0,
    )


def test_parameter_catalog_is_deterministic_and_binary64_exact() -> None:
    model = _ParameterModel()
    first = build_recurrence_template_catalog(
        model, _parameter_catalog(), compiled_model_digest=_MODEL_DIGEST
    )
    second = build_recurrence_template_catalog(
        model, _parameter_catalog(), compiled_model_digest=_MODEL_DIGEST
    )

    assert first == second
    assert first.canonical_json == second.canonical_json
    alpha = next(item for item in first.parameters if item.name == "alpha")
    assert alpha.default_value == ExactComplexRationalV1.from_binary64(0.1)
    assert alpha.prepared_parameter_id == 0
    beta = next(item for item in first.parameters if item.name == "beta")
    assert beta.parameter_kind == "derived"
    assert beta.default_value is None
    assert beta.prepared_parameter_id is None
    assert len(first.evaluator_bindings) == 1
    assert first.evaluator_bindings[0].semantic_template_ids == (beta.template_id,)


def test_catalog_round_trip_preserves_builder_output() -> None:
    catalog = build_recurrence_template_catalog(
        _ParameterModel(),
        _parameter_catalog(),
        compiled_model_digest=_MODEL_DIGEST,
    )
    loaded = RecurrenceTemplateCatalog.from_dict(json.loads(catalog.canonical_json))
    assert loaded == catalog


class _ComplexParameterModel(_ParameterModel):
    def runtime_parameter_defaults(self):
        return {"alpha": (0.1, 0.2)}

    def runtime_parameter_type(self, name):
        assert name == "alpha"
        return "complex"

    def runtime_derived_parameter_defaults(self):
        return {"beta": complex(0.2, 0.4)}


def test_complex_parameter_retains_authoritative_prepared_kernel_index() -> None:
    catalog = build_recurrence_template_catalog(
        _ComplexParameterModel(),
        _parameter_catalog(parameter_index=17),
        compiled_model_digest=_MODEL_DIGEST,
    )

    alpha = next(item for item in catalog.parameters if item.name == "alpha")
    assert alpha.value_type == "complex"
    assert alpha.prepared_parameter_id == 17
    assert alpha.default_value == ExactComplexRationalV1.from_binary64(0.1, 0.2)


def test_model_identity_mismatch_is_rejected() -> None:
    catalog = replace(_parameter_catalog(), model_name="different-model")
    with pytest.raises(PreparedKernelCatalogError, match="model identity"):
        build_recurrence_template_catalog(
            _ParameterModel(), catalog, compiled_model_digest=_MODEL_DIGEST
        )


def test_stale_prepared_kernel_signature_is_rejected() -> None:
    catalog = _parameter_catalog()
    stale = replace(catalog.kernels[0], exact_expressions=("3*alpha",))
    mutated = replace(catalog, kernels=(stale,))
    with pytest.raises(PreparedKernelCatalogError, match="stale canonical signature"):
        build_recurrence_template_catalog(
            _ParameterModel(), mutated, compiled_model_digest=_MODEL_DIGEST
        )


class _ScalarModel(Model):
    def __init__(self) -> None:
        particle = Particle(
            pdg=101,
            anti_pdg=101,
            spin=1,
            dimension=1,
            color_rep=1,
        )
        super().__init__(
            name="generic-scalar-model",
            particles={101: particle},
            vertices=(Vertex(0, (101, 101, 101)),),
        )
        self.source_kernel_id = -1

    def color_rep(self, pdg):
        return self.particle(pdg).color_rep

    def is_fermion(self, pdg):
        del pdg
        return False

    def is_chiral_eligible(self, pdg):
        del pdg
        return False

    def is_fundamental_colored_fermion(self, pdg):
        del pdg
        return False

    def is_massless_adjoint_vector(self, pdg):
        del pdg
        return False

    def quantum_number_flow(self, particle_id):
        del particle_id
        return (("generic-charge", "0"),)

    def vertex_evaluation_equivalence(self, kind):
        assert kind == 0
        return VertexEvaluationEquivalence(class_id="generic-scalar-exact-identity-v1")

    def vertex_coupling_orders(self, vertex):
        assert vertex.kind == 0
        return (("GENERIC", 1),)

    def vertex_color_structure(self, vertex):
        assert vertex.kind == 0
        return "singlet"

    def vertex_color_weight(self, vertex, *, color_accuracy):
        assert vertex.kind == 0
        assert color_accuracy == "lc"
        return (1.0, 0.0)

    def propagator_lowering_rule(self, particle_id, chirality=0):
        assert particle_id == 101
        assert chirality == 0
        return PropagatorLoweringRule(
            particle_id=particle_id,
            chirality=chirality,
            backend="identity",
            full_tensor_network_ready=True,
            applies_propagator=False,
            kernel="generic-scalar-identity",
            kind="identity",
            mass_class="not-applicable",
            auxiliary_policy="external-synthetic-scalar",
        )

    def recurrence_source_kernel_id(self, particle_id, chirality, helicity, spin_state):
        assert (particle_id, chirality, helicity, spin_state) == (101, 0, 0, 0)
        return self.source_kernel_id


def _kernel_namespace(
    *,
    contract_kind: str,
    inputs: tuple[PreparedKernelInput, ...],
    expressions: tuple[str, ...],
    output_layout: tuple[str, ...],
):
    return SimpleNamespace(
        kernel_id=-1,
        contract_kind=contract_kind,
        canonical_signature=_signature(
            contract_kind, inputs, expressions, output_layout
        ),
        exact_expressions=expressions,
        inputs=inputs,
        output_layout=output_layout,
        proof_classes=(),
    )


def _scalar_catalog(model: _ScalarModel):
    source_inputs = (PreparedKernelInput(role="momentum", component=0, symbol="p0"),)
    source = _kernel_namespace(
        contract_kind="source",
        inputs=source_inputs,
        expressions=("1",),
        output_layout=("scalar:c0",),
    )
    vertex_inputs = (
        PreparedKernelInput(role="left-current", component=0, symbol="left0"),
        PreparedKernelInput(role="right-current", component=0, symbol="right0"),
    )
    vertex = _kernel_namespace(
        contract_kind="vertex",
        inputs=vertex_inputs,
        expressions=("left0*right0",),
        output_layout=("scalar:c0",),
    )
    ordered = sorted((source, vertex), key=lambda item: item.canonical_signature)
    for kernel_id, kernel in enumerate(ordered):
        kernel.kernel_id = kernel_id
    model.source_kernel_id = source.kernel_id

    state = PreparedParticleState(
        particle_id=101,
        identity=model._particle_identity_ir(101).canonical_id,
        orientation="self-conjugate",
        basis="scalar",
        chirality=0,
        dimension=1,
    )
    equivalence = model.vertex_evaluation_equivalence(0)
    vertex_binding = PreparedVertexBinding(
        key=VertexKernelKey(0, (101, 101, 101), 0, 0, 0, (1.0, 0.0)),
        kernel_id=vertex.kernel_id,
        canonical_input_order=equivalence.input_order,
        equivalence_class=equivalence.class_id,
        equivalence_factor=equivalence.factor,
        input_exchange_factor=equivalence.input_exchange_factor,
        left_state=state,
        right_state=state,
        result_state=state,
    )
    propagator_binding = PreparedPropagatorBinding(
        key=PropagatorKernelKey(101, 0),
        kernel_id=None,
        state=state,
        applies_propagator=False,
        propagator_kind="identity",
        mass_class="not-applicable",
        gauge=None,
        model_parameters=(),
    )
    return SimpleNamespace(
        model_name=model.name,
        kernels=tuple(ordered),
        vertex_bindings=(vertex_binding,),
        propagator_bindings=(propagator_binding,),
        closure_bindings=(),
        model_parameter_kernel_id=None,
        unsupported_variants=(),
    )


def test_model_generic_scalar_catalog_covers_source_flow_color_and_propagator() -> None:
    model = _ScalarModel()
    catalog = build_recurrence_template_catalog(
        model,
        _scalar_catalog(model),  # type: ignore[arg-type]
        compiled_model_digest=_MODEL_DIGEST,
    )

    assert len(catalog.current_states) == 1
    assert len(catalog.sources) == 1
    assert len(catalog.quantum_flows) == 1
    assert len(catalog.transitions) == 1
    assert len(catalog.propagators) == 1
    assert not catalog.propagators[0].applies_propagator
    assert catalog.color_contractions[0].rule_kind == "singlet"
    assert {item.contract_kind for item in catalog.evaluator_bindings} == {
        "source",
        "vertex",
    }
    assert "built-in" not in catalog.canonical_json
    assert "ufo" not in catalog.canonical_json.lower()


def test_source_fill_uses_a_generic_runtime_template() -> None:
    model = _ScalarModel()
    model.recurrence_source_kernel_id = None  # type: ignore[method-assign]
    catalog = build_recurrence_template_catalog(
        model,
        _scalar_catalog(model),  # type: ignore[arg-type]
        compiled_model_digest=_MODEL_DIGEST,
    )

    source = next(
        binding
        for binding in catalog.evaluator_bindings
        if binding.contract_kind == "source"
    )
    assert source.callable_kind == "rusticol-template"
    assert source.prepared_kernel_id is None
    assert source.runtime_template is not None
    assert source.runtime_template.startswith("rusticol.source-fill.scalar.v1:")


class _GhostFilteringModel(_ScalarModel):
    def __init__(self) -> None:
        super().__init__()
        ghost = Particle(
            pdg=909,
            anti_pdg=-909,
            spin=-1,
            dimension=1,
            color_rep=8,
        )
        anti_ghost = replace(ghost, pdg=-909, anti_pdg=909)
        self.particles = {**self.particles, 909: ghost, -909: anti_ghost}
        self.vertices = (*self.vertices, Vertex(1, (909, -909, 101)))

    def source_wavefunction_kind(self, particle_id):
        if abs(int(particle_id)) == 909:
            return "ghost"
        return super().source_wavefunction_kind(particle_id)


def test_ghost_only_bindings_are_excluded_from_recurrence_semantics() -> None:
    model = _GhostFilteringModel()
    prepared = _scalar_catalog(model)
    physical = prepared.vertex_bindings[0]
    ghost_left = PreparedParticleState(
        particle_id=909,
        identity=model._particle_identity_ir(909).canonical_id,
        orientation=model._particle_identity_ir(909).orientation,
        basis=model._current_basis(909, 0),
        chirality=0,
        dimension=1,
    )
    ghost_right = PreparedParticleState(
        particle_id=-909,
        identity=model._particle_identity_ir(-909).canonical_id,
        orientation=model._particle_identity_ir(-909).orientation,
        basis=model._current_basis(-909, 0),
        chirality=0,
        dimension=1,
    )
    ghost_binding = PreparedVertexBinding(
        key=VertexKernelKey(1, (909, -909, 101), 0, 0, 0, (1.0, 0.0)),
        kernel_id=physical.kernel_id,
        canonical_input_order=(0, 1),
        equivalence_class="ghost-interaction",
        equivalence_factor=(1.0, 0.0),
        input_exchange_factor=None,
        left_state=ghost_left,
        right_state=ghost_right,
        result_state=physical.result_state,
    )
    prepared.vertex_bindings = (physical, ghost_binding)

    catalog = build_recurrence_template_catalog(
        model,
        prepared,  # type: ignore[arg-type]
        compiled_model_digest=_MODEL_DIGEST,
    )

    assert {state.particle_id for state in catalog.current_states} == {101}
    assert len(catalog.transitions) == 1
    assert all("ghost" not in item.contract_kind for item in catalog.evaluator_bindings)


def test_direct_contraction_uses_exact_runtime_closure_template() -> None:
    model = _ScalarModel()
    model._direct_contraction_ir_by_state = {
        (101, 0, 101, 0): ContractionIR(
            name="generic-scalar-pairing",
            left_basis="scalar",
            right_basis="scalar",
            coefficients=((0.5, 0.0),),
        )
    }

    catalog = build_recurrence_template_catalog(
        model,
        _scalar_catalog(model),  # type: ignore[arg-type]
        compiled_model_digest=_MODEL_DIGEST,
    )

    assert len(catalog.closures) == 1
    closure = catalog.closures[0]
    assert closure.component_coefficients == (
        ExactComplexRationalV1.from_binary64(0.5),
    )
    evaluator = next(
        binding
        for binding in catalog.evaluator_bindings
        if binding.resolver_key == closure.evaluator_resolver_key
    )
    assert evaluator.callable_kind == "rusticol-template"
    assert evaluator.runtime_template is not None
    assert evaluator.runtime_template.startswith("rusticol.closure-reduce.v1:")


def test_mutated_vertex_equivalence_factor_is_rejected() -> None:
    model = _ScalarModel()
    catalog = _scalar_catalog(model)
    binding = replace(catalog.vertex_bindings[0], equivalence_factor=(-1.0, 0.0))
    catalog.vertex_bindings = (binding,)
    with pytest.raises(PreparedKernelCatalogError, match=r"stale.*proof metadata"):
        build_recurrence_template_catalog(
            model,
            catalog,  # type: ignore[arg-type]
            compiled_model_digest=_MODEL_DIGEST,
        )


class _NondeterministicFlowModel(_ScalarModel):
    def __init__(self) -> None:
        super().__init__()
        self._flow_calls = 0

    def allowed_quantum_flows(self, vertex, left_index, right_index):
        self._flow_calls += 1
        coupling = (1.0 if self._flow_calls % 2 else 2.0, 0.0)
        return (
            QuantumFlow(
                chirality=0,
                spin_state=0,
                flavour_flow=(101,),
                quantum_number_flow=(("generic-charge", "0"),),
                coupling=coupling,
            ),
        )


def test_nondeterministic_quantum_flow_callback_fails_closed() -> None:
    model = _NondeterministicFlowModel()
    with pytest.raises(RecurrenceTemplateError, match="nondeterministic"):
        build_recurrence_template_catalog(
            model,
            _scalar_catalog(model),  # type: ignore[arg-type]
            compiled_model_digest=_MODEL_DIGEST,
        )


class _UnsupportedColorModel(_ScalarModel):
    def vertex_color_structure(self, vertex):
        del vertex
        return "opaque-model-tensor"


def test_unsupported_color_semantics_fail_closed() -> None:
    model = _UnsupportedColorModel()
    with pytest.raises(RecurrenceTemplateError, match="cannot encode color rule"):
        build_recurrence_template_catalog(
            model,
            _scalar_catalog(model),  # type: ignore[arg-type]
            compiled_model_digest=_MODEL_DIGEST,
        )
