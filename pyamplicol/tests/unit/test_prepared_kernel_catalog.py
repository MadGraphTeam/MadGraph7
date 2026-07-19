# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections import Counter
from copy import copy
from dataclasses import replace
from pathlib import Path

import pytest

from pyamplicol.models import (
    BuiltinSMModel,
    CompiledUFOModel,
    compile_model_source,
)
from pyamplicol.models.prepared_catalog import (
    PreparedKernelCatalogError,
    PropagatorKernelKey,
    build_prepared_kernel_catalog,
)

MODEL_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "sm"
)


@pytest.fixture(scope="module")
def builtin_catalog():
    model = BuiltinSMModel()
    return model, build_prepared_kernel_catalog(model)


@pytest.fixture(scope="module")
def external_sm_catalog():
    compiled = compile_model_source(
        MODEL_ROOT / "sm.json",
        restriction=str((MODEL_ROOT / "restrict_default.json").resolve()),
        use_cache=True,
    )
    model = CompiledUFOModel(compiled)
    return compiled, model, build_prepared_kernel_catalog(model)


def test_builtin_catalog_is_deterministic_and_process_independent(
    builtin_catalog,
) -> None:
    model, expected = builtin_catalog
    reordered = copy(model)
    reordered.vertices = tuple(reversed(model.vertices))

    assert build_prepared_kernel_catalog(model) == expected
    assert build_prepared_kernel_catalog(reordered) == expected
    assert (
        build_prepared_kernel_catalog(reordered).resolver_manifest()
        == expected.resolver_manifest()
    )


def test_every_builtin_vertex_kind_has_a_constructive_binding(
    builtin_catalog,
) -> None:
    model, catalog = builtin_catalog

    assert (
        {binding.key.kind for binding in catalog.vertex_bindings}
        == {vertex.kind for vertex in model.vertices}
        == set(range(25))
    )
    assert not [
        gap
        for gap in catalog.unsupported_variants
        if gap.contract_kind == "vertex"
    ]


def test_builtin_chirality_dimensions_and_propagator_coverage(
    builtin_catalog,
) -> None:
    model, catalog = builtin_catalog
    by_id = catalog.by_id

    for binding in catalog.vertex_bindings:
        assert binding.left_state.dimension == model.current_dimension(
            binding.key.particles[0],
            binding.key.left_chirality,
        )
        assert binding.right_state.dimension == model.current_dimension(
            binding.key.particles[1],
            binding.key.right_chirality,
        )
        assert binding.result_state.dimension == model.current_dimension(
            binding.key.particles[2],
            binding.key.result_chirality,
        )
        assert by_id[binding.kernel_id].output_dimension == (
            binding.result_state.dimension
        )

    expected: set[PropagatorKernelKey] = set()
    for particle_id in sorted(
        {
            *(
                particle_id
                for particle_id in model.particles
                if model.source_wavefunction_kind(particle_id) != "ghost"
            ),
            *(
                model.anti_particle(particle_id)
                for particle_id in model.particles
                if model.source_wavefunction_kind(particle_id) != "ghost"
            ),
        }
    ):
        chiralities = (
            (0,)
            if model.auxiliary_kind(particle_id) is not None
            or model.particle(particle_id).spin < 0
            else tuple(
                sorted(
                    {state.chirality for state in model.source_spin_states(particle_id)}
                )
            )
        )
        expected.update(
            PropagatorKernelKey(particle_id, chirality) for chirality in chiralities
        )
    assert {binding.key for binding in catalog.propagator_bindings} == expected
    assert all(
        (binding.kernel_id is not None) == binding.applies_propagator
        for binding in catalog.propagator_bindings
    )


def test_direct_contractions_remain_native_while_vertex_closures_are_catalogued(
    builtin_catalog,
) -> None:
    _model, catalog = builtin_catalog
    kinds = Counter(kernel.contract_kind for kernel in catalog.kernels)

    assert "direct-contraction" not in kinds
    assert kinds["closure"] > 0
    assert catalog.closure_bindings
    assert all(binding.projection == "scalar" for binding in catalog.closure_bindings)


def test_catalog_exact_contracts_are_finite_namespaced_and_nonempty(
    builtin_catalog,
) -> None:
    _model, catalog = builtin_catalog
    forbidden = ("indeterminate", "nan", "infinity", "complexinf")

    for kernel in catalog.kernels:
        assert kernel.exact_expressions
        assert kernel.output_dimension == len(kernel.output_layout)
        assert all(
            expression and not any(marker in expression.lower() for marker in forbidden)
            for expression in kernel.exact_expressions
        )
        assert all(
            descriptor.symbol.startswith("pyamplicol::") for descriptor in kernel.inputs
        )


def test_external_sm_active_oriented_kernels_and_parameters_are_catalogued(
    external_sm_catalog,
) -> None:
    _compiled, model, catalog = external_sm_catalog
    active_kinds = {vertex.kind for vertex in model.vertices}

    assert {binding.key.kind for binding in catalog.vertex_bindings} == active_kinds
    assert catalog.model_parameter_kernel_id is not None
    parameter_kernel = catalog.by_id[catalog.model_parameter_kernel_id]
    assert parameter_kernel.contract_kind == "model-parameter"
    assert parameter_kernel.output_dimension == len(
        model.runtime_derived_parameter_definitions()
    )
    assert not catalog.closure_bindings
    assert not catalog.unsupported_variants


def test_external_sm_catalog_signatures_ignore_compiled_inventory_order(
    external_sm_catalog,
) -> None:
    compiled, _model, expected = external_sm_catalog
    reordered_ir = replace(
        compiled.ir,
        oriented_kernels=tuple(reversed(compiled.ir.oriented_kernels)),
    )
    reordered = CompiledUFOModel(replace(compiled, ir=reordered_ir))
    actual = build_prepared_kernel_catalog(reordered)

    assert tuple(kernel.canonical_signature for kernel in actual.kernels) == tuple(
        kernel.canonical_signature for kernel in expected.kernels
    )
    assert actual.resolver_mappings() == expected.resolver_mappings()


def test_external_and_builtin_orientation_metadata_share_generic_contracts(
    builtin_catalog,
    external_sm_catalog,
) -> None:
    _builtin_model, builtin = builtin_catalog
    _compiled, _external_model, external = external_sm_catalog
    builtin_states = {
        (
            binding.left_state.orientation,
            binding.left_state.basis,
            binding.left_state.dimension,
        )
        for binding in builtin.vertex_bindings
    }
    external_states = {
        (
            binding.left_state.orientation,
            binding.left_state.basis,
            binding.left_state.dimension,
        )
        for binding in external.vertex_bindings
    }

    assert ("self-conjugate", "lorentz-vector", 4) in (builtin_states & external_states)
    assert any(
        state[1:] == ("weyl-chiral", 2) for state in builtin_states & external_states
    )


def test_catalog_fails_closed_when_a_vertex_kind_has_no_exact_lowering() -> None:
    class BrokenBuiltin(BuiltinSMModel):
        def vertex_component_expression(self, kind, *args, **kwargs):
            if kind == 0:
                raise ValueError("deliberately unavailable")
            return super().vertex_component_expression(kind, *args, **kwargs)

    with pytest.raises(
        PreparedKernelCatalogError,
        match=r"cannot lower admitted vertex orientation kind=0",
    ):
        build_prepared_kernel_catalog(BrokenBuiltin())


def test_catalog_fails_closed_when_one_admitted_propagator_is_missing() -> None:
    class BrokenBuiltin(BuiltinSMModel):
        def _propagator_ir(self, particle_id, chirality=0):
            if particle_id == 21:
                raise ValueError("deliberately unavailable")
            return super()._propagator_ir(particle_id, chirality)

    with pytest.raises(
        PreparedKernelCatalogError,
        match=r"cannot lower admitted propagator particle=21, chirality=0",
    ):
        build_prepared_kernel_catalog(BrokenBuiltin())
