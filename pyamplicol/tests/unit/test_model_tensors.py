# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

from pyamplicol.models.compiler_entry import (
    _annotate_oriented_kernel_color_projections,
)
from pyamplicol.models.contracts import (
    CompiledOrientedKernel,
    CompiledParticleRecord,
    CompiledVertexTerm,
)
from pyamplicol.models.tensors import (
    classify_trilinear_color_expression,
    normalize_color_expression,
    project_trilinear_color_expression,
)
from pyamplicol.processes.model import ModelParticleCatalog


def _particle(name: str, color: int) -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=name,
        pdg_code=9000001,
        spin=1,
        color=color,
        mass="ZERO",
        width="ZERO",
        charge=0.0,
        quantum_numbers=(("electric_charge", "0"),),
        ghost_number=0,
        propagating=True,
        goldstoneboson=False,
        propagator=None,
    )


def test_model_tensors_reject_unsupported_colored_representations() -> None:
    with pytest.raises(ValueError, match="unsupported UFO color representation 6"):
        normalize_color_expression("1", [6, 1, 1])

    with pytest.raises(ValueError, match="particle 'sextet'"):
        ModelParticleCatalog("synthetic", (_particle("sextet", 6),))


def test_model_tensors_keep_supported_singlet_projection() -> None:
    normalized = normalize_color_expression("1", [1, 1, 1])
    assert normalized.expression == "1"


def test_trilinear_color_projection_proves_scaled_permuted_generator() -> None:
    normalized = normalize_color_expression("2*UFO::T(3,1,2)", [3, -3, 8])

    assert project_trilinear_color_expression(
        normalized.expression,
        [3, -3, 8],
    ) == ("fundamental-generator", 2.0 + 0.0j)


def test_trilinear_color_projection_does_not_trust_ufo_source_spelling() -> None:
    assert classify_trilinear_color_expression(
        "1",
        "UFO::T(3,1,2)",
        [3, -3, 8],
    ) == ("generic-tensor", 1.0 + 0.0j)


def test_external_compiler_rejects_unimplemented_symmetric_color_tensor() -> None:
    particles = tuple(
        CompiledParticleRecord(
            name=f"adjoint_{index}",
            antiname=f"adjoint_{index}",
            pdg_code=8_100_000 + index,
            spin=3,
            color=8,
            mass="ZERO",
            width="ZERO",
            charge=0.0,
            quantum_numbers=(("electric_charge", "0"),),
            ghost_number=0,
            propagating=True,
            goldstoneboson=False,
            propagator=None,
        )
        for index in range(3)
    )
    color = normalize_color_expression("UFO::d(1,2,3)", [8, 8, 8])
    term = CompiledVertexTerm(
        id=0,
        vertex="symmetric-adjoint",
        particles=tuple(particle.name for particle in particles),
        color_index=0,
        lorentz_index=0,
        color_source="UFO::d(1,2,3)",
        color_expression=color.expression,
        lorentz_name="L",
        lorentz_source="1",
        lorentz_expression="1",
        coupling="GC",
        coupling_expression="1",
        coupling_orders=(),
    )
    kernel = CompiledOrientedKernel(
        kind=0,
        term_id=0,
        vertex=term.vertex,
        particles=tuple(particle.name for particle in particles),
        source_particle_legs=(0, 1, 2),
        component_expressions=("1",),
        coupling_expression="1",
        coupling_orders=(),
        runtime_parameters=(),
        color_source=term.color_source,
        color_expression=term.color_expression,
    )

    with pytest.raises(ValueError, match="unsupported trilinear color tensor"):
        _annotate_oriented_kernel_color_projections((kernel,), particles, (term,))


def test_oriented_identity_retains_the_exact_vertex_color_proof() -> None:
    particles = (
        _particle("anti", -3),
        _particle("singlet", 1),
        _particle("fund", 3),
    )
    color = normalize_color_expression("UFO::Identity(1,3)", [-3, 1, 3])
    term = CompiledVertexTerm(
        id=0,
        vertex="identity",
        particles=("anti", "singlet", "fund"),
        color_index=0,
        lorentz_index=0,
        color_source="UFO::Identity(1,3)",
        color_expression=color.expression,
        lorentz_name="L",
        lorentz_source="1",
        lorentz_expression="1",
        coupling="GC",
        coupling_expression="1",
        coupling_orders=(),
    )
    kernel = CompiledOrientedKernel(
        kind=0,
        term_id=0,
        vertex=term.vertex,
        particles=("fund", "singlet", "fund"),
        source_particle_legs=(2, 1, 0),
        component_expressions=("1",),
        coupling_expression="1",
        coupling_orders=(),
        runtime_parameters=(),
        color_source="oriented-identity",
        color_expression="1",
    )

    (annotated,) = _annotate_oriented_kernel_color_projections(
        (kernel,), particles, (term,)
    )

    assert annotated.color_projection_structure == "color-identity"
    assert len(annotated.lc_color_transition_terms) == 1
    witness = annotated.lc_color_transition_terms[0]
    assert witness.component_operation == "inherit-left"
    assert witness.input_permutation == (0, 1)
