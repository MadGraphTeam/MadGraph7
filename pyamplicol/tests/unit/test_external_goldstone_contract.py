# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pyamplicol.models import CompiledUFOModel, compile_model_source
from pyamplicol.models.loading import CompiledModel

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
def compiled_external_sm() -> CompiledModel:
    return compile_model_source(
        MODEL_ROOT / "sm.json",
        restriction=str((MODEL_ROOT / "restrict_default.json").resolve()),
        use_cache=False,
    )


def test_packaged_external_sm_uses_unique_absorbing_goldstone_contract(
    compiled_external_sm: CompiledModel,
) -> None:
    model = CompiledUFOModel(compiled_external_sm)

    assert model.inactive_goldstone_names == frozenset({"G0", "G+", "G-"})
    assert len(model.vertices) == 469
    for particle_id in (23, 24, -24):
        propagator = model._propagator_ir(particle_id)
        assert propagator.kind == "vector"
        assert propagator.mass_class == "massive"
        assert propagator.gauge == "unitary"
        assert propagator.goldstone_policy == "absorbed"


def test_declared_unrelated_scalar_uses_supported_unique_match_contract(
    compiled_external_sm: CompiledModel,
) -> None:
    # No compiled partner relation ties H to Z. Marking the arbitrary-PDG scalar
    # as a Goldstone makes the unique absorbing-vector match the supported proof.
    particles = tuple(
        replace(
            particle,
            pdg_code=700_001,
            mass="MZ",
            width="WZ",
            goldstoneboson=True,
        )
        if particle.name == "H"
        else particle
        for particle in compiled_external_sm.ir.particles
    )
    compiled = replace(
        compiled_external_sm,
        ir=replace(compiled_external_sm.ir, particles=particles),
    )

    model = CompiledUFOModel(compiled)
    declared_goldstone = next(
        particle for particle in particles if particle.name == "H"
    )
    vector = next(particle for particle in particles if particle.name == "Z")

    assert declared_goldstone.pdg_code == 700_001
    assert (
        declared_goldstone.mass,
        declared_goldstone.color,
        declared_goldstone.charge,
    ) == (vector.mass, vector.color, vector.charge)
    assert model._goldstone_is_redundant_in_unitary_gauge(declared_goldstone)
    assert "H" in model.inactive_goldstone_names
    assert not any(700_001 in vertex.particles for vertex in model.vertices)


def test_unflagged_unrelated_scalar_is_not_absorbed_by_degenerate_vector(
    compiled_external_sm: CompiledModel,
) -> None:
    particles = tuple(
        replace(
            particle,
            pdg_code=700_002,
            mass="MZ",
            width="WZ",
        )
        if particle.name == "H"
        else particle
        for particle in compiled_external_sm.ir.particles
    )
    compiled = replace(
        compiled_external_sm,
        ir=replace(compiled_external_sm.ir, particles=particles),
    )

    model = CompiledUFOModel(compiled)
    unrelated_scalar = next(particle for particle in particles if particle.name == "H")
    vector = next(particle for particle in particles if particle.name == "Z")

    assert unrelated_scalar.goldstoneboson is False
    assert (
        unrelated_scalar.mass,
        unrelated_scalar.color,
        unrelated_scalar.charge,
    ) == (vector.mass, vector.color, vector.charge)
    assert "H" not in model.inactive_goldstone_names
    assert any(700_002 in vertex.particles for vertex in model.vertices)


def test_ambiguous_degenerate_absorbing_vectors_fail_closed(
    compiled_external_sm: CompiledModel,
) -> None:
    particles = tuple(
        replace(
            particle,
            pdg_code=900_001,
            mass="MZ",
            width="WZ",
        )
        if particle.name == "a"
        else particle
        for particle in compiled_external_sm.ir.particles
    )
    compiled = replace(
        compiled_external_sm,
        ir=replace(compiled_external_sm.ir, particles=particles),
    )

    with pytest.raises(
        ValueError,
        match=r"Goldstone 'G0' ambiguously matches absorbing vectors \['a', 'Z'\]",
    ):
        CompiledUFOModel(compiled)


def test_custom_vector_propagator_does_not_absorb_a_goldstone(
    compiled_external_sm: CompiledModel,
) -> None:
    propagators = tuple(
        replace(propagator, custom=True) if propagator.particle == "Z" else propagator
        for propagator in compiled_external_sm.ir.propagators
    )
    compiled = replace(
        compiled_external_sm,
        ir=replace(compiled_external_sm.ir, propagators=propagators),
    )

    model = CompiledUFOModel(compiled)

    assert model._propagator_ir(23).kind == "custom"
    assert model._propagator_ir(23).goldstone_policy == "model-supplied"
    assert model.inactive_goldstone_names == frozenset({"G+", "G-"})
    assert any(250 in vertex.particles for vertex in model.vertices)
