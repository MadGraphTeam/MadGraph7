# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pyamplicol._internal.physics.symbols import symbols
from pyamplicol.models import compiler_contacts
from pyamplicol.models import compiler_symbolica as _sym
from pyamplicol.models.compiler_contacts import (
    CONTACT_DECOMPOSITION_ALGORITHM,
    CONTACT_DECOMPOSITION_ALGORITHM_VERSION,
    _four_point_contact_color_split,
    _record_contact_decomposition_proofs,
)
from pyamplicol.models.compiler_entry import _compile_four_point_contact_kernels
from pyamplicol.models.contracts import (
    CompiledModelIR,
    CompiledParticleRecord,
    CompiledVertexTerm,
)
from pyamplicol.models.loading import (
    CompiledModel,
    compiler_fingerprint,
    load_compiled_model,
)


def _adjoint(name: str, pdg: int, *, spin: int = 3) -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=name,
        pdg_code=pdg,
        spin=spin,
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


def _term(*, color_source: str, color_expression: str) -> CompiledVertexTerm:
    return CompiledVertexTerm(
        id=901,
        vertex="V_adversarial_contact",
        particles=("a", "b", "c", "d"),
        color_index=0,
        lorentz_index=0,
        color_source=color_source,
        color_expression=color_expression,
        lorentz_name="L_contact",
        lorentz_source="1",
        lorentz_expression="1",
        coupling="GC_contact",
        coupling_expression="1",
        coupling_orders=(),
    )


def _proof_term(
    term: CompiledVertexTerm,
    particles: tuple[CompiledParticleRecord, ...],
    *,
    model_name: str,
) -> CompiledVertexTerm:
    return _record_contact_decomposition_proofs(
        (term,),
        particles,
        model_symbols=symbols.model(model_name),
    )[0]


def test_unproved_colored_four_point_contact_fails_closed() -> None:
    term = _term(
        color_source="UFO::{}::T(1,2,3)",
        color_expression="model_adversarial::T(1,2,3)",
    )
    particles = tuple(
        _adjoint(name, 9_300_000 + index)
        for index, name in enumerate(term.particles)
    )

    proved_term = _proof_term(term, particles, model_name="adversarial-contact")
    proof = proved_term.contact_decomposition_proof

    assert _four_point_contact_color_split(term, 0) is None
    assert proof is not None
    assert proof.status == "unsupported"
    assert proof.splits == ()
    assert {reason.code for reason in proof.unsupported_reasons} == {
        "unsupported-color-factor-count"
    }
    assert dict(proof.unsupported_reasons[0].context)[
        "normalized_color_expression"
    ] == term.color_expression
    auxiliaries, kernels = _compile_four_point_contact_kernels(
        (proved_term,),
        particles,
        start_kind=0,
        model_symbols=symbols.model("adversarial-contact"),
    )

    assert auxiliaries == ()
    assert kernels == ()


def test_literal_color_singlet_keeps_generic_contact_split() -> None:
    term = _term(color_source="1", color_expression="1")

    split = _four_point_contact_color_split(term, 2)

    assert split is not None
    pair, remaining, *_metadata = split
    assert pair == (0, 1)
    assert remaining == 3


def test_structure_constant_contact_preserves_exact_color_coefficient() -> None:
    unit_expression = (
        "spenso::f(ufo_c_2,ufo_c_dummy_7_adjoint,ufo_c_1)"
        "*spenso::f(ufo_c_dummy_7_adjoint,ufo_c_3,ufo_c_4)"
    )
    scaled = _term(
        color_source=(
            "-3/2*UFO::{}::f(2,-7,1)*UFO::{}::f(-7,3,4)"
        ),
        color_expression=f"-3/2*{unit_expression}",
    )
    unit = _term(
        color_source="UFO::{}::f(2,-7,1)*UFO::{}::f(-7,3,4)",
        color_expression=unit_expression,
    )
    particles = tuple(
        _adjoint(name, 9_400_000 + index, spin=1)
        for index, name in enumerate(scaled.particles)
    )
    scaled = _proof_term(
        scaled,
        particles,
        model_name="contact-color-coefficient-scaled",
    )
    unit = _proof_term(
        unit,
        particles,
        model_name="contact-color-coefficient-unit",
    )

    split = _four_point_contact_color_split(scaled, 2)
    assert split is not None
    assert split[-1] == "-3/2"
    proof = scaled.contact_decomposition_proof
    assert proof is not None
    assert proof.status == "proven"
    assert proof.algorithm == CONTACT_DECOMPOSITION_ALGORITHM
    assert proof.algorithm_version == CONTACT_DECOMPOSITION_ALGORITHM_VERSION
    assert proof.original_color_source == scaled.color_source
    assert proof.normalized_color_expression == scaled.color_expression
    assert proof.original_lorentz_source == scaled.lorentz_source
    assert proof.normalized_lorentz_expression == scaled.lorentz_expression
    chosen = next(item for item in proof.splits if item.result_leg == 2)
    assert chosen.decomposition_kind == "two-structure-constants"
    assert chosen.pair_legs == (1, 0)
    assert chosen.remaining_leg == 3
    assert chosen.outer_color_factor == (2, -7, 1)
    assert chosen.final_color_factor == (-7, 3, 4)
    assert chosen.color_coefficient == "-3/2"
    assert chosen.component_axis_order == ()
    assert chosen.component_basis_order == (0,)
    assert chosen.component_expansion == ((0, 1),)
    assert chosen.dummy_index_mapping is not None
    assert chosen.dummy_index_mapping.source_index == -7
    assert chosen.dummy_index_mapping.normalized_symbol == "ufo_c_dummy_7_adjoint"
    assert chosen.dummy_index_mapping.outer_slot == 1
    assert chosen.dummy_index_mapping.final_slot == 0
    partials = tuple(
        item for item in chosen.orientations if item.stage == "partial"
    )
    finals = tuple(item for item in chosen.orientations if item.stage == "final")
    assert tuple(item.input_legs for item in partials) == ((1, 0), (0, 1))
    assert tuple(item.permutation_parity for item in partials) == (-1, 1)
    assert tuple(item.scalar_prefactor for item in partials) == ("1", "-1")
    assert tuple(item.input_legs for item in finals) == ((-1, 3), (3, -1))
    assert tuple(item.permutation_parity for item in finals) == (-1, 1)
    assert tuple(item.scalar_prefactor for item in finals) == ("-3/2", "3/2")

    model_symbols = symbols.model("contact-color-coefficient")
    _scaled_auxiliaries, scaled_kernels = _compile_four_point_contact_kernels(
        (scaled,),
        particles,
        start_kind=0,
        model_symbols=model_symbols,
    )
    _unit_auxiliaries, unit_kernels = _compile_four_point_contact_kernels(
        (unit,),
        particles,
        start_kind=0,
        model_symbols=model_symbols,
    )
    scaled_finals = tuple(
        kernel for kernel in scaled_kernels if kernel.vertex.endswith("::contact-final")
    )
    unit_finals = tuple(
        kernel for kernel in unit_kernels if kernel.vertex.endswith("::contact-final")
    )

    _sym._ensure_symbolica()
    assert len(scaled_finals) == len(unit_finals) > 0
    for scaled_kernel, unit_kernel in zip(
        scaled_finals,
        unit_finals,
        strict=True,
    ):
        assert scaled_kernel.particles == unit_kernel.particles
        for scaled_component, unit_component in zip(
            scaled_kernel.component_expressions,
            unit_kernel.component_expressions,
            strict=True,
        ):
            difference = (
                _sym.E(scaled_component) + _sym.E("3/2") * _sym.E(unit_component)
            ).expand()
            assert difference == _sym.E("0")


def test_structure_constant_contact_rejects_residual_color_tensor() -> None:
    term = _term(
        color_source=(
            "UFO::{}::f(-1,1,2)*UFO::{}::f(3,4,-1)*UFO::{}::T(1,2,3)"
        ),
        color_expression=(
            "spenso::f(ufo_c_dummy_1_adjoint,ufo_c_1,ufo_c_2)"
            "*spenso::f(ufo_c_3,ufo_c_4,ufo_c_dummy_1_adjoint)"
            "*spenso::t(ufo_c_1,ufo_c_2,ufo_c_3)"
        ),
    )

    particles = tuple(
        _adjoint(name, 9_500_000 + index)
        for index, name in enumerate(term.particles)
    )
    proved_term = _proof_term(
        term,
        particles,
        model_name="residual-contact-color",
    )
    proof = proved_term.contact_decomposition_proof

    assert _four_point_contact_color_split(term, 0) is None
    assert proof is not None
    assert proof.status == "unsupported"
    assert {reason.code for reason in proof.unsupported_reasons} == {
        "non-scalar-color-prefactor"
    }


def test_contact_proof_round_trips_and_lowering_does_not_rediscover(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expression = (
        "spenso::f(ufo_c_2,ufo_c_dummy_11_adjoint,ufo_c_1)"
        "*spenso::f(ufo_c_dummy_11_adjoint,ufo_c_3,ufo_c_4)"
    )
    term = _term(
        color_source="UFO::{}::f(2,-11,1)*UFO::{}::f(-11,3,4)",
        color_expression=expression,
    )
    particles = tuple(
        _adjoint(name, 9_600_000 + index, spin=1)
        for index, name in enumerate(term.particles)
    )
    proved_term = _proof_term(term, particles, model_name="serialized-contact-proof")
    model = CompiledModelIR(
        name="serialized-contact-proof",
        orders=(),
        parameters=(),
        particles=particles,
        couplings=(),
        propagators=(),
        vertex_terms=(proved_term,),
        oriented_kernels=(),
        direct_contractions=(),
        closure_contractions=(),
    )
    compiled = CompiledModel(
        source={"kind": "json"},
        producer=compiler_fingerprint(),
        model={"name": model.name},
        ir=model,
        parameter_defaults={},
        capabilities={},
        issues=(),
        phase_timings={},
        conversion_seconds=0.0,
    )
    serialized_path = compiled.write(tmp_path / "serialized-contact-proof")
    payload = json.loads(serialized_path.read_text(encoding="utf-8"))
    term_payload = payload["ir"]["vertex_terms"]
    assert isinstance(term_payload, list)
    assert term_payload[0]["contact_decomposition_proof"]["status"] == "proven"
    restored = load_compiled_model(serialized_path).ir.vertex_terms[0]

    invalid_payload = json.loads(serialized_path.read_text(encoding="utf-8"))
    invalid_payload["ir"]["vertex_terms"][0]["contact_decomposition_proof"][
        "algorithm_version"
    ] = CONTACT_DECOMPOSITION_ALGORITHM_VERSION + 1
    invalid_path = tmp_path / "invalid-contact-proof.pyAmplicol-model.json"
    invalid_path.write_text(json.dumps(invalid_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported contact decomposition proof"):
        load_compiled_model(invalid_path)

    def forbidden_discovery(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("contact split was rediscovered during lowering")

    monkeypatch.setattr(
        compiler_contacts,
        "_four_point_contact_color_split",
        forbidden_discovery,
    )
    auxiliaries, kernels = _compile_four_point_contact_kernels(
        (restored,),
        particles,
        start_kind=0,
        model_symbols=symbols.model("serialized-contact-proof"),
    )

    assert (
        restored.contact_decomposition_proof
        == proved_term.contact_decomposition_proof
    )
    assert auxiliaries
    assert any(kernel.vertex.endswith("::contact-final") for kernel in kernels)


def test_contact_proof_identity_and_algorithm_fail_closed() -> None:
    expression = (
        "spenso::f(ufo_c_2,ufo_c_dummy_13_adjoint,ufo_c_1)"
        "*spenso::f(ufo_c_dummy_13_adjoint,ufo_c_3,ufo_c_4)"
    )
    term = _term(
        color_source="UFO::{}::f(2,-13,1)*UFO::{}::f(-13,3,4)",
        color_expression=expression,
    )
    particles = tuple(
        _adjoint(name, 9_700_000 + index, spin=1)
        for index, name in enumerate(term.particles)
    )
    proved_term = _proof_term(term, particles, model_name="closed-contact-proof")
    proof = proved_term.contact_decomposition_proof
    assert proof is not None

    stale = replace(proved_term, color_expression="1")
    with pytest.raises(ValueError, match="proof identity mismatch"):
        _compile_four_point_contact_kernels(
            (stale,),
            particles,
            start_kind=0,
            model_symbols=symbols.model("closed-contact-proof"),
        )

    with pytest.raises(ValueError, match="unsupported contact decomposition proof"):
        replace(
            proof,
            algorithm_version=CONTACT_DECOMPOSITION_ALGORITHM_VERSION + 1,
        )

    with pytest.raises(ValueError, match="has no contact decomposition proof"):
        CompiledModelIR(
            name="missing-contact-proof",
            orders=(),
            parameters=(),
            particles=particles,
            couplings=(),
            propagators=(),
            vertex_terms=(term,),
            oriented_kernels=(),
            direct_contractions=(),
            closure_contractions=(),
        )

    assert compiler_fingerprint()["contact_decomposition_policy"] == (
        f"{CONTACT_DECOMPOSITION_ALGORITHM}-v"
        f"{CONTACT_DECOMPOSITION_ALGORITHM_VERSION}"
    )
