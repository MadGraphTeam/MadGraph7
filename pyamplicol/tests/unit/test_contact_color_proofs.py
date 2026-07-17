# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pyamplicol._internal.physics.symbols import symbols
from pyamplicol.models import compiler_symbolica as _sym
from pyamplicol.models.compiler_contacts import _four_point_contact_color_split
from pyamplicol.models.compiler_entry import _compile_four_point_contact_kernels
from pyamplicol.models.contracts import CompiledParticleRecord, CompiledVertexTerm


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


def test_unproved_colored_four_point_contact_fails_closed() -> None:
    term = _term(
        color_source="UFO::{}::T(1,2,3)",
        color_expression="model_adversarial::T(1,2,3)",
    )
    particles = tuple(
        _adjoint(name, 9_300_000 + index)
        for index, name in enumerate(term.particles)
    )

    assert _four_point_contact_color_split(term, 0) is None
    auxiliaries, kernels = _compile_four_point_contact_kernels(
        (term,),
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

    split = _four_point_contact_color_split(scaled, 2)
    assert split is not None
    assert split[-1] == "-3/2"

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

    assert _four_point_contact_color_split(term, 0) is None
