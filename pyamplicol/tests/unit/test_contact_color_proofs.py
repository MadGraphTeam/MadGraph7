# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pyamplicol._internal.physics.symbols import symbols
from pyamplicol.models.compiler_contacts import _four_point_contact_color_split
from pyamplicol.models.compiler_entry import _compile_four_point_contact_kernels
from pyamplicol.models.contracts import CompiledParticleRecord, CompiledVertexTerm


def _adjoint(name: str, pdg: int) -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=name,
        pdg_code=pdg,
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
