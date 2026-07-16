# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pyamplicol.models.contracts import CompiledModelIR, CompiledParticleRecord
from pyamplicol.processes.ir import build_process_ir
from pyamplicol.processes.model import build_model_process_ir


def _record(
    name: str,
    antiname: str,
    pdg: int,
    spin: int,
    color: int,
) -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=antiname,
        pdg_code=pdg,
        spin=spin,
        color=color,
        mass="ZERO",
        width="ZERO",
        charge=0.0,
        ghost_number=0,
        propagating=True,
        goldstoneboson=False,
        propagator=None,
    )


def test_process_ir_preserves_physical_and_all_outgoing_order() -> None:
    ir = build_process_ir("d d~ > Z g")

    assert ir.process == "d d~ > z g"
    assert ir.key == "d_dbar_to_z_g"
    assert tuple(leg.particle for leg in ir.legs) == ("d", "d~", "z", "g")
    assert tuple(leg.outgoing_particle for leg in ir.legs) == (
        "d~",
        "d",
        "z",
        "g",
    )
    assert ir.initial_pdgs == (1, -1)
    assert ir.final_pdgs == (23, 21)
    assert ir.outgoing_pdgs == (-1, 1, 23, 21)
    assert ir.quark_labels == (2,)
    assert ir.antiquark_labels == (1,)


def test_process_external_model_uses_the_same_crossing_contract() -> None:
    particles = (
        _record("d", "d~", 1, 2, 3),
        _record("d~", "d", -1, 2, -3),
        _record("Z", "Z", 23, 3, 1),
        _record("g", "g", 21, 3, 8),
    )
    model = CompiledModelIR(
        name="synthetic-sm",
        orders=(),
        parameters=(),
        particles=particles,
        couplings=(),
        propagators=(),
        vertex_terms=(),
        oriented_kernels=(),
    )

    ir = build_model_process_ir("d d~ > Z g", model)

    assert ir.process == "d d~ > Z g"
    assert ir.outgoing_pdgs == (-1, 1, 23, 21)
    assert tuple(leg.label for leg in ir.legs) == (1, 2, 3, 4)
    assert ir.quark_lines.quark_pair_count == 1
