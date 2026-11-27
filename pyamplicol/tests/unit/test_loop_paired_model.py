# SPDX-License-Identifier: 0BSD
"""Paired source values/diagnostics survive loop-model preparation
(``loops.model.build_loop_model``, which v2's catalogue builds on)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pyamplicol.loops.model import build_loop_model
from pyamplicol.loops.scalar_frontend import lower_scalar_program
from pyamplicol.models.loop_import import RawLoopModel, load_raw_loop_model

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def raw(loop_sm_ufo, loop_native):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native preparation requires licensed Symbolica")
    path = str(loop_sm_ufo)
    return load_raw_loop_model(path)


@pytest.fixture(scope="module")
def paired(raw):
    return build_loop_model(raw, scalar_program=lower_scalar_program(raw))


def test_owner_and_exact_record_identity_are_preserved(raw, paired):
    assert paired.scalar_program.library.raw is raw
    for entry in paired.kernels:
        assert (
            entry.coupling
            is paired.scalar_program.definitions[("couplings", entry.coupling.id)]
        )
        assert entry.scalar_expression(0) == entry.coupling.coefficient(0)
    copied = RawLoopModel.from_dict(raw.to_dict())
    with pytest.raises(ValueError, match="different raw source"):
        build_loop_model(copied, scalar_program=paired.scalar_program)
    incomplete = replace(paired.scalar_program, definitions={})
    with pytest.raises(ValueError, match="declarations differ"):
        build_loop_model(raw, scalar_program=incomplete)


def test_value_only_kernel_requires_library_even_with_zero_diagnostic(paired):
    from symbolica import E

    from pyamplicol.loops.model_expressions import LaurentExpression

    part = paired.scalar_program.library.lower("complexconjugate(z)")
    entry = paired.kernels[0]
    definition = replace(
        entry.coupling, value=LaurentExpression(part.value), diagnostic=E("0")
    )
    guarded = replace(entry, coupling=definition)
    with pytest.raises(NotImplementedError, match="owning scalar helper library"):
        guarded.scalar_value(0)


def test_empty_outer_loop_groups_are_source_wildcards(raw):
    payload = raw.to_dict()
    record = payload["collections"]["counterterm_vertices"][0]
    record["fields"]["loop_particles"] = []
    changed = RawLoopModel.from_dict(payload)
    model = build_loop_model(changed, scalar_program=lower_scalar_program(changed))
    entries = [
        entry
        for entry in model.kernels
        if entry.collection == "counterterm_vertices"
        and entry.vertex_id == record["id"]
    ]
    assert entries and all(entry.loop_particles == ((),) for entry in entries)
    assert (
        changed.collections["counterterm_vertices"][0]["fields"]["loop_particles"] == ()
    )


def test_full_ew_scalar_program_can_prepare_unsummed_model_ir(loop_gmu_ufo):
    pytest.importorskip("symbolica")
    path = str(loop_gmu_ufo)
    raw = load_raw_loop_model(path, restriction="full")
    program = lower_scalar_program(raw)
    model = build_loop_model(raw, scalar_program=program)
    assert model.kernels and model.scalar_program is program
    assert any(e.scalar_expression().diagnostic != 0 for e in model.kernels)
    assert not model.scheme_validated and not model.completeness_validated
