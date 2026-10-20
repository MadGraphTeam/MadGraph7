# SPDX-License-Identifier: 0BSD
"""Paired source values/diagnostics survive model, CT and field preparation."""

from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.loops.model import build_loop_model
from pyamplicol.loops.physical_counterterms import (
    distinct_flavour_model_virtual_program,
    top_pair_model_virtual_program,
)
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


def _points(name):
    path = Path(__file__).resolve().parents[2] / "tools/loop_reference" / name
    return np.asarray(
        json.loads(path.read_text(), parse_float=str, parse_int=str)["points"],
        dtype=str,
    )


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


# Open defect (41ccde87): the CT Bose-slot proof requires every ordinary native
# coupling to be owned by a paired source's scalar program, so a paired source
# over the independent legacy control model (same coupling names, different
# expressions) is rejected. Real paired sources compile through the generic
# source path; this legacy-control composition needs an explicit ownership
# decision before these controls can run again.
_LEGACY_CONTROL_OWNERSHIP = pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason="41ccde87 rejects paired sources over the independent control model",
)


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", Decimal("3e-13")),
        ("double-double", None, "generic", Decimal("3e-29")),
        ("arbitrary", 60, "generic", Decimal("3e-55")),
    ],
)
@_LEGACY_CONTROL_OWNERSHIP
def test_paired_qcd_preserves_components_and_native_field_helpers(
    raw, paired, arithmetic, precision, engine, tolerance
):
    points = _points("top_pair_qcd.json")[:2]
    options = dict(arithmetic=arithmetic, precision=precision, reduction_engine=engine)
    legacy = top_pair_model_virtual_program(build_loop_model(raw), **options)
    program = top_pair_model_virtual_program(paired, **options)
    assert all(c.scalar_program is paired.scalar_program for c in program.currents)
    assert all(
        c.scalar_program is paired.scalar_program for c in program.field_currents
    )
    assert any(c.coefficient_diagnostic != 0 for c in program.field_currents)
    scales, inputs = ["900000", "2100000"], ["0.100000000000000000000000001", "173"]
    old = legacy.evaluate_decimal(points, scales, model_values=inputs)
    new = program.evaluate_decimal(points, scales, model_values=inputs)
    with localcontext() as context:
        context.prec = 100
        for name in (
            "born",
            "virtual",
            "bare",
            "uv",
            "r2",
            "uv_local",
            "uv_external_fields",
        ):
            for left, right in zip(
                getattr(old, name).flat, getattr(new, name).flat, strict=True
            ):
                assert abs(Decimal(left) - Decimal(right)) < tolerance * max(
                    abs(Decimal(left)), Decimal("1")
                )
    # Conditional source coefficients cannot be extracted through a value-only
    # insertion interface even if this particular point has a safe condition.
    with pytest.raises(NotImplementedError, match="paired scalar"):
        program.external_fields[0].coefficient(0)
    current = next(
        c for c in program.currents if any(d != 0 for d in c.coefficient_diagnostics)
    )
    with pytest.raises(NotImplementedError, match="paired scalar"):
        current.insertion.coefficient_substitutions(current.epsilon_power)


def _modified_coupling(raw, expression):
    payload = raw.to_dict()
    # Source R2 gluon/quark vertex occurs in the already calibrated uudd control.
    base = build_loop_model(raw)
    entry = next(e for e in base.kernels if e.vertex_name == "V_R2GUU")
    fields = payload["collections"]["couplings"][entry.coupling.id]["fields"]
    old = fields["value"]
    assert old["kind"] == "expression"
    fields["value"] = {
        "kind": "expression",
        "expression": expression.format(original=old["expression"]),
    }
    changed = RawLoopModel.from_dict(payload)
    return build_loop_model(changed, scalar_program=lower_scalar_program(changed))


@pytest.mark.parametrize(
    "arithmetic,precision",
    [("double", None), ("double-double", None), ("arbitrary", 60)],
)
@_LEGACY_CONTROL_OWNERSHIP
def test_inactive_ct_helper_and_zero_value_diagnostic_are_checked(
    raw, arithmetic, precision
):
    # cond's source helper returns one for either branch even when its invalid
    # condition chooses a finite value. Only the paired diagnostic catches this.
    source = _modified_coupling(
        raw, "({original}) * (1 if MU_R < 1500 else cond(1/(MT-173),1,1))"
    )
    program = distinct_flavour_model_virtual_program(
        source, arithmetic=arithmetic, precision=precision
    )
    points = _points("uudd_qcd.json")[:2]
    good = program.evaluate_decimal(
        points, ["1000000", "1000000"], model_values=[".118", "173"]
    )
    assert np.isfinite(good.virtual.astype(float)).all()
    with pytest.raises(
        ArithmeticError,
        match=r"counterterm scalar coefficients and active diagnostics.*\[\(1, 1\)\]",
    ):
        program.evaluate_decimal(
            points, ["1000000", "4000000"], model_values=[".118", "173"]
        )
    zero = _modified_coupling(raw, "0 if 1/(MU_R-1000) else 0")
    zero_program = distinct_flavour_model_virtual_program(
        zero, arithmetic=arithmetic, precision=precision
    )
    assert any(
        all(v == 0 for v in c.coefficient_expressions)
        and any(d != 0 for d in c.coefficient_diagnostics)
        for c in zero_program.currents
    )
    with pytest.raises(
        ArithmeticError, match="counterterm scalar coefficients and active diagnostics"
    ):
        zero_program.evaluate_decimal(
            points[:1], ["1000000"], model_values=[".118", "173"]
        )


@_LEGACY_CONTROL_OWNERSHIP
def test_external_field_branch_keeps_ct_dependency_diagnostic(raw):
    payload = raw.to_dict()
    particle = next(
        p for p in payload["collections"]["particles"] if p["fields"]["pdg_code"] == 6
    )
    fields = particle["fields"]["counterterm"]["entries"]
    assert fields
    for _key, coefficients in fields:
        for item in coefficients["entries"]:
            item[1] = f"({item[1]}) * (1 if MU_R < 1500 else cond(1/(MT-173),1,1))"
    changed = RawLoopModel.from_dict(payload)
    source = build_loop_model(changed, scalar_program=lower_scalar_program(changed))
    program = top_pair_model_virtual_program(
        source, arithmetic="arbitrary", precision=60
    )
    points = _points("top_pair_qcd.json")[:2]
    program.evaluate_decimal(
        points, ["1000000", "1000000"], model_values=[".118", "173"]
    )
    with pytest.raises(
        ArithmeticError,
        match=r"external-field scalar coefficients and active diagnostics"
        r".*\[\(1, 1\)\]",
    ):
        program.evaluate_decimal(
            points, ["1000000", "4000000"], model_values=[".118", "173"]
        )
