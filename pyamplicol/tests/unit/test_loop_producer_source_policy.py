# SPDX-License-Identifier: 0BSD
"""Producer source admission uses dependency programs and exact owner snapshots."""

from dataclasses import replace
from decimal import Decimal, localcontext

import pytest

from pyamplicol.api.virtual_results import VirtualArithmeticSpec, VirtualStabilityPolicy
from pyamplicol.loops.scalar_bindings import (
    NativeScalarBindings,
    NativeScaleBindings,
    SourceBindingFailure,
    prepare_scalar_bindings,
)
from tests.unit.test_loop_scale_bindings import source as scale_source

pytestmark = pytest.mark.loop_fast


@pytest.fixture
def source(tmp_path):
    return scale_source.__wrapped__(tmp_path)


@pytest.fixture
def native():
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    verify_native_module(_rusticol)
    assert load_loop_native() is _rusticol._loop
    return _rusticol._loop


def policy(*levels):
    return VirtualStabilityPolicy(
        tuple(VirtualArithmeticSpec(*level) for level in levels), "1e-20", "1e-30"
    )


@pytest.fixture
def guarded(source):
    plan = prepare_scalar_bindings(
        source, (("x", 0), ("z", 0), ("MU_R", 0), ("safe", 0))
    )
    library = plan.program.library
    guards = [
        library.lower("1 / (1 if z.imag == 0 else 0)", locals={"z": symbol})
        for kind, symbol in zip(plan.external_types, plan.input_symbols, strict=True)
        if kind == "real"
    ]
    symbols = dict(zip(plan.external_names, plan.input_symbols, strict=True))
    ratio = library.lower(
        "(cmath.exp(x+800)+cmath.exp(MU_R+801))/cmath.exp(x+802)", locals=symbols
    )
    guards[0] = replace(guards[0], diagnostic=guards[0].diagnostic * ratio.value)
    prepared = replace(
        plan.prepared, guards_state=library.evaluator(guards, plan.input_symbols).save()
    )
    changed = replace(plan)
    object.__setattr__(changed, "_prepared", prepared)
    return plan, changed


def test_failed_primary_snapshot_preserves_exact_owner_and_atomic_update(
    guarded, native
):
    baseline, guarded = guarded
    with pytest.raises(SourceBindingFailure, match="real model input type"):
        NativeScalarBindings(guarded, backend=native)
    with pytest.raises(SourceBindingFailure):
        NativeScalarBindings(
            guarded, backend=native, stability=policy(("double-double", None))
        )
    configured = policy(("double-double", None), ("arbitrary", 60), ("arbitrary", 80))
    owner = NativeScalarBindings(guarded, backend=native, stability=configured)
    before = owner.snapshot()
    assert before.binding_failure_stage == "real model input type"
    errors = []
    for name in ("arguments", "values", "arguments"):
        with pytest.raises(SourceBindingFailure) as failure:
            getattr(before, name)
        errors.append(failure.value)
    assert len({id(error) for error in errors}) == len(errors)
    assert not isinstance(before._failure, BaseException)
    for invalid in ({"x": "NaN"}, {"x": ("1", "1e-1000")}, {"unknown": "1"}):
        with pytest.raises(ValueError):
            owner.update(invalid)
        assert owner.snapshot() is before
    # This source branch compares a genuinely complex value with a real value;
    # no configured arithmetic makes its active diagnostic valid.
    with pytest.raises(SourceBindingFailure):
        owner.update({"MU_R": ".5"})
    assert owner.snapshot() is before
    exact = "3.00000000000000000000000000000000000000000000001"
    updated = owner.update({"x": exact})
    assert updated is owner.snapshot() and updated.revision == 1
    assert updated.overrides["x"] == (exact, "0")
    assert before.revision == 0 and before.overrides == {}
    high = guarded.prepared.bind(backend=native, arithmetic="arbitrary", precision=80)
    restored = high.reconstruct_snapshot(owner, updated)
    assert restored.revision == 1 and restored.binding_failure_stage is None
    with localcontext() as context:
        context.prec = 100
        assert abs(
            Decimal(restored.values.to_decimal()[0][0]) - Decimal(exact)
        ) < Decimal("1e-75")
    reference = NativeScalarBindings(
        baseline,
        backend=native,
        arithmetic="arbitrary",
        precision=80,
        parameters=updated.overrides,
    )
    assert restored.values.to_decimal() == reference.values.to_decimal()
    with pytest.raises(ValueError, match="strictly increase"):
        guarded.prepared.bind(
            backend=native, arithmetic="arbitrary", precision=80, stability=configured
        )


def test_failed_source_scale_batches_keep_original_ids_and_validate_inputs(
    guarded, native
):
    _, plan = guarded
    configured = policy(("arbitrary", 60), ("arbitrary", 80))
    state = NativeScalarBindings(plan, backend=native, stability=configured)
    view = NativeScaleBindings(state)
    for ids in ([91, 7, 40], [], [40]):
        scales = native.NativeBatch.from_decimal(
            [("4", "0")] * len(ids), len(ids), 1, row_ids=ids
        )
        values, failures = view.evaluate_partial_native(scales)
        assert (values.rows, values.columns, values.row_ids) == (0, 4, [])
        assert [failure.point_id for failure in failures] == list(ids)
        assert all(failure.stage == "real model input type" for failure in failures)
        with pytest.raises(SourceBindingFailure):
            view.evaluate_native(scales)
    invalid = native.NativeBatch.from_decimal([("-1", "0")], 1, 1, row_ids=[77])
    _, failures = view.evaluate_partial_native(invalid)
    assert {failure.stage for failure in failures} == {
        "positive real renormalization scales",
        "real model input type",
    }
    foreign = native.NativeBatch.from_decimal(
        [], 0, 1, arithmetic="arbitrary", precision=80
    )
    with pytest.raises(ValueError):
        view.evaluate_partial_native(foreign)
    other = NativeScalarBindings(plan, backend=native, stability=configured)
    with pytest.raises(ValueError, match="different global state"):
        view.evaluate_partial_native(invalid, snapshot=other.snapshot())
