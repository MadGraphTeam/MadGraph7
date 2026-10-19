# SPDX-License-Identifier: 0BSD
"""Configured native source admission retains exact, atomic owner state.

Guard-only synthetic artifacts are retained from existing physical programs;
complete amplitude rescue is independently exercised by the Rust control and
public adapter tests. These checks do not repeat that campaign.
"""
import json
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def native_policy_fixture():
    root = os.environ.get("PYAMPLICOL_SOURCE_BINDING_POLICY_FIXTURES")
    if not root:
        pytest.skip("retained guard-only source-binding fixtures required")
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    assert load_loop_native() is _rusticol._loop
    verify_native_module(_rusticol)
    return Path(root), _rusticol._loop


def _policy(*levels):
    return (list(levels), "1e-10", "1e-10", False)


@pytest.mark.parametrize("kind", ["program", "runtime"])
def test_native_policy_precedes_binding_and_stays_with_atomic_owner(
    native_policy_fixture, kind
):
    root, native = native_policy_fixture
    if kind == "program":
        factory = native.NativeLoopProgram
        path = root / "resolved-guard.pacbin"
    else:
        factory = native.NativeVirtualRuntime
        path = root / "resolved-artifact"
        if not path.is_dir():
            pytest.skip("public guard artifact has not been wrapped yet")
    primary_policy = _policy(
        ("double-double", None), ("arbitrary", 60), ("arbitrary", 80)
    )
    with pytest.raises(ValueError, match="nonfinite"):
        factory(str(path))
    # DoubleFloat shares the finite exponent range; no admitted source exists
    # when it is the only configured alternative.
    with pytest.raises(ValueError, match="nonfinite"):
        factory(str(path), stability=_policy(("double-double", None)))
    owner = factory(str(path), stability=primary_policy)
    assert owner.stability == primary_policy
    state = owner.source_state
    initial = state.snapshot()
    assert initial.binding_failure_stage == "real model input type"
    with pytest.raises(ValueError, match="nonfinite"):
        initial.arguments
    with pytest.raises(ValueError, match="nonfinite"):
        initial.values
    # A retained policy cannot be silently discarded at the complete-result
    # boundary. Only verified results carry the required attempt diagnostics.
    with pytest.raises(ValueError, match="verified evaluation"):
        owner.evaluate([], [])
    verified = owner.evaluate_verified([], [], *primary_policy[:3])
    assert verified.result.point_ids == []
    assert json.loads(verified.diagnostics_json) == []
    for changes in (
        {"aS": ["not-a-number", "0"]},
        {"MT": ["173", "1e-1000"]},
        {"unknown_parameter": ["1", "0"]},
    ):
        with pytest.raises(ValueError):
            state.update(changes)
        snapshot = state.snapshot()
        assert snapshot.revision == initial.revision
        assert snapshot.overrides == initial.overrides
        assert snapshot.binding_failure_stage == initial.binding_failure_stage
    mass = "190.000000000000000000000000000000000000001"
    updated = state.update({"MT": [mass, "0"]})
    assert updated.revision == initial.revision + 1
    assert updated.overrides["MT"] == [mass, "0"]
    assert updated.binding_failure_stage == "real model input type"
    assert initial.overrides.get("MT") != updated.overrides["MT"]
    assert initial.revision == 0
    same = owner.rebind()
    assert same.stability == primary_policy
    assert same.source_state.snapshot().overrides == updated.overrides
    # Inheritance is validated against the new primary; no silent policy
    # filtering or accidental aliasing of the original owner is allowed.
    with pytest.raises(ValueError, match="strictly increase"):
        owner.rebind(arithmetic="double-double")
    replacement = _policy(("arbitrary", 60), ("arbitrary", 80))
    dd = owner.rebind(arithmetic="double-double", stability=replacement)
    assert dd.stability == replacement
    assert dd.source_state.snapshot().binding_failure_stage == "real model input type"
    ap = owner.rebind(
        arithmetic="arbitrary", precision=60,
        stability=_policy(("arbitrary", 80)),
    )
    assert ap.source_state.snapshot().binding_failure_stage is None
    assert ap.source_state.snapshot().values.nonfinite_rows() == []
    assert owner.stability == primary_policy
    assert owner.source_state.snapshot().overrides == updated.overrides
    if kind == "runtime":
        owner.clear()
        assert owner.stability == primary_policy
        assert owner.source_state.snapshot().revision == updated.revision
        after_clear = owner.evaluate_verified([], [], *primary_policy[:3])
        assert json.loads(after_clear.diagnostics_json) == []
