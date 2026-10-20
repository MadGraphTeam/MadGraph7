# SPDX-License-Identifier: 0BSD
"""Retained policies select diagnostic-preserving transport for every observable."""

import json
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol.api import VirtualArithmeticSpec, VirtualStabilityPolicy
from pyamplicol.runtime import virtual

from .test_public_virtual_colour import _transport_backend
from .test_public_virtual_spin import _runtime

pytestmark = pytest.mark.loop_fast


def _policy(precision=90):
    return VirtualStabilityPolicy(
        (VirtualArithmeticSpec("arbitrary", precision),), "1e-8", "1e-9", True
    )


def _verified(handle, name, observed):
    ordinary = (
        handle.evaluate_directed if name == "directed" else handle.evaluate_partial
    )

    def evaluate(points, scales, higher, relative, absolute, **options):
        observed.append((higher, relative, absolute, options.copy()))
        options.pop("verify_successful")
        if name == "directed":
            options["partial"] = True
        result = ordinary(points, scales, **options)
        diagnostics = [
            {
                "point_id": point,
                "status": "success",
                "accuracy_established": True,
                "agreeing_bits": [233, 299],
                "attempts": [],
                "detail": "retained-policy transport control",
            }
            for point in range(len(points))
        ]
        return SimpleNamespace(result=result, diagnostics_json=json.dumps(diagnostics))

    setattr(
        handle,
        f"evaluate_{name}_verified" if name == "directed" else "evaluate_verified",
        evaluate,
    )

    def raw(*args, **kwargs):
        pytest.fail("configured policy entered a result-only native method")

    setattr(
        handle, "evaluate_directed" if name == "directed" else "evaluate_partial", raw
    )


def test_load_policy_is_transported_before_source_binding(monkeypatch):
    _, native = _transport_backend(monkeypatch, "full")
    calls = []

    def bind(path, **kwargs):
        calls.append(kwargs)
        return native

    monkeypatch.setattr(
        virtual, "load_loop_native", lambda: SimpleNamespace(NativeVirtualRuntime=bind)
    )
    policy = _policy()
    backend = virtual.VirtualRuntimeBackend(
        Path("fixture"),
        arithmetic="arbitrary",
        precision=70,
        model_parameters={"x": "1e400"},
        stability=policy,
    )
    assert calls[0]["stability"] == virtual._stability_input(policy)
    assert calls[0]["parameters"] == {"x": ["1E+400", "0"]}
    assert backend.inspect()["stability"]["relative_tolerance"] == "1e-8"
    with pytest.raises(TypeError, match="VirtualStabilityPolicy"):
        virtual.VirtualRuntimeBackend(Path("fixture"), stability="not a policy")
    assert len(calls) == 1


@pytest.mark.parametrize("resolved", [False, True])
def test_retained_and_call_policy_keep_diagnostics_and_rebind_scope(
    monkeypatch, resolved
):
    backend, native = _transport_backend(monkeypatch, "full")
    retained, override = _policy(), _policy(110)
    backend._default_stability = retained
    observed, rebound = [], []
    _verified(native, "ordinary", observed)
    native.clear = lambda: None

    def rebind(**options):
        rebound.append(options)
        candidate = copy(native)
        candidate.arithmetic = options["arithmetic"]
        candidate.precision = options["precision"]
        candidate.precision_bits = 332
        return candidate

    native.rebind = rebind
    operation = backend.evaluate_resolved if resolved else backend.evaluate
    points = [[[1, 0, 0, 1]]]
    first = operation(points, mu_r2=1)
    assert first.diagnostics[0].accuracy_established
    assert (
        first.diagnostics[0].verification.detail == "retained-policy transport control"
    )
    changed = operation(
        points, mu_r2=1, arithmetic="arbitrary", precision=100, stability=override
    )
    assert rebound[-1]["stability"] == virtual._stability_input(override)
    assert changed.diagnostics[0].precision_bits == 332
    assert backend._native is native  # Per-call replacement is not a new default.
    backend.clear()
    operation([], mu_r2=[])
    assert observed[-1][0] == virtual._stability_input(retained)[0]
    assert observed[-2][0] == virtual._stability_input(override)[0]


def test_retained_policy_applies_to_directed_spin_and_preserves_exact_sources(
    monkeypatch,
):
    runtime, _, sources = _runtime(monkeypatch)
    backend, observed = runtime._backend, []
    backend._default_stability = _policy()
    _verified(backend._native, "directed", observed)
    literal = "1.00000000000000000000000000000000000000000000001"
    runtime.set_spin_correlation_vectors({1: [0, (literal, "-2"), 0, 0]})
    value = runtime.evaluate_correlated([[[1, 0, 0, 1]]], mu_r2=1)
    assert value.diagnostics[0].accuracy_established
    assert sources[-1][1][0][1] == [literal, "-2"]
    runtime.clear()
    assert runtime.evaluate_correlated([], mu_r2=[]).shape == (0, 3, 3)
    assert len(observed) == 2
