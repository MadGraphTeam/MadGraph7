# SPDX-License-Identifier: 0BSD
"""The native SDK serves loop-induced processes (m-lit LIT-SDK, PLAN §3.5,
§2.a.10; RULED Q36, Q41, Q8).

On the two LI-O1 rows (g g > h g g and u u~ > h g g, full colour, their
observables declared; ``loop_v2_induced_observables_rows``):

* an existing client opens the loop-induced artifact directly (no opt-in)
  and reads empty Born counts: zero Born orders and Born colour components
  in the shape, zero Born values in every result; the metadata names the
  kind and the interference (``contribution_kind`` "loop-induced",
  ``interference`` "conj(loop)*loop");
* every ordinary value equals the Python API's (``Runtime.evaluate``):
  binary64 bitwise (the executor's amplitudes and the counterterm kets added
  in Python's order, squared by the one native square of Q41), double-double
  and arbitrary precision equal as ``Decimal`` values (squared in
  ``v2_decimal`` in the canonical order); the PyO3 runtime
  (``evaluate_v2_artifact``, the same Rust code) as well;
* the pole selections "uv" and "ir" are refused in the Python API's words
  (its ``CompatibilityError``), and nothing changes;
* the reduction setter (``rusticol_virtual_set_reduction``) takes "auto",
  "ninjars" and "tirust" with the Python API's requested / effective /
  reason record (the one policy, ``reduction_policy``); an explicit TIRust
  call equals the Python API's TIRust call (bitwise where served, the same
  failed statuses and reasons where it fails closed); an unknown backend is
  an invalid argument.
"""

from __future__ import annotations

import ctypes
import json
import math
from decimal import Decimal
from pathlib import Path

import pytest

from tests.unit.loop_v2_induced_observables_rows import ROWS, generate
from tests.unit.test_loop_v2_induced import _fixture
from tests.unit.test_loop_v2_sdk_extras import (
    _Axes,
    _error,
    _Request,
    _ResultShape,
    _sdk,
)

pytestmark = pytest.mark.loop_physics

STATUS_INVALID_ARGUMENT = 1
STATUS_RUNTIME_ERROR = 3
SUCCESS_STATUS = 4  # RUSTICOL_VIRTUAL_ACCURACY_NOT_ESTABLISHED
FAILED_STATUS = 3


@pytest.fixture(scope="module")
def artifacts(loop_sm_ufo, loop_native, tmp_path_factory):
    return {
        name: generate(tmp_path_factory.mktemp(name), name, expression)
        for name, expression in ROWS.items()
    }


def _inputs(name: str, count: int = 2):
    """The fixture's first ``count`` points rounded to binary64 (the C ABI's
    inputs; the Python API reads their exact values) and their mu_R^2."""

    fixture = _fixture(name)
    points = [
        [[float(Decimal(c)) for c in momentum] for momentum in point]
        for point in fixture["points"][:count]
    ]
    scales = [float(Decimal(mu)) ** 2 for mu in fixture["mu_r"][:count]]
    return points, scales


def _exact(points, scales):
    return (
        [[[str(Decimal(c)) for c in p] for p in point] for point in points],
        [str(Decimal(s)) for s in scales],
    )


def _json(sdk, function, handle) -> dict:
    needed = ctypes.c_size_t()
    assert function(handle, None, ctypes.c_size_t(0), ctypes.byref(needed)) == 0
    buffer = ctypes.create_string_buffer(needed.value)
    assert function(handle, buffer, needed, ctypes.byref(needed)) == 0
    return json.loads(buffer.value)


def _open(sdk, artifact: Path):
    handle = ctypes.c_void_p()
    status = sdk.rusticol_virtual_open(
        str(artifact).encode(), b"p", None, None, ctypes.byref(handle)
    )
    assert status == 0, _error(sdk)
    return handle


def _evaluate(sdk, handle, points, scales) -> dict:
    """One ordinary C ABI evaluation: its shape, values, statuses, decimal
    record and diagnostics."""

    flat = [c for point in points for p in point for c in p]
    momenta = (ctypes.c_double * len(flat))(*flat)
    mu = (ctypes.c_double * len(scales))(*scales)
    request = _Request(momenta, len(flat), len(points), None, mu, len(scales))
    result = ctypes.c_void_p()
    status = sdk.rusticol_virtual_evaluate(
        handle, ctypes.byref(request), ctypes.byref(result)
    )
    assert status == 0, _error(sdk)
    try:
        shape = _ResultShape()
        assert sdk.rusticol_virtual_result_shape(result, ctypes.byref(shape)) == 0
        born = (ctypes.c_double * max(shape.born_values, 1))()
        loop = (ctypes.c_double * max(shape.virtual_values, 1))()
        statuses = (ctypes.c_uint32 * len(points))()
        assert sdk.rusticol_virtual_result_copy_born(result, born, shape.born_values) == 0
        assert (
            sdk.rusticol_virtual_result_copy_virtual(result, loop, shape.virtual_values)
            == 0
        )
        assert (
            sdk.rusticol_virtual_result_copy_statuses(result, statuses, len(points))
            == 0
        )
        return {
            "shape": shape,
            "virtual": list(loop)[: shape.virtual_values],
            "statuses": list(statuses),
            "decimal": _json(sdk, sdk.rusticol_virtual_result_decimal_json, result),
            "diagnostics": _json(
                sdk, sdk.rusticol_virtual_result_diagnostics_json, result
            ),
        }
    finally:
        sdk.rusticol_virtual_result_free(result)


def _python_flat(result) -> list:
    """The Python API's values in the SDK's layout (absent poles NaN)."""

    out = []
    for point in result.values:
        if point is None:
            continue
        for laurent in point:
            out.extend(math.nan if x is None else x for x in laurent)
    return out


def _same(found: list, expected: list) -> bool:
    return len(found) == len(expected) and all(
        (math.isnan(a) and math.isnan(float(b))) or a == b
        for a, b in zip(found, expected, strict=True)
    )


@pytest.mark.parametrize("name", sorted(ROWS))
def test_loop_induced_sdk_values_equal_the_python_api(artifacts, loop_native, name):
    from pyamplicol import Runtime

    artifact = artifacts[name]
    runtime = Runtime.load(artifact, process="p")
    points, scales = _inputs(name)
    exact_points, exact_scales = _exact(points, scales)
    sdk = _sdk()
    handle = _open(sdk, artifact)
    try:
        axes = _Axes()
        assert sdk.rusticol_virtual_shape(handle, ctypes.byref(axes)) == 0
        assert axes.born_orders == 0 and axes.born_color_components == 0
        assert axes.virtual_orders == len(runtime.evaluate(
            exact_points[:1], mu_r2=exact_scales[:1]
        ).values[0])
        metadata = _json(sdk, sdk.rusticol_virtual_metadata_json, handle)
        assert metadata["contribution_kind"] == "loop-induced"
        assert metadata["interference"] == "conj(loop)*loop"
        assert metadata["born_orders"] == []
        capabilities = _json(sdk, sdk.rusticol_virtual_capabilities_json, handle)
        assert capabilities["contribution_kind"] == "loop-induced"
        assert capabilities["pole_selections"] == ["all", "none"]
        for poles in ("all", "none"):
            assert sdk.rusticol_virtual_set_poles(handle, poles.encode()) == 0
            found = _evaluate(sdk, handle, points, scales)
            expected = runtime.evaluate(exact_points, mu_r2=exact_scales, poles=poles)
            assert found["shape"].born_values == 0
            assert found["statuses"] == [SUCCESS_STATUS] * len(points)
            assert _same(found["virtual"], _python_flat(expected)), (poles, name)
            assert found["decimal"] is None
            # The PyO3 runtime: the same Rust code.
            native = loop_native.evaluate_v2_artifact(
                str(artifact),
                [[c for p in point for c in p] for point in exact_points],
                exact_scales,
                process="p",
                poles=poles,
            )
            assert [failures for _values, failures in native] == [[], []]
            assert [values[0] for values, _f in native] == [[], []]
            flat = [
                x for values, _f in native for laurent in values[1] for x in laurent
            ]
            width = 3 if poles == "all" else 1
            assert flat == [
                x
                for point in expected.values
                for laurent in point
                for x in laurent[:width]
            ]
        # Extended carriers: the Python API's Decimal values, one point.
        assert sdk.rusticol_virtual_set_poles(handle, b"all") == 0
        for arithmetic, digits in (("double-double", 0), ("arbitrary", 32)):
            status = sdk.rusticol_virtual_set_arithmetic(
                handle, arithmetic.encode(), ctypes.c_uint32(digits)
            )
            assert status == 0, _error(sdk)
            found = _evaluate(sdk, handle, points[:1], scales[:1])
            options = {"arithmetic": arithmetic}
            if digits:
                options["precision"] = digits
            expected = runtime.evaluate(
                exact_points[:1], mu_r2=exact_scales[:1], **options
            )
            literals = found["decimal"]["virtual"]
            values = [x for laurent in expected.values[0] for x in laurent]
            assert found["decimal"]["born"] == []
            assert [Decimal(x) for x in literals] == [Decimal(x) for x in values]
            assert found["virtual"] == [float(Decimal(x)) for x in literals]
    finally:
        sdk.rusticol_virtual_free(handle)


def test_loop_induced_split_poles_are_refused_in_the_python_apis_words(artifacts):
    from pyamplicol import Runtime
    from pyamplicol.api.errors import CompatibilityError

    artifact = artifacts["li_gg_hgg"]
    runtime = Runtime.load(artifact, process="p")
    points, scales = _inputs("li_gg_hgg", 1)
    exact_points, exact_scales = _exact(points, scales)
    sdk = _sdk()
    handle = _open(sdk, artifact)
    try:
        for poles in ("uv", "ir"):
            with pytest.raises(CompatibilityError) as refused:
                runtime.evaluate(exact_points, mu_r2=exact_scales, poles=poles)
            status = sdk.rusticol_virtual_set_poles(handle, poles.encode())
            assert status == STATUS_RUNTIME_ERROR
            assert str(refused.value) in _error(sdk)
        # Nothing changed: the handle still evaluates every pole.
        found = _evaluate(sdk, handle, points, scales)
        assert found["diagnostics"]["poles"] == "all"
        assert len(found["virtual"]) == 3 * found["shape"].axes.virtual_orders
    finally:
        sdk.rusticol_virtual_free(handle)


@pytest.mark.parametrize("name", sorted(ROWS))
def test_the_reduction_setter_records_the_python_apis_choice(
    artifacts, loop_native, name
):
    from pyamplicol import Runtime

    artifact = artifacts[name]
    runtime = Runtime.load(artifact, process="p")
    points, scales = _inputs(name)
    exact_points, exact_scales = _exact(points, scales)
    tir = "tirust" in json.loads(loop_native.reduction_backends())
    sdk = _sdk()
    handle = _open(sdk, artifact)
    try:
        default = _evaluate(sdk, handle, points, scales)
        expected = runtime.evaluate(exact_points, mu_r2=exact_scales)
        record = expected.diagnostics[0].reduction
        assert default["diagnostics"]["reduction"] == {
            "requested": record.requested,
            "effective": record.effective,
            "reason": record.reason,
        }
        capabilities = _json(sdk, sdk.rusticol_virtual_capabilities_json, handle)
        assert capabilities["reduction"]["native"] == json.loads(
            loop_native.reduction_backends()
        )
        assert capabilities["reduction"]["auto_default"] == "tirust"
        for backend in ("auto", "ninjars", "tirust"):
            status = sdk.rusticol_virtual_set_reduction(handle, backend.encode())
            if backend == "tirust" and not tir:
                assert status == STATUS_RUNTIME_ERROR
                assert "not built into this native extension" in _error(sdk)
                continue
            assert status == 0, _error(sdk)
            found = _evaluate(sdk, handle, points, scales)
            expected = runtime.evaluate(
                exact_points, mu_r2=exact_scales, reduction=backend
            )
            record = expected.diagnostics[0].reduction
            assert found["diagnostics"]["reduction"] == {
                "requested": record.requested,
                "effective": record.effective,
                "reason": record.reason,
            }
            served = [value is not None for value in expected.values]
            assert found["statuses"] == [
                SUCCESS_STATUS if ok else FAILED_STATUS for ok in served
            ]
            assert _same(found["virtual"], _python_flat(expected)), backend
            failures = found["diagnostics"]["failures"]
            for index, ok in enumerate(served):
                if not ok:
                    assert "; ".join(failures[str(index)]) == (
                        expected.diagnostics[index].detail
                    )
        assert sdk.rusticol_virtual_set_reduction(handle, b"golem") == (
            STATUS_INVALID_ARGUMENT
        )
        assert "unknown reduction backend" in _error(sdk)
    finally:
        sdk.rusticol_virtual_free(handle)
