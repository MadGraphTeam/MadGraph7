# SPDX-License-Identifier: 0BSD
"""The native SDK on a TIR-only artifact (m-lit lit-eft; LIT-SDK's one
``"auto"`` definition, DECISIONS 2026-11-07 21:50 R7: whoever lands second
keeps ONE).

A family beyond NinjaRS's rank <= N + 2 makes an artifact TIR-only: its
declaration records ``reductions = ["tirust"]``. The C ABI reads that
admitted set as the Python API's runtime does, through the one policy
(``rusticol_loop_reduction::reduction_policy``):

* the capabilities name TIRust as the ``"auto"`` default and the only
  admitted backend, as the Python API's ``inspect``;
* ``"auto"`` evaluates on TIRust with the Python API's reason, statuses,
  failure reasons and values (bitwise, at the exact values of the doubles);
* an explicit ``"ninjars"`` and the cross-backend check are refused typed
  with the Python API's message (``ReductionUnavailable``).

g g > x0 x0 x0 on HC_NLO_X0 (``tools/loop_v2_eft_execution.py``'s row, its
results without a correctness claim) is generated here, so the test is
nightly, as lit-eft's.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from tests.unit.test_loop_v2_sdk_extras import _error, _sdk
from tests.unit.test_loop_v2_sdk_induced import _record
from tests.unit.test_loop_v2_sdk_stability import (
    FAILED_STATUS,
    STATUS_RUNTIME_ERROR,
    SUCCESS_STATUS,
    _evaluate,
    _flat,
    _json,
    _same,
    _set,
)

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "loop_v2_eft_execution", ROOT / "tools" / "loop_v2_eft_execution.py"
)
assert _spec is not None and _spec.loader is not None
eft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eft)


@pytest.mark.loop_nightly
def test_the_sdk_serves_a_tir_only_artifact_as_the_python_api(
    hc_nlo_x0_ufo, loop_native, tmp_path
):
    from pyamplicol.loops.v2.reduction import ReductionUnavailable

    if not loop_native.tirust_available():
        pytest.fail("the TIR-only row needs a native extension with TIRust")
    artifact = eft.generate(tmp_path, "x0x0x0", hc_nlo_x0_ufo)
    runtime = eft.runtime(artifact)
    assert runtime.inspect()["reductions"] == ["tirust"]
    exact, exact_scales = eft.points("x0x0x0")
    # The SDK reads each double as its exact value: the Python API is given
    # those exact values.
    points = [[[float(Fraction(c)) for c in p] for p in point] for point in exact]
    scales = [float(Fraction(s)) for s in exact_scales]
    texts = [[[str(Decimal(c)) for c in p] for p in point] for point in points]
    mu_r2 = [str(Decimal(s)) for s in scales]
    parameters = json.dumps({k: [v, "0"] for k, v in eft.parameters().items()})

    sdk = _sdk()
    handle = ctypes.c_void_p()
    assert (
        sdk.rusticol_virtual_open(
            str(artifact).encode(),
            b"p",
            parameters.encode(),
            None,
            ctypes.byref(handle),
        )
        == 0
    ), _error(sdk)
    try:
        # The Python side runs on_failure="return": the same policy here (the
        # SDK's default best-effort serves refused points with status
        # BEST_EFFORT, test_loop_v2_best_effort).
        assert sdk.rusticol_virtual_set_on_failure(handle, b"return") == 0, _error(
            sdk
        )
        capabilities = _json(sdk.rusticol_virtual_capabilities_json, handle)
        summary = runtime.inspect()["reduction"]
        assert capabilities["reduction"]["auto_default"] == "tirust"
        assert capabilities["reduction"]["admitted"] == summary["admitted"]
        assert summary["admitted"] == ["tirust"]

        status, found = _evaluate(sdk, handle, points, scales)
        assert status == 0, _error(sdk)
        expected = runtime.evaluate(texts, mu_r2=mu_r2, on_failure="return")
        # The full call-level record (requested / effective / reason, the
        # fallback, its point count and the unchecked setting), as the
        # loop-induced SDK test compares it.
        assert found["diagnostics"]["reduction"] == _record(
            expected.diagnostics[0].reduction, expected
        )
        assert found["diagnostics"]["reduction"]["reason"] == eft.ONLY
        served = [value is not None for value in expected.values]
        assert any(served)
        assert found["statuses"] == [
            SUCCESS_STATUS if ok else FAILED_STATUS for ok in served
        ]
        assert _same(found["virtual"], _flat(expected))
        failures = found["diagnostics"]["failures"]
        for index, ok in enumerate(served):
            if not ok:
                assert "; ".join(failures[str(index)]) == (
                    expected.diagnostics[index].detail
                )

        # An explicit NinjaRS: refused by the call, in the Python API's words.
        assert sdk.rusticol_virtual_set_reduction(handle, b"ninjars") == 0
        status, _ = _evaluate(sdk, handle, points, scales)
        assert status == STATUS_RUNTIME_ERROR
        with pytest.raises(ReductionUnavailable) as refused:
            runtime.evaluate(texts, mu_r2=mu_r2, reduction="ninjars")
        assert _error(sdk) == str(refused.value)

        # The cross-backend check (with a rotation, as the Python API needs)
        # would compare with NinjaRS: refused typed.
        assert sdk.rusticol_virtual_set_reduction(handle, b"auto") == 0
        assert _set(sdk, handle, 1, None, False, False, True) == 0, _error(sdk)
        status, _ = _evaluate(sdk, handle, points, scales)
        assert status == STATUS_RUNTIME_ERROR
        from pyamplicol.api.virtual_results import VirtualStabilityChecks

        with pytest.raises(ReductionUnavailable) as refused:
            runtime.evaluate(
                texts,
                mu_r2=mu_r2,
                stability_checks=VirtualStabilityChecks(1, cross_backend=True),
            )
        assert _error(sdk) == str(refused.value)
    finally:
        sdk.rusticol_virtual_free(handle)
