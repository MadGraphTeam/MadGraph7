# SPDX-License-Identifier: 0BSD
"""The poles companion's seeds on the slice's compression certificates.

The companion of the certified-scaleless families (``LoopSlice.scaleless``,
design §12.4) shares the slice's colour, so the compressions the slice's seed
table certified compress its readings too (``LoopSlice._companion_seeds``);
on its own it has no single-legged instance to certify them from five legs
on and kept a generic seed per reading. Against that oracle
(``assemble.SHARED_COMPANION_CERTIFICATES = False``), natively evaluated at
frozen stream points: the finite parts and the ``"all"`` values are bitwise
equal, the ``"uv"`` and ``"ir"`` values and both origins' parts equal to
rounding (the seeds' decomposition changes the summation of the companion's
lanes); g g > g g g bitwise in every mode (default tier), g g > g g g g to
rounding (nightly).
"""

from __future__ import annotations

import pytest

from pyamplicol.loops.v2 import assemble
from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
from pyamplicol.loops.v2.plan_export import NativeExecutor
from tools.loop_benchmark import streams

PARAMETERS = {"aS": "0.118", "MT": "173"}


def _values(slice_, key: str):
    executor = NativeExecutor(slice_, forward=False, inputs="native")
    frozen = streams.load(key)
    points = [streams.exact(frozen, k) for k in range(2)]
    scales = ["1000000"] * 2
    out = {}
    for mode in ("all", "uv", "ir"):
        out[mode] = executor.evaluate(points, scales, parameters=PARAMETERS, poles=mode)
    return out


def _companions(loop_sm_compiled_model, loop_native, monkeypatch, expression, n):
    from pyamplicol.models.loading import load_compiled_model

    model = load_compiled_model(loop_sm_compiled_model)
    request = SliceRequest(
        expression,
        {"QCD": n, "QED": 0},
        {"QCD": n + 2, "QED": 0},
        fixed_parameters={"MB": "0"},
    )
    shared = LoopSlice.generate(model, request, loop_native)
    monkeypatch.setattr(assemble, "SHARED_COMPANION_CERTIFICATES", False)
    own = LoopSlice.generate(model, request, loop_native)
    return shared, own


def _compare(shared, own, key, tolerance):
    assert len(shared.scaleless.table.classes) < len(own.scaleless.table.classes)
    mine, theirs = _values(shared, key), _values(own, key)
    for mode, results in mine.items():
        for a, b in zip(results, theirs[mode], strict=True):
            for sector, laurent in a.components["virtual"].items():
                other = b.components["virtual"][sector].coefficients
                scale = max(abs(x) for x in other)
                assert laurent.coefficients[0] == other[0], (mode, sector)
                if mode == "all" or tolerance == 0:
                    assert laurent.coefficients == other, (mode, sector)
                for x, y in zip(laurent.coefficients, other, strict=True):
                    assert abs(x - y) <= tolerance * scale, (mode, sector)
            if mode != "all" and tolerance == 0:
                assert a.pole_origins == b.pole_origins


@pytest.mark.loop_physics
def test_companion_seeds_on_the_slice_certificates_ggg(
    loop_sm_compiled_model, loop_native, monkeypatch
):
    shared, own = _companions(
        loop_sm_compiled_model, loop_native, monkeypatch, "g g > g g g", 3
    )
    _compare(shared, own, "gg_ggg", 0.0)


@pytest.mark.loop_nightly
def test_companion_seeds_on_the_slice_certificates_gggg(
    loop_sm_compiled_model, loop_native, monkeypatch
):
    shared, own = _companions(
        loop_sm_compiled_model, loop_native, monkeypatch, "g g > g g g g", 4
    )
    _compare(shared, own, "gg_gggg", 1e-15)
