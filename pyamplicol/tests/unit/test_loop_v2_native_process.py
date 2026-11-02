# SPDX-License-Identifier: 0BSD
"""A v2 process evaluated natively without its LoopSlice (WP3.1 I6).

``native_process.NativeProcess`` stores the plan, the native tree program
(with the Born and counterterm closures and the parameter program) and a
structural assembly record. Loaded back from its byte members, it evaluates
with generation forbidden and gives bitwise the results of the slice's own
native executor: Born, every component per sector and the totals, with and
without poles, in the "uv" and "ir" pole selections (with both origins'
parts: the stored companion of the certified-scaleless families), in
binary64 and double-double, and the stability estimate of a rotation and the
reduction-direction check.
"""

from __future__ import annotations

import pytest

from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
from pyamplicol.loops.v2.native_process import NativeProcess
from pyamplicol.loops.v2.plan_export import NativeExecutor
from pyamplicol.loops.v2.stability import Rotation, StabilityChecks

PARAMETERS = {"aS": "0.118", "MT": "173"}


@pytest.mark.loop_physics
def test_stored_native_process_evaluates_without_generation(
    loop_sm_compiled_model, loop_native, monkeypatch
):
    from pyamplicol.models.loading import load_compiled_model
    from tools.loop_reference.exact_points import generate

    slice_ = LoopSlice.generate(
        load_compiled_model(loop_sm_compiled_model),
        SliceRequest(
            "g g > t t~", {"QCD": 2}, {"QCD": 4}, fixed_parameters={"MB": "0"}
        ),
        loop_native,
    )
    reference = NativeExecutor(slice_, forward=False, inputs="native")
    process = NativeProcess.of(reference)
    members = process.members()
    plan = reference._plan_json
    # The companion of the certified-scaleless families is stored with it.
    assert slice_.scaleless is not None and process.scaleless is not None
    assert "scaleless-tree-program.json" in members
    companion = process.scaleless.plan_json

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a stored native process must not generate")

    monkeypatch.setattr(LoopSlice, "generate", forbidden)
    loaded = NativeProcess.from_members(plan, dict(members), companion).executor(
        loop_native
    )
    points = [
        [[str(x) for x in p] for p in point]
        for point in generate([0, 0, 173, 173], count=2)
    ]
    scales = ["490000", "1960000"]
    checks = StabilityChecks(
        rotations=(Rotation.about("z", "3/5", "4/5"),), reduction_direction=True
    )
    for options in (
        {},
        {"poles": "none"},
        {"poles": "uv"},
        {"poles": "ir", "stability": checks},
        {"stability": checks},
        {"precision": Precision.double_double()},
    ):
        count = 1 if "precision" in options else 2
        mine = loaded.evaluate(
            points[:count], scales[:count], parameters=PARAMETERS, **options
        )
        theirs = reference.evaluate(
            points[:count], scales[:count], parameters=PARAMETERS, **options
        )
        for a, b in zip(mine, theirs, strict=True):
            assert a.born == b.born
            assert a.components == b.components
            assert a.totals == b.totals
            assert "virtual" in a.components
            assert a.pole_origins == b.pole_origins
            if "stability" in options:
                assert a.stability.relative == b.stability.relative
