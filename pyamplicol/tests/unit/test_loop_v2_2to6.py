# SPDX-License-Identifier: 0BSD
"""The 2->6 validation harness without a provider (M-FINAL MF-8G-3 (ii)-(v),
(vii); ``tools/loop_benchmark/stored_checks.py``), on stored native processes
of lower multiplicity: Catani's prediction formed natively from the stored
dipole orbits, the rotation estimate E, the universal double pole, binary64
against double-double, the Ward identity on a process stored from the full
tree kernels, and the default rescue policy's level-0 count.

The default tier runs the harness once at g g > g g (generation in seconds);
the nightly ids (append-only, shared with MF-8Q) run it at six and seven
gluons, the multiplicities that prove the 2->6 run a pure compute step.
"""

from __future__ import annotations

import pytest

from pyamplicol.loops.v2.native_process import NativeProcess
from pyamplicol.loops.v2.plan_export import NativeExecutor
from tools.loop_benchmark import pyamplicol_v2
from tools.loop_benchmark.stored_checks import check_process


def _stored(compiled, native, key: str, accuracy: str, kernels: str) -> NativeProcess:
    """The G1 harness's process ``key`` generated (on the full tree kernels
    with ``kernels="full"``), stored with native point inputs and read back
    from its members (``NativeProcess.from_members``)."""

    from pyamplicol.loops.v2.assemble import LoopSlice
    from pyamplicol.loops.v2.ward import ward_slice

    request = pyamplicol_v2._request(key, accuracy, compiled)
    generate = ward_slice if kernels == "full" else LoopSlice.generate
    slice_ = generate(compiled, request, native)
    process = NativeProcess.of(NativeExecutor(slice_, inputs="native", forward=False))
    return NativeProcess.from_members(process.plan, process.members())


@pytest.fixture(scope="module")
def compiled(loop_sm_compiled_model):
    from pyamplicol.models.loading import load_compiled_model

    return load_compiled_model(loop_sm_compiled_model)


def _assert_passed(record, *, catani: bool, ward: bool) -> None:
    assert record["passed"] is True, record["stages"]
    for row in record["checked"]:
        assert row["passed"], row
        assert row["catani_formed"] is catani
        assert row["double_pole"]["passed"]
        assert row["relative_estimate"] is not None
    for row in record.get("dd", ()):
        assert row["passed"], row
    if ward:
        # A binary64 point beyond the binary64 scope (O1) is recorded, not
        # asserted; at least one row is.
        asserted = [row["passed"] for row in record["ward"] if row["asserted"]]
        assert asserted and all(asserted)
    assert record["rescue"]["of"] == len(record["rescue"]["level0_estimates"])


@pytest.mark.loop_physics
def test_stored_checks_on_a_stored_ward_process(compiled, loop_native):
    process = _stored(compiled, loop_native, "gg_gg", "full", "full")
    assert process.record["kernel_variant"] == "full"
    assert "catani" in process.record
    record = check_process(
        process,
        loop_native,
        "gg_gg",
        "full",
        points=[0, 1],
        dd_points=[0],
        ward_points=[0],
        rescue_points=[0, 1, 2, 3],
    )
    _assert_passed(record, catani=True, ward=True)
    assert record["rescue"]["count"] == 0


@pytest.mark.loop_nightly
def test_stored_checks_at_six_gluons(compiled, loop_native):
    process = _stored(compiled, loop_native, "gg_gggg", "full", "full")
    record = check_process(
        process,
        loop_native,
        "gg_gggg",
        "full",
        points=[0, 1, 2],
        dd_points=[0],
        ward_points=[0],
        rescue_points=list(range(8)),
        threads=16,
    )
    _assert_passed(record, catani=True, ward=True)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("accuracy", ("full", "lc"))
def test_stored_checks_at_seven_gluons(compiled, loop_native, accuracy):
    process = _stored(compiled, loop_native, "gg_ggggg", accuracy, "production")
    record = check_process(
        process,
        loop_native,
        "gg_ggggg",
        accuracy,
        points=[0],
        dd_points=[0],
        rescue_points=[0, 1],
        threads=16,
    )
    # Catani's operator is the full-colour prediction.
    _assert_passed(record, catani=accuracy == "full", ward=False)
