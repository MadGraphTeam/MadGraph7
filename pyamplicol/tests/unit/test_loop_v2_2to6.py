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
from tools.loop_benchmark import pyamplicol_v2, streams
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
    # The Ward store records its massless vector legs (the stored view has no
    # model data): first, middle and last are checked.
    assert process.record["gauge_legs"] == [1, 2, 3, 4]
    assert {tuple(row["legs"]) for row in record["ward"]} == {(1, 3, 4)}
    # A required prediction that does not form fails the point (here the
    # record no longer applies at the run's MT).
    conditions = process.record["catani"]["conditions"]
    conditions[0] = [conditions[0][0], not conditions[0][1]]
    failed = check_process(
        process, loop_native, "gg_gg", "full", points=[0], dd_points=[0]
    )
    assert failed["passed"] is False
    assert [row["catani_formed"] for row in failed["checked"]] == [False]
    assert [row["passed"] for row in failed["checked"] + failed["dd"]] == [False, False]


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


# MF-8Q (the quark 2->6 rows u u~ > 6g and u u~ > u u~ 4g; append-only): the
# same harness on their lower-multiplicity analogues, Ward with the external
# gluons, and the user's Q-A ruling that a top-included 2->6 request stays
# accepted (the timed rows are top-free).


@pytest.mark.loop_nightly
@pytest.mark.parametrize("key", ("uux_uuxg", "uux_gggg"))
def test_stored_checks_on_quark_rows(compiled, loop_native, key):
    """A quark row stored from the full tree kernels: Catani's poles (native),
    E, the double pole, binary64 against double-double and Ward with
    epsilon -> p on its external gluons only (``ward.gauge_legs``)."""

    process = _stored(compiled, loop_native, key, "full", "full")
    gluons = [
        leg for leg, pdg in enumerate(streams.spec(key)["pdgs"], start=1) if pdg == 21
    ]
    assert process.record["gauge_legs"] == gluons
    record = check_process(
        process,
        loop_native,
        key,
        "full",
        points=[0, 1],
        dd_points=[0],
        ward_points=[0],
        rescue_points=[0, 1, 2, 3],
        threads=8,
    )
    _assert_passed(record, catani=True, ward=True)
    legs = {leg for row in record["ward"] for leg in row["legs"]}
    assert legs and legs <= set(gluons)


@pytest.mark.loop_nightly
def test_quark_row_at_leading_colour(compiled, loop_native):
    """u u~ > 4g at LC (production kernels): E, the double pole with the
    leading Casimirs and double-double; no Catani operator at LC."""

    process = _stored(compiled, loop_native, "uux_gggg", "lc", "production")
    record = check_process(
        process,
        loop_native,
        "uux_gggg",
        "lc",
        points=[0, 1],
        dd_points=[0],
        rescue_points=[0, 1],
        threads=8,
    )
    _assert_passed(record, catani=False, ward=False)


_TOP_CARD = """
action = "generate"
[model]
source = {model}
[process]
entries = [{{expression={process}, name="p"}}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
[color]
accuracy = "lc"
contraction = "direct"
[loop]
r2_strategy = "model"
[loop.preparation]
fixed_parameters = {{MB = "0"}}
[generation]
output = {output}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""


@pytest.mark.loop_nightly
def test_a_top_included_2to6_request_passes_the_dry_run(
    loop_sm_compiled_model, tmp_path
):
    """USER Q-A (2026-10-31): the 2->6 rows are generated and timed top-free,
    but a request with the top in the loop must stay accepted (no veto-only
    path, no multiplicity cap): u u~ > 6g without a particle veto passes the
    public dry run (``Generator.plan``: request and model preparation)."""

    import json

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    expression = "u u~ > g g g g g g"
    card = tmp_path / "card.toml"
    card.write_text(
        _TOP_CARD.format(
            model=json.dumps(str(loop_sm_compiled_model)),
            process=json.dumps(expression),
            output=json.dumps(str(tmp_path / "artifact")),
        )
    )
    config = resolve_config(card)
    assert not config.effective.generation.particle_veto
    plan = Generator(config).plan(
        ProcessSet((ProcessRequest.parse(expression, name="p"),))
    )
    assert plan.unsupported_features == ()
    assert [p.expression.lower() for p in plan.concrete_processes] == [expression]
