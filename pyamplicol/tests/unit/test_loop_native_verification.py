# SPDX-License-Identifier: 0BSD
"""Real saved-source execution through explicit native cross-precision rescue."""

import json
from dataclasses import replace

import pytest

from .test_loop_artifact_writer import source_process, target_metadata, write_artifact
from .test_loop_native_failures import source_plan

__all__ = ["source_plan", "target_metadata"]


def test_exact_scale_subset_rescue_and_snapshot_updates(source_plan):
    import _loop_reduction as native

    path, points = source_plan
    runtime = native.NativeLoopProgram(str(path), reduction_engine="fast")
    result = runtime.evaluate_verified(
        [points[0], points[1], points[0]],
        ["1000000", "1e-400", "1000000"],
        [("arbitrary", 60), ("arbitrary", 80)],
        "1e-8",
        "1e-9",
        point_ids=[91, 4, 70],
    )
    assert result.result.point_ids == [91, 4, 70]
    assert result.result.failures == []
    diagnostics = json.loads(result.diagnostics_json)
    assert [x["status"] for x in diagnostics] == ["success", "rescued", "success"]
    assert [len(x["attempts"]) for x in diagnostics] == [1, 3, 1]
    rescued = diagnostics[1]
    assert rescued["accuracy_established"]
    assert rescued["attempts"][0]["failures"][0]["point_id"] == 4
    assert rescued["attempts"][-1]["agreement"] is True
    assert rescued["attempts"][-1]["conversion_agreement"] is True
    # Independent execution confirms the retry used the exact tiny positive
    # scale, which had underflowed in binary64, and returned requested f64 data.
    expected = runtime.rebind(arithmetic="arbitrary", precision=80).evaluate(
        [points[1]], ["1e-400"], point_ids=[4]
    )
    assert result.result.born.precision_bits == 53
    for batch, reference in zip(
        [result.result.born, *result.result.component_batches],
        [expected.born, *expected.component_batches],
        strict=True,
    ):
        assert all(
            ok
            for _, ok in batch.take_rows([1]).agreement_rows(reference, "1e-8", "1e-9")
        )
    # The same cached higher owner must recompute changed/reverted exact inputs;
    # its own previous mutable snapshot must never supply stale overrides.
    baseline = runtime.source_state.snapshot().overrides.get("aS", ["0.118", "0"])
    for value in (["0.12", "0"], baseline):
        snapshot = runtime.source_state.update({"aS": value})
        verified = runtime.evaluate_verified(
            [points[0]],
            ["1000000"],
            [("double-double", None)],
            "1e-8",
            "1e-9",
            verify_successful=True,
        )
        assert verified.result.failures == []
        assert verified.result.source_revision == snapshot.revision
        assert json.loads(verified.diagnostics_json)[0]["accuracy_established"]
        reference = runtime.evaluate([points[0]], ["1000000"])
        assert verified.result.born.to_decimal() == reference.born.to_decimal()


@pytest.mark.parametrize("resolved", [False, True])
def test_verification_empty_all_failed_and_unattainable_output(source_plan, resolved):
    import _loop_reduction as native

    path, points = source_plan
    runtime = native.NativeLoopProgram(str(path), reduction_engine="fast")
    policy = ([("arbitrary", 50), ("arbitrary", 70)], "1e-25", "1e-30")
    empty = runtime.evaluate_verified([], [], *policy, resolved=resolved)
    assert empty.result.point_ids == empty.result.failures == []
    assert json.loads(empty.diagnostics_json) == []
    # Two AP evaluations can agree while their binary64 output cannot satisfy
    # the user's requested bound. Such a point must stay unavailable.
    result = runtime.evaluate_verified(
        [points[0]],
        ["1e-400"],
        *policy,
        point_ids=[91],
        resolved=resolved,
    )
    assert result.result.point_ids == [] and result.result.born.rows == 0
    assert {f.point_id for f in result.result.failures} == {91}
    diagnostic = json.loads(result.diagnostics_json)[0]
    assert diagnostic["status"] == "accuracy-not-established"
    assert not diagnostic["accuracy_established"]
    assert diagnostic["attempts"][-1]["agreement"] is True
    assert diagnostic["attempts"][-1]["conversion_agreement"] is False
    if resolved:
        accepted = runtime.evaluate_verified(
            [points[0]],
            ["1e-400"],
            policy[0],
            "1e-8",
            "1e-9",
            point_ids=[91],
            resolved=True,
        )
        h = len(json.loads(runtime.metadata_json())["helicities"])
        assert accepted.result.point_ids == [91 * h + i for i in range(h)]
        assert accepted.result.failures == []
        assert json.loads(accepted.diagnostics_json)[0]["status"] == "rescued"


def test_actual_precision_growth_and_tolerances_are_required(source_plan):
    import _loop_reduction as native

    path, _ = source_plan
    runtime = native.NativeLoopProgram(str(path), arithmetic="double-double")
    for higher in (
        [("double-double", None)],
        [("arbitrary", 20)],
        [("arbitrary", 60), ("arbitrary", 40)],
    ):
        with pytest.raises(ValueError, match="strictly increase actual precision bits"):
            runtime.evaluate_verified([], [], higher, "1e-8", "0")
    for rtol, atol in (("0", "0"), ("NaN", "0"), ("-1e-5", "0")):
        with pytest.raises(ValueError, match="tolerances"):
            runtime.evaluate_verified([], [], [("arbitrary", 60)], rtol, atol)


def test_public_typed_verification_and_clear(source_plan, tmp_path, target_metadata):
    from pyamplicol import (
        EvaluationError,
        Runtime,
        VirtualArithmeticSpec,
        VirtualStabilityPolicy,
        VirtualVerificationDiagnostic,
    )

    path, points = source_plan
    process = source_process(path)
    process = replace(process, metadata=replace(process.metadata, resolved=True))
    destination = tmp_path / "artifact"
    write_artifact(destination, [process])
    runtime = Runtime.load(destination)
    policy = VirtualStabilityPolicy(
        (VirtualArithmeticSpec("double-double"),), "1e-8", "1e-9", True
    )
    for operation in (runtime.evaluate, runtime.evaluate_resolved):
        result = operation([points[0]], mu_r2="1000000", stability=policy)
        assert result.diagnostics[0].accuracy_established
        assert isinstance(
            result.diagnostics[0].verification, VirtualVerificationDiagnostic
        )
        assert result.diagnostics[0].verification.attempts[-1].agreement
    runtime.clear()
    assert runtime.evaluate([], mu_r2=[], stability=policy).diagnostics == ()
    # One successful higher result alone does not rescue a failed primary.
    with pytest.raises(EvaluationError) as error:
        runtime.evaluate(
            [points[0]],
            mu_r2="1e-400",
            stability=VirtualStabilityPolicy(
                (VirtualArithmeticSpec("arbitrary", 60),),
                "1e-8",
                "1e-9",
            ),
        )
    assert error.value.failures and error.value.verification
    assert error.value.verification[0].status == "accuracy-not-established"


def test_all_precision_attempts_fail_without_placeholder_values(source_plan, tmp_path):
    import io

    import _loop_reduction as native
    from symbolica import E, S

    from pyamplicol.generation.evaluator_container import (
        PacbinMemberSource,
        PacbinReader,
        write_pacbin_atomic,
    )
    from pyamplicol.loops.physical import _saved

    path, points = source_plan
    plan = json.loads(native.PortableLoopPlan(str(path)).metadata_json())
    width = sum(
        len(plan["schedules"][i]["output_slots"]) for i in plan["born"]["schedules"]
    )
    parameters = [S(f"verification_fault::born{i}") for i in range(width)]
    # Explicit transport injection: a Born projection divides by the identically
    # zero current at aS=0, so every carrier fails at the same scheduled stage.
    faulty = _saved([E("1") / parameters[0]], parameters)
    state = plan["helicity_projection"]["born"]
    destination = tmp_path / "verification-fault.pacbin"
    with PacbinReader.open(path) as reader:
        members = [
            PacbinMemberSource(
                member.logical_path,
                member.kind,
                io.BytesIO(
                    faulty
                    if member.logical_path == state
                    else reader.read_member(member.logical_path, length=member.length)
                ),
            )
            for member in reader.members
        ]
    write_pacbin_atomic(destination, members)
    runtime = native.NativeLoopProgram(
        str(destination),
        reduction_engine="fast",
        parameters={"aS": ["0", "0"]},
    )
    result = runtime.evaluate_verified(
        points[:1],
        ["1000000"],
        [("double-double", None), ("arbitrary", 60)],
        "1e-8",
        "1e-9",
        point_ids=[91],
        resolved=True,
    )
    assert result.result.point_ids == [] and result.result.totals.rows == 0
    assert {failure.point_id for failure in result.result.failures} == {91}
    diagnostic = json.loads(result.diagnostics_json)[0]
    assert diagnostic["status"] == "failed" and not diagnostic["accuracy_established"]
    assert len(diagnostic["attempts"]) == 3
    assert all(not attempt["numerical_success"] for attempt in diagnostic["attempts"])
    assert all(
        failure["helicity"] in range(16)
        for attempt in diagnostic["attempts"]
        for failure in attempt["failures"]
    )


@pytest.mark.parametrize(
    "arithmetic,precision,higher,rtol,atol",
    [
        ("double-double", None, 60, "1e-20", "1e-25"),
        ("arbitrary", 40, 70, "1e-30", "1e-35"),
    ],
)
def test_verification_preserves_requested_high_precision_carrier(
    source_plan,
    arithmetic,
    precision,
    higher,
    rtol,
    atol,
):
    import _loop_reduction as native

    path, points = source_plan
    runtime = native.NativeLoopProgram(
        str(path), arithmetic=arithmetic, precision=precision
    )
    result = runtime.evaluate_verified(
        points[:1],
        ["1000000"],
        [("arbitrary", higher)],
        rtol,
        atol,
        verify_successful=True,
        point_ids=[70],
    )
    assert result.result.failures == []
    assert result.result.born.precision_bits == runtime.precision_bits
    diagnostic = json.loads(result.diagnostics_json)[0]
    assert diagnostic["status"] == "success" and diagnostic["accuracy_established"]
    assert len(diagnostic["attempts"]) == 2
    assert diagnostic["agreeing_bits"][0] == runtime.precision_bits
