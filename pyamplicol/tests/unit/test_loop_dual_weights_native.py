# SPDX-License-Identifier: 0BSD
"""Saved shared weights retain the original physical schedule as an oracle."""

import json
import os
from pathlib import Path

import pytest

from .test_loop_portable import _rewrite


@pytest.fixture(scope="module")
def weight_artifacts():
    root = os.environ.get("PYAMPLICOL_DUAL_WEIGHT_FIXTURES")
    if not root:
        pytest.skip("set PYAMPLICOL_DUAL_WEIGHT_FIXTURES to retained physical controls")
    native = pytest.importorskip("_loop_reduction")
    path = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/gluon_scattering_qcd.json"
    )
    points = json.loads(path.read_text(), parse_float=str, parse_int=str)["points"]
    return native, Path(root), points


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", "3e-10"),
        ("double-double", None, "generic", "3e-25"),
        ("arbitrary", 70, "generic", "3e-55"),
    ],
)
def test_saved_weighted_and_original_complete_observables(
    weight_artifacts, arithmetic, precision, engine, tolerance
):
    native, root, points = weight_artifacts
    options = dict(arithmetic=arithmetic, precision=precision, reduction_engine=engine)
    weighted = native.NativeLoopProgram(str(root / "photon-weighted.pacbin"), **options)
    original = native.NativeLoopProgram(str(root / "photon-original.pacbin"), **options)
    directed_weighted = native.NativeLoopProgram(
        str(root / "photon-weighted-directed.pacbin"), **options
    )
    directed_original = native.NativeLoopProgram(
        str(root / "photon-original-directed.pacbin"), **options
    )
    vector = [["0.1", "0.2"], ["0.8", "-0.3"], ["0.4", "1"], ["-0.6", "0.5"]]
    for size in (0, 1, 5, 3):
        chosen = [points[index % len(points)] for index in range(size)]
        scales = [str(900000 + 10000 * index) for index in range(size)]
        ids = [91, 7, 30, 150, 2][:size]
        calls = [("evaluate", {}), ("evaluate_resolved", {}), ("evaluate_directed", {})]
        calls.extend(
            (
                "evaluate_directed",
                {"spin_sources": {leg: [vector] * size for leg in legs}},
            )
            for legs in ((3,), (3, 4))
        )
        for method, extra in calls:
            left_runtime, right_runtime = (
                (directed_weighted, directed_original)
                if method == "evaluate_directed"
                else (weighted, original)
            )
            a = getattr(left_runtime, method)(chosen, scales, point_ids=ids, **extra)
            b = getattr(right_runtime, method)(chosen, scales, point_ids=ids, **extra)
            aa = [a.born, *a.component_batches, a.totals]
            bb = [b.born, *b.component_batches, b.totals]
            if a.directed_first is not None:
                aa.extend((a.directed_first, a.directed_second))
                bb.extend((b.directed_first, b.directed_second))
            assert a.point_ids == b.point_ids == ids
            for left, right in zip(aa, bb, strict=True):
                assert all(
                    ok for _, ok in left.agreement_rows(right, tolerance, tolerance)
                )


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda w: w.update(kind="unknown"), "unsupported dual Born"),
        (lambda w: w["columns"].append(w["columns"][0]), "column/family"),
        (lambda w: w["columns"][0].__setitem__(0, 999), "key out of range"),
        (lambda w: w["columns"][0].__setitem__(1, 999), "key out of range"),
        (
            lambda w: w["columns"][0].__setitem__(2, len(w["term_graphs"])),
            "key out of range",
        ),
        (lambda w: w["term_graphs"].__setitem__(0, 999), "column/family"),
        (lambda w: w["family_columns"].pop(), "column/family"),
        (lambda w: w["family_columns"][0].pop(), "family selection"),
    ],
)
def test_native_weight_descriptor_rejects_inconsistent_keys_and_selections(
    weight_artifacts, tmp_path, mutate, message
):
    native, root, _ = weight_artifacts
    target = tmp_path / "invalid.pacbin"
    _rewrite(
        root / "photon-weighted.pacbin",
        target,
        lambda data: mutate(data["dual_born_weights"]),
    )
    with pytest.raises(ValueError, match=message):
        native.PortableLoopPlan(str(target))


@pytest.mark.parametrize("resolved", [False, True])
def test_weight_stage_failure_replays_surviving_original_ids(
    weight_artifacts, tmp_path, resolved
):
    import io

    from symbolica import E, S

    from pyamplicol.generation.evaluator_container import (
        PacbinMemberKind,
        PacbinMemberSource,
        PacbinReader,
        write_pacbin_atomic,
    )
    from pyamplicol.loops.physical import _saved
    from pyamplicol.loops.portable import _PLAN_MEMBER

    native, root, points = weight_artifacts
    with PacbinReader.open(root / "photon-weighted.pacbin") as reader:
        payloads = {
            item.logical_path: (
                item.kind,
                reader.read_member(item.logical_path, length=item.length),
            )
            for item in reader.members
        }
    data = json.loads(payloads[_PLAN_MEMBER][1])
    # Test-only faults: one Born output carries E_in - 500; the weight map
    # divides by that column. Earlier currents/Born projection stay finite.
    energy = data["bound_positions"]["loop_recurrence::p0_0"]
    group = data["born"]
    graph = next(
        i for i, columns in enumerate(group["input_columns"]) if energy in columns
    )
    schedule = data["schedules"][group["schedules"][graph]]
    count, width = schedule["parameter_count"], len(schedule["output_slots"])
    parameters = [S(f"dual_weight_failure::x{i}") for i in range(count)]
    state = "evaluators/test-born-energy.state"
    payloads[state] = (
        PacbinMemberKind.SYMBOLICA_EXACT_STATE,
        _saved(
            [
                parameters[group["input_columns"][graph].index(energy)] - 500,
                *[E("0")] * (width - 1),
            ],
            parameters,
        ),
    )
    schedule.update(
        states=[state],
        input_slots=[list(range(count))],
        output_slots=list(range(count, count + width)),
    )
    widths = [len(data["schedules"][i]["output_slots"]) for i in group["schedules"]]
    born = [S(f"dual_weight_failure::b{i}") for i in range(sum(widths))]
    for projection in (data, data["helicity_projection"]):
        weights = projection["dual_born_weights"]
        payloads[weights["state"]] = (
            PacbinMemberKind.SYMBOLICA_EXACT_STATE,
            _saved(
                [
                    E("1") / born[sum(widths[:graph])],
                    *[E("0")] * (len(weights["columns"]) - 1),
                ],
                born,
            ),
        )
    used = set()

    def references(value):
        if isinstance(value, dict):
            for item in value.values():
                references(item)
        elif isinstance(value, list):
            for item in value:
                references(item)
        elif isinstance(value, str) and value in payloads:
            used.add(value)

    references(data)
    payloads[_PLAN_MEMBER] = (
        PacbinMemberKind.LOOP_RUNTIME_PLAN,
        json.dumps(data).encode(),
    )
    path = tmp_path / "weight-stage-fault.pacbin"
    write_pacbin_atomic(
        path,
        [
            PacbinMemberSource(name, kind, io.BytesIO(value))
            for name, (kind, value) in payloads.items()
            if name == _PLAN_MEMBER or name in used
        ],
    )
    runtime = native.NativeLoopProgram(str(path), reduction_engine="fast")
    result = runtime.evaluate(
        [points[1], points[0], points[1]],
        ["1000000"] * 3,
        point_ids=[91, 7, 30],
        resolved=resolved,
        partial=True,
    )
    assert result.point_ids == [91, 30]
    assert result.failures and {failure.point_id for failure in result.failures} == {7}
    assert all(
        failure.stage == "dual Born weights" and failure.family_index is None
        for failure in result.failures
    )
    replay = runtime.evaluate(
        [points[1], points[1]], ["1000000"] * 2, point_ids=[91, 30], resolved=resolved
    )
    for left, right in zip(
        [result.born, *result.component_batches],
        [replay.born, *replay.component_batches],
        strict=True,
    ):
        assert left.to_decimal() == right.to_decimal()
