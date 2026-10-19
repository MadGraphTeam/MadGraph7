# SPDX-License-Identifier: 0BSD
"""Native saved helicity projections preserve all coupling and Laurent axes."""

import json
import os
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from .test_loop_artifact_writer import source_process, target_metadata, write_artifact
from .test_loop_physical_model_orders import catalogue, control
from .test_loop_source_portable import source_control

# These fixtures prepare actual source currents once; evaluation below must not
# depend on any generation object or repeated symbolic lowering.
__all__ = ["catalogue", "control", "source_control", "target_metadata"]


@pytest.fixture(scope="module")
def resolved_plan(request, tmp_path_factory):
    from pyamplicol.loops.portable import write_loop_pacbin

    retained = os.environ.get("PYAMPLICOL_RESOLVED_SOURCE_PACBIN")
    if retained:
        fixture = (
            Path(__file__).resolve().parents[2]
            / "tools/loop_reference/gluon_scattering_qcd.json"
        )
        points = json.loads(fixture.read_text(), parse_int=str, parse_float=str)[
            "points"
        ]
        return Path(retained), points
    program, _, points = request.getfixturevalue("source_control")
    path = tmp_path_factory.mktemp("resolved-loop") / "loop.pacbin"
    write_loop_pacbin(program.virtual, path, resolved=True)
    return path, points.tolist()


@pytest.fixture
def resolved_artifact(resolved_plan, tmp_path, target_metadata):
    path, points = resolved_plan
    process = source_process(path)
    process = replace(process, metadata=replace(process.metadata, resolved=True))
    destination = tmp_path / "artifact"
    write_artifact(destination, [process])
    return destination, points


def _assert_close(actual, expected, tolerance):
    with localcontext() as context:
        context.prec = 110
        a, b = Decimal(actual), Decimal(expected)
        assert abs(a - b) <= Decimal(tolerance) * max(abs(b), Decimal("1"))


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", 16, "3e-11"),
        ("double-double", 32, "3e-25"),
        ("arbitrary", 70, "1e-58"),
    ],
)
def test_saved_native_resolved_total_and_helicity_poles(
    resolved_artifact, monkeypatch, arithmetic, precision, tolerance
):
    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import ResolvedVirtualEvaluation

    from .test_loop_source_portable import _forbid_source

    path, points = resolved_artifact
    _forbid_source(monkeypatch)
    runtime = Runtime.load(path, arithmetic=arithmetic, precision=precision)
    options = dict(mu_r2=["1000000", "900000", "2100000"])
    inclusive = runtime.evaluate(points, **options)
    resolved = runtime.evaluate_resolved(points, **options)
    assert isinstance(resolved, ResolvedVirtualEvaluation)
    assert resolved.shape == (3, 1, 16, 1, 3)
    assert resolved.born_shape == (3, 1, 16, 1)
    assert resolved.virtual_color_ids == resolved.born_color_ids == ("full",)
    assert resolved.helicity_ids == tuple(f"h{i}" for i in range(16))
    assert resolved.epsilon_powers == (0, -1, -2)
    total = resolved.total()
    for p in range(3):
        _assert_close(total.born_values[p][0], inclusive.born_values[p][0], tolerance)
        for k in range(3):
            _assert_close(total.values[p][0][k], inclusive.values[p][0][k], tolerance)
        # Universal gluon double poles apply to every resolved Born helicity,
        # including Born-zero states, independently of the projection executor.
        from symbolica import E, Expression

        from pyamplicol.loops.model_expressions import parse_exact_ufo

        evaluator = Expression.evaluator_multiple(
            [E("-6/pi") * parse_exact_ufo("0.118000000000000000000000001")],
            [],
            n_cores=1,
            jit_compile=False,
        )
        factor = evaluator.evaluate_complex_with_prec([], 90)[0].to_decimal_tuple()[0]
        with localcontext() as context:
            context.prec = 110
            for h in range(16):
                born = Decimal(resolved.born_values[p][0][h][0])
                pole = resolved.values[p][0][h][0].double_pole
                _assert_close(pole, factor * born, tolerance)
    empty = runtime.evaluate_resolved([], mu_r2=[])
    assert empty.shape == (0, 1, 16, 1, 3)
    assert empty.total().values == ()
    permuted = runtime.evaluate_resolved(
        [points[2], points[0]], mu_r2=["2100000", "1000000"]
    )
    assert permuted.values == (resolved.values[2], resolved.values[0])
    assert permuted.born_values == (resolved.born_values[2], resolved.born_values[0])


def test_native_sparse_point_ids_and_capability_boundary(resolved_artifact):
    import _loop_reduction as native

    path, points = resolved_artifact
    runtime = native.NativeVirtualRuntime(str(path), process="gg")
    data = json.loads(runtime.metadata_json())
    assert data["physics"]["resolved"] is True
    result = runtime.evaluate_resolved(points[:1], ["1000000"], point_ids=[91])
    assert result.point_ids == list(range(91 * 16, 92 * 16))
    assert all(
        lane == point_id and status == "success"
        for point_id, statuses in zip(result.point_ids, result.statuses, strict=True)
        for lane, status in statuses
    )
    assert data["plan"]["helicity_projection"]["born"]


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda d: d["helicity_projection"].update(families=[]),
            "helicity family count",
        ),
        (
            lambda d: d["helicity_projection"].update(
                born=d["bare_evaluators"]["born_projector"]
            ),
            "shape",
        ),
        (lambda d: d.update(helicity_projection=None), "unreferenced"),
    ],
)
def test_resolved_plan_malformed_capabilities(
    resolved_plan, tmp_path, mutation, message
):
    import _loop_reduction as native

    from .test_loop_portable import _rewrite

    path, _ = resolved_plan
    destination = tmp_path / "malformed.pacbin"
    _rewrite(path, destination, mutation)
    with pytest.raises(ValueError, match=message):
        native.PortableLoopPlan(str(destination))


def test_public_metadata_must_match_resolved_plan(resolved_plan):
    import _loop_reduction as native

    path, _ = resolved_plan
    process = source_process(path)
    data = json.loads(native.PortableLoopPlan(str(path)).metadata_json())
    with pytest.raises(ValueError, match="resolved capability"):
        process.metadata.validate_plan(data)
    replace(process.metadata, resolved=True).validate_plan(data)
    with pytest.raises(TypeError, match="boolean"):
        replace(process.metadata, resolved=1)


@pytest.fixture(scope="module")
def mixed_plan(request, tmp_path_factory):
    from pyamplicol.loops.portable import write_loop_pacbin

    from .test_loop_physical_model_orders import _program

    retained = os.environ.get("PYAMPLICOL_RESOLVED_MIXED_PACBIN")
    if retained:
        return Path(retained)
    program = _program(
        request.getfixturevalue("control"),
        request.getfixturevalue("catalogue"),
        reduction_engine="fast",
        coefficient_engine="jit",
    )
    path = tmp_path_factory.mktemp("resolved-mixed") / "loop.pacbin"
    write_loop_pacbin(program, path, resolved=True)
    return path


def test_native_mixed_order_helicity_axes(mixed_plan):
    import _loop_reduction as native

    runtime = native.NativeLoopProgram(str(mixed_plan), reduction_engine="fast")
    plan = json.loads(runtime.metadata_json())
    fixture = Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    points = json.loads(fixture.read_text(), parse_float=str, parse_int=str)["points"][
        :2
    ]
    sectors = plan["order_layout"]["virtual"]
    assert len(sectors) == 3
    assert [item["order"] for item in sectors] == [
        [["QCD", 2], ["QED", 4]],
        [["QCD", 4], ["QED", 2]],
        [["QCD", 6]],
    ]
    options = dict(
        scales_squared=["1000000", "900000"], model_values=["0.118", "132.507", "173"]
    )
    inclusive = runtime.evaluate(points, **options)
    resolved = runtime.evaluate(points, resolved=True, **options)
    helicities = len(plan["helicities"])
    assert resolved.point_ids == list(range(len(points) * helicities))
    with localcontext() as context:
        context.prec = 70
        for single, summed in zip(
            [resolved.born, *resolved.component_batches],
            [inclusive.born, *inclusive.component_batches],
            strict=True,
        ):
            width = single.columns
            values = [complex(float(re), float(im)) for re, im in single.to_decimal()]
            reference = [
                complex(float(re), float(im)) for re, im in summed.to_decimal()
            ]
            # Independently reduce the helicity rows column by column. This
            # checks every squared order, epsilon and local completion component.
            for p in range(len(points)):
                for column in range(width):
                    actual = sum(
                        values[(p * helicities + h) * width + column]
                        for h in range(helicities)
                    )
                    expected = reference[p * width + column]
                    assert abs(actual - expected) < 3e-11 * max(1, abs(expected))


def test_resolved_failure_names_physical_point_and_helicity(resolved_artifact):
    from types import SimpleNamespace

    from pyamplicol import Runtime
    from pyamplicol.api.errors import EvaluationError

    path, points = resolved_artifact
    runtime = Runtime.load(path)
    statuses = [[] for _ in range(32)]
    statuses[19] = [(19, "failed")]
    backend = runtime._backend
    backend._native = SimpleNamespace(
        arithmetic="double",
        precision=None,
        evaluate_resolved=lambda *args, **kwargs: SimpleNamespace(
            point_ids=list(range(32)),
            r2_strategy="model",
            statuses=statuses,
        ),
    )
    with pytest.raises(EvaluationError, match="'point': 1, 'helicity': 'h3'"):
        runtime.evaluate_resolved(points[:2], mu_r2="1000000")


def test_resolved_parameter_update_clear_and_retained_results(resolved_artifact):
    from pyamplicol import Runtime

    path, points = resolved_artifact
    runtime = Runtime.load(path)
    first = runtime.evaluate_resolved(points[:1], mu_r2="1000000")
    retained = first.values
    runtime.set_model_parameters({"aS": "0.12"})
    runtime.clear()
    second = runtime.evaluate_resolved(points[:1], mu_r2="1000000")
    assert first.values == retained
    ratio = 0.12 / 0.118
    for h in range(16):
        _assert_close(
            second.born_values[0][0][h][0],
            first.born_values[0][0][h][0] * ratio**2,
            "3e-11",
        )
        for pole in range(3):
            _assert_close(
                second.values[0][0][h][0][pole],
                first.values[0][0][h][0][pole] * ratio**3,
                "3e-11",
            )


def test_native_nonfinite_resolved_projection_retains_physical_id(
    resolved_plan, tmp_path
):
    import io

    import _loop_reduction as native
    from symbolica import E, S

    from pyamplicol.generation.evaluator_container import (
        PacbinMemberSource,
        PacbinReader,
        write_pacbin_atomic,
    )
    from pyamplicol.loops.physical import _saved

    path, points = resolved_plan
    plan = json.loads(native.PortableLoopPlan(str(path)).metadata_json())
    width = sum(
        len(plan["schedules"][i]["output_slots"]) for i in plan["born"]["schedules"]
    )
    parameters = [S(f"resolved_nonfinite_test::born{i}") for i in range(width)]
    # A real native projection fault at zero strong coupling. The descriptor
    # remains well-shaped; only execution can expose division by a zero Born
    # current. This avoids depending on roundoff in structural helicity zeros.
    faulty = _saved([E("1") / parameters[0]], parameters)
    state = plan["helicity_projection"]["born"]
    destination = tmp_path / "nonfinite-projection.pacbin"
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
        str(destination), reduction_engine="fast", parameters={"aS": ["0", "0"]}
    )
    result = runtime.evaluate(
        points[:1], ["1000000"], point_ids=[91], resolved=True, partial=True
    )
    assert result.point_ids == [] and result.born.rows == 0
    assert result.failures and {f.point_id for f in result.failures} == {91}
    assert all(
        f.stage == "Born projection" and f.helicity in range(16)
        for f in result.failures
    )
    with pytest.raises(ValueError, match="point_id: 91"):
        runtime.evaluate(points[:1], ["1000000"], point_ids=[91], resolved=True)
