# SPDX-License-Identifier: 0BSD
"""Complete physical FFT execution against the retained exact direct schedule."""

import json
import os
from pathlib import Path

import pytest

from .test_loop_portable import _rewrite


@pytest.fixture(scope="module")
def fft_controls():
    root = os.environ.get("PYAMPLICOL_LOOP_FFT_FIXTURES")
    if not root:
        pytest.skip("retained certified full/NLC sources required")
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    assert load_loop_native() is _rusticol._loop
    verify_native_module(_rusticol)
    point_file = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/gluon_scattering_qcd.json"
    )
    points = json.loads(point_file.read_text(), parse_float=str, parse_int=str)[
        "points"
    ]
    return Path(root), _rusticol._loop, points


def _blocks(result):
    return result.born, *result.component_batches, result.totals


@pytest.mark.parametrize("accuracy", ["nlc", "full"])
@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", "3e-9"),
        ("double-double", None, "generic", "3e-25"),
        ("arbitrary", 70, "generic", "3e-55"),
    ],
)
def test_actual_complete_ordinary_resolved_and_carrier_fallback(
    fft_controls, accuracy, arithmetic, precision, engine, tolerance
):
    root, native, points = fft_controls
    options = dict(arithmetic=arithmetic, precision=precision, reduction_engine=engine)
    actual = native.NativeLoopProgram(str(root / accuracy / "fft.pacbin"), **options)
    expected = native.NativeLoopProgram(
        str(root / accuracy / "direct.pacbin"), **options
    )
    plan = json.loads(actual.metadata_json())
    for selected in (
        plan["dual_born_weights"],
        plan["helicity_projection"]["dual_born_weights"],
    ):
        assert selected["rectangular"]["kernels"]
        assert selected["rectangular"]["residuals"]
    expected_algorithm = "symmetric-group-fft" if arithmetic == "double" else "direct"
    for size in (0, 1, 3):
        chosen, ids = points[:size], [91, 7, 2**64 - 1][:size]
        scales = [str(900000 + 10000 * index) for index in range(size)]
        for resolved in (False, True):
            a = actual.evaluate(chosen, scales, point_ids=ids, resolved=resolved)
            b = expected.evaluate(chosen, scales, point_ids=ids, resolved=resolved)
            assert a.point_ids == b.point_ids == ids and not a.failures
            selection = json.loads(a.primary_color_contraction_json)
            assert selection["requested"] == "symmetric-group-fft"
            assert selection["algorithm"] == expected_algorithm
            assert (selection["regular_kernels"] > 0) == (arithmetic == "double")
            assert (selection["direct_residual_entries"] > 0) == (
                arithmetic == "double"
            )
            assert selection["reason"]
            if arithmetic != "double":
                assert (
                    selection["reason"] == "exact higher-precision direct contraction"
                )
            for left, right in zip(_blocks(a), _blocks(b), strict=True):
                assert all(
                    ok for _, ok in left.agreement_rows(right, tolerance, tolerance)
                )
    repeated = actual.evaluate(points[:1], ["900000"], point_ids=[91])
    reference = expected.evaluate(points[:1], ["900000"], point_ids=[91])
    for left, right in zip(_blocks(repeated), _blocks(reference), strict=True):
        assert all(ok for _, ok in left.agreement_rows(right, tolerance, tolerance))


def test_actual_weight_coefficients_match_independent_saved_state(fft_controls):
    root, native, _ = fft_controls
    for accuracy in ("nlc", "full"):
        directory = root / accuracy
        descriptor = (directory / "rectangle.json").read_text()
        record = json.loads(descriptor)
        columns = json.loads((directory / "columns.json").read_text())
        state = (directory / "weights.bin").read_bytes()
        constants = (directory / "constants.bin").read_bytes()
        width = sum(len(alias["columns"]) for alias in record["born_aliases"])
        for arithmetic, precision, tolerance in (
            ("double", None, "3e-12"),
            ("double-double", None, "1e-29"),
            ("arbitrary", 70, "1e-60"),
        ):
            options = dict(arithmetic=arithmetic, precision=precision)
            actual = native.DualBornWeights(
                state, descriptor, constants, columns, **options
            )
            expected = native.CoefficientEvaluator(state, **options)
            for size in (0, 1, 3, 1):
                data = [
                    (
                        f"{row + col}.12345678901234567890123456789",
                        f"{row - col}.34567890123456789012345678901",
                    )
                    for row in range(size)
                    for col in range(width)
                ]
                batch = native.NativeBatch.from_decimal(
                    data, size, width, row_ids=[91, 7, 30][:size], **options
                )
                left, right = (
                    actual.evaluate_batch(batch),
                    expected.evaluate_batch(batch),
                )
                assert left.row_ids == right.row_ids == [91, 7, 30][:size]
                assert all(
                    ok for _, ok in left.agreement_rows(right, tolerance, tolerance)
                )


@pytest.mark.parametrize("resolved", [False, True])
def test_actual_subset_rescue_reports_primary_fft_and_exact_direct_attempts(
    fft_controls, resolved
):
    root, native, points = fft_controls
    runtime = native.NativeLoopProgram(
        str(root / "nlc/fft.pacbin"), reduction_engine="fast"
    )
    result = runtime.evaluate_verified(
        [points[0], points[1], points[0]],
        ["1000000", "1e-400", "1000000"],
        [("arbitrary", 60), ("arbitrary", 80)],
        "1e-8",
        "1e-9",
        point_ids=[91, 4, 70],
        resolved=resolved,
    )
    assert result.result.point_ids == [91, 4, 70] and not result.result.failures
    assert (
        json.loads(result.result.primary_color_contraction_json)["algorithm"]
        == "symmetric-group-fft"
    )
    diagnostics = json.loads(result.diagnostics_json)
    assert [row["status"] for row in diagnostics] == ["success", "rescued", "success"]
    assert [len(row["attempts"]) for row in diagnostics] == [1, 3, 1]
    for row in diagnostics:
        assert (
            row["attempts"][0]["color_contraction"]["algorithm"]
            == "symmetric-group-fft"
        )
    assert all(
        attempt["color_contraction"]["algorithm"] == "direct"
        and attempt["color_contraction"]["reason"]
        for attempt in diagnostics[1]["attempts"][1:]
    )
    expected = native.NativeLoopProgram(
        str(root / "nlc/direct.pacbin"),
        arithmetic="arbitrary",
        precision=80,
        reduction_engine="generic",
    ).evaluate([points[1]], ["1e-400"], point_ids=[4], resolved=resolved)
    rows = expected.born.rows
    for left, right in zip(_blocks(result.result), _blocks(expected), strict=True):
        assert all(
            ok
            for _, ok in left.take_rows(
                list(range(rows, 2 * rows)), right.row_ids
            ).agreement_rows(right, "1e-8", "1e-9")
        )


def test_saved_rectangle_requires_exhaustive_nonduplicated_coverage(
    fft_controls, tmp_path
):
    root, native, _ = fft_controls
    for index, mutate in enumerate(
        (
            lambda rectangle: rectangle["residuals"].pop(),
            lambda rectangle: rectangle["residuals"].append(rectangle["residuals"][0]),
        )
    ):
        destination = tmp_path / f"corrupt-{index}.pacbin"
        _rewrite(
            root / "nlc/fft.pacbin",
            destination,
            lambda data, mutate=mutate: mutate(
                data["dual_born_weights"]["rectangular"]
            ),
        )
        with pytest.raises(ValueError):
            native.PortableLoopPlan(str(destination))


def test_repeated_resolved_export_keeps_model_roles_separate(fft_controls):
    root, native, _ = fft_controls
    for accuracy in ("nlc", "full"):
        for name in ("direct.pacbin", "fft.pacbin", "ordinary-after-resolved.pacbin"):
            plan = json.loads(
                native.PortableLoopPlan(str(root / accuracy / name)).metadata_json()
            )
            assert set(plan["model_evaluators"]) == {
                "projector",
                "assemble",
                "sum_order_values",
            }
            assert (plan.get("helicity_projection") is None) == (
                name == "ordinary-after-resolved.pacbin"
            )


def test_public_selected_bank_and_per_call_precision_diagnostics(fft_controls):
    from pyamplicol import Runtime, VirtualArithmeticSpec, VirtualStabilityPolicy

    from .test_runtime_source_binding_rescue import _close

    root, _, points = fft_controls
    policy = VirtualStabilityPolicy(
        (VirtualArithmeticSpec("arbitrary", 60),), "1e-8", "1e-9", True
    )
    for accuracy in ("nlc", "full"):
        actual = Runtime.load(root / accuracy / "fft-artifact")
        expected = Runtime.load(root / accuracy / "direct-artifact")
        info = actual.inspect()["color_contraction"]
        assert info["ordinary"]["algorithm"] == "symmetric-group-fft"
        assert info["resolved"]["algorithm"] == "symmetric-group-fft"
        for name in ("evaluate", "evaluate_resolved"):
            left = getattr(actual, name)(points[:1], mu_r2="1000000", stability=policy)
            right = getattr(expected, name)(points[:1], mu_r2="1000000")
            _close(left.born_values, right.born_values)
            _close(left.values, right.values)
            diagnostic = left.diagnostics[0]
            assert (
                diagnostic.primary_color_contraction.algorithm == "symmetric-group-fft"
            )
            assert (
                diagnostic.verification.attempts[-1].color_contraction.algorithm
                == "direct"
            )
            assert diagnostic.verification.attempts[-1].color_contraction.reason
            assert diagnostic.accuracy_established
        high = actual.evaluate(
            points[:1], mu_r2="1000000", arithmetic="arbitrary", precision=70
        )
        assert high.diagnostics[0].primary_color_contraction.algorithm == "direct"
        assert high.diagnostics[0].primary_color_contraction.reason
        actual.clear()
        assert actual.evaluate([], mu_r2=[]).valid == ()
        # A call-local arithmetic fallback does not change the handle's operation.
        assert (
            actual.inspect()["color_contraction"]["ordinary"]["algorithm"]
            == "symmetric-group-fft"
        )


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", "3e-9"),
        ("double-double", None, "generic", "3e-25"),
        ("arbitrary", 70, "generic", "3e-55"),
    ],
)
def test_portable_python_view_binds_shared_operation_without_producer_work(
    fft_controls, monkeypatch, arithmetic, precision, engine, tolerance
):
    from pyamplicol.loops.portable import PortableLoopPlan

    from .test_loop_portable import _forbid_producer

    root, native, points = fft_controls
    _forbid_producer(monkeypatch)
    options = dict(arithmetic=arithmetic, precision=precision, reduction_engine=engine)
    path = root / "nlc/fft.pacbin"
    view = PortableLoopPlan.open(path, backend=native).bind(**options)
    expected = native.NativeLoopProgram(str(root / "nlc/direct.pacbin"), **options)
    assert view.bare.dual_born_weights.algorithm == (
        "symmetric-group-fft" if arithmetic == "double" else "direct"
    )
    actual = view.evaluate_ordered_native(points[:1], ["1000000"], point_ids=[91])
    reference = expected.evaluate(points[:1], ["1000000"], point_ids=[91])
    assert actual.batch.point_ids == reference.point_ids == [91]
    for left, right in zip(_blocks(actual), _blocks(reference), strict=True):
        assert all(ok for _, ok in left.agreement_rows(right, tolerance, tolerance))
