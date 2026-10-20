# SPDX-License-Identifier: 0BSD
"""Saved projected flavour sharing against independent member instructions."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from .test_loop_bose_correlated_runtime import _source_rows
from .test_loop_native_directed import _blocks, _values
from .test_loop_physical_correlations import _close, _conjugate

pytestmark = pytest.mark.loop_physics


@pytest.fixture(scope="module")
def projected_ordered_artifacts(saved_loop_fixture):
    location = str(saved_loop_fixture("projected-ordered-flavor"))
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    verify_native_module(_rusticol)
    assert load_loop_native() is _rusticol._loop
    point_file = (
        Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    )
    points = json.loads(point_file.read_text(), parse_float=str, parse_int=str)[
        "points"
    ]
    return Path(location), _rusticol._loop, points


def _compare_complete(actual, expected, ids, key, tolerance, sectors):
    assert actual.point_ids == actual.requested_point_ids == ids
    assert expected.point_ids == expected.requested_point_ids == ids
    assert (
        actual.history_ids
        == expected.history_ids
        == ["born", "casimir", "ordered", "swapped"]
    )
    assert actual.spin_legs == expected.spin_legs == list(key)
    assert actual.observable_kind == expected.observable_kind == "directed-complex"
    assert not actual.failures and not expected.failures
    blocks = _blocks(actual)
    reference_blocks = _blocks(expected)
    # Nine exposed component/term blocks plus the independent summed totals.
    assert len(blocks) == len(reference_blocks) == 10
    for left, right in zip(blocks, reference_blocks, strict=True):
        assert left.rows == right.rows == len(ids)
        assert left.columns == right.columns
        assert left.row_ids == right.row_ids
        assert all(
            equal for _, equal in left.agreement_rows(right, tolerance, tolerance)
        )
    assert len(actual.statuses) == len(expected.statuses) == len(ids)
    for index, (left, right) in enumerate(
        zip(actual.statuses, expected.statuses, strict=True)
    ):
        assert len(left) == len(right) > 0
        assert all(row == index and status == "success" for row, status in left)
        assert all(row == index and status == "success" for row, status in right)
    shape = (len(ids), 4, sectors, 3)
    total = _values(actual.component_batches[0], shape)
    first = _values(actual.directed_first, shape)
    second = _values(actual.directed_second, shape)
    _close(total, first + second, Decimal(tolerance))
    _close(total[:, 2], _conjugate(total[:, 3]), Decimal(tolerance))
    _close(first[:, 2], _conjugate(second[:, 3]), Decimal(tolerance))
    return total


@pytest.mark.parametrize("accuracy", ("full", "lc", "nlc"))
@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance,bit",
    (
        ("double", None, "fast", "3e-9", 20),
        ("double-double", None, "generic", "3e-24", 70),
        ("arbitrary", 70, "generic", "3e-54", 170),
    ),
)
def test_saved_projected_ordered_member_complete_components(
    projected_ordered_artifacts,
    accuracy,
    arithmetic,
    precision,
    engine,
    tolerance,
    bit,
):
    root, native, points = projected_ordered_artifacts
    with localcontext() as context:
        context.prec = 120
        alpha = str(Decimal("0.118") * (1 + Decimal(2) ** -bit))
        parameters = {
            "aS": (alpha, "0"),
            "MT": ("173", "0"),
            "MU_R": ("1000", "0"),
            "MB": ("0", "0"),
            "ymb": ("0", "0"),
        }
        mapped = native.NativeVirtualRuntime(
            str(root / f"{accuracy}-mapped-artifact"),
            process="ssgg",
            arithmetic=arithmetic,
            precision=precision,
            parameters=parameters,
        )
        independent = native.NativeLoopProgram(
            str(root / f"independent-{accuracy}.pacbin"),
            arithmetic=arithmetic,
            precision=precision,
            reduction_engine=engine,
            parameters=parameters,
        )
        metadata = json.loads(mapped.metadata_json())
        assert metadata["process_id"] == "ssgg"
        assert metadata["representative_process_id"] == "ddgg"
        assert metadata["physics"]["color_accuracy"] == accuracy
        assert metadata["physics"]["correlations"]
        assert not metadata["physics"]["resolved"]
        plan = metadata["plan"]
        directed = plan["directed_projection"]
        family = directed["color_family_projection"]
        assert family["accuracy"] == accuracy
        assert [item["order"] for item in family["families"]] == [0, 1, 2]
        assert all(
            item["directed"]["color_family_projection"] == family
            for item in plan["spin_projections"]
        )
        catalog = metadata["selected_correlations"]
        assert catalog["histories"] == directed["histories"]
        assert catalog["spin_classes"] == [[3], [3, 4], [4]]
        sectors = len(plan["order_layout"]["virtual"])
        assert tuple(mapped.source_state.snapshot().overrides["aS"]) == (alpha, "0")
        for key in ((), (3,), (4,), (3, 4)):
            for size in (0, 1, 3) if arithmetic == "double" else (1,):
                batch = [points[1], points[0], points[1]][:size]
                ids = [2**64 - 1, 7, 42][:size]
                scales = ["900000", "1000000", "900000"][:size]
                sources = _source_rows(key, size)
                actual = mapped.evaluate_directed(
                    batch, scales, point_ids=ids, spin_vectors=sources
                )
                expected = independent.evaluate_directed(
                    batch, scales, point_ids=ids, spin_vectors=sources
                )
                total = _compare_complete(
                    actual, expected, ids, key, tolerance, sectors
                )
                if size and accuracy == "nlc":
                    assert any(
                        abs(value) > Decimal("1e-6")
                        for value in total[:, 2, :, :, 1].flat
                    )
                if size == 3:
                    assert (
                        actual.component_batches[0].take_rows([0]).to_decimal()
                        == actual.component_batches[0].take_rows([2]).to_decimal()
                    )
        # Correlated output is contracted. Existing ordinary signed LC output
        # remains a separate plan; this artifact does not claim resolved axes.
        with pytest.raises(ValueError, match="resolved"):
            mapped.evaluate_resolved(points[:1], ["1000000"])
