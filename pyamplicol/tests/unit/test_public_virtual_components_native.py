# SPDX-License-Identifier: 0BSD
"""Actual native saved-source LC consumer, independent from cold generation."""

import json
import os
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from pyamplicol import Runtime
from pyamplicol.api import VirtualArithmeticSpec, VirtualStabilityPolicy
from pyamplicol.api.errors import EvaluationError


@pytest.fixture(scope="module")
def artifact():
    path = os.environ.get("PYAMPLICOL_SOURCE_PUBLIC_LC_ARTIFACT")
    if not path:
        pytest.skip(
            "set PYAMPLICOL_SOURCE_PUBLIC_LC_ARTIFACT for actual signed anchors"
        )
    pytest.importorskip("_loop_reduction")
    source = Path(path).parent / "gg-lc-expected.json"
    return Path(path), json.loads(source.read_text())


def _close(left, right, rtol):
    with localcontext() as context:
        context.prec = 100
        left, right = Decimal(left), Decimal(right)
        assert abs(left - right) <= Decimal(rtol) * max(
            abs(left), abs(right), Decimal("1e-20")
        )


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", 16, "3e-10"),
        ("double-double", 16, "3e-23"),
        ("arbitrary", 70, "3e-54"),
    ],
)
def test_native_lc_public_components_reconstruct_inclusive(
    artifact, arithmetic, precision, tolerance
):
    path, expected = artifact
    runtime = Runtime.load(path, arithmetic=arithmetic, precision=precision)
    options = {"mu_r2": expected["scales"]}
    inclusive = runtime.evaluate(expected["points"], **options)
    resolved = runtime.evaluate_resolved(expected["points"], **options)
    total = resolved.total()
    assert runtime.physics.contribution_kind == "virtual"
    assert runtime.physics.color_accuracy == inclusive.color_accuracy == "lc"
    assert runtime.physics.color_flow_ids == ()
    assert len(runtime.physics.born_anchored_color_components) == 6
    assert (
        resolved.born_color_ids
        == resolved.virtual_color_ids
        == runtime.physics.color_ids
    )
    assert resolved.shape == (2, 1, 16, 6, 3)
    for p in range(2):
        _close(total.born_values[p][0], inclusive.born_values[p][0], tolerance)
        for pole in range(3):
            _close(total.values[p][0][pole], inclusive.values[p][0][pole], tolerance)
    empty = runtime.evaluate_resolved([], mu_r2=[])
    assert empty.shape == (0, 1, 16, 6, 3)
    assert empty.total().shape == (0, 1, 3)


def test_native_lc_public_resolved_subset_rescue(artifact):
    path, expected = artifact
    runtime = Runtime.load(path)
    points = [expected["points"][0]] * 3
    scales = ["1000000", "1e-400", "1000000"]
    with pytest.raises(EvaluationError) as caught:
        runtime.evaluate_resolved(points, mu_r2=scales)
    assert {failure.point_id for failure in caught.value.failures} == {1}
    result = runtime.evaluate_resolved(
        points,
        mu_r2=scales,
        stability=VirtualStabilityPolicy(
            (
                VirtualArithmeticSpec("arbitrary", 60),
                VirtualArithmeticSpec("arbitrary", 80),
            ),
            "1e-8",
            "1e-9",
        ),
    )
    assert result.shape == (3, 1, 16, 6, 3)
    assert [d.status for d in result.diagnostics] == ["success", "rescued", "success"]
    assert [len(d.verification.attempts) for d in result.diagnostics] == [1, 3, 1]
    history = result.diagnostics[1].verification
    assert history.point_id == 1 and history.attempts[-1].agreement
    assert history.attempts[-1].conversion_agreement
    reference = runtime.evaluate_resolved(
        [points[1]], mu_r2=[scales[1]], arithmetic="arbitrary", precision=80
    )
    for h in range(16):
        for c in range(6):
            _close(
                result.born_values[1][0][h][c],
                reference.born_values[0][0][h][c],
                "1e-8",
            )
            for pole in range(3):
                _close(
                    result.values[1][0][h][c][pole],
                    reference.values[0][0][h][c][pole],
                    "1e-8",
                )
