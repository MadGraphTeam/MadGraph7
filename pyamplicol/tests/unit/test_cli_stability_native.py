# SPDX-License-Identifier: 0BSD
"""CLI controls over retained source artifacts; no model or graph construction."""

import io
import json
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from pyamplicol.cli import run_cli

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def retained(saved_loop_fixtures):
    directory = str(saved_loop_fixtures)
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    assert load_loop_native() is _rusticol._loop
    verify_native_module(_rusticol)
    root = Path(directory)
    data = json.loads(
        (root / "source-colour-components/gg-lc-expected.json").read_text()
    )
    return root, data


def _close(actual, expected):
    with localcontext() as context:
        context.prec = 80
        a, b = Decimal(str(actual)), Decimal(str(expected))
        assert abs(a - b) <= Decimal("2e-8") * max(abs(a), abs(b)) + Decimal("1e-9")


def test_cli_policy_rescues_binding_before_evaluation(retained, tmp_path):
    root, data = retained
    points = tmp_path / "points.json"
    points.write_text(json.dumps(data["points"][:1]))
    arguments = (
        "evaluate",
        str(root / "source-binding-rescue/resolved-artifact"),
        "--momenta",
        str(points),
        "--mu-r2",
        "1000000",
        "--json",
    )
    output, errors = io.StringIO(), io.StringIO()
    assert run_cli(arguments, stdout=output, stderr=errors) == 2
    assert not output.getvalue()
    output, errors = io.StringIO(), io.StringIO()
    assert (
        run_cli(
            (
                *arguments,
                "--stability-higher",
                "arbitrary:60",
                "--stability-higher",
                "arbitrary:80",
                "--stability-relative-tolerance",
                "1e-8",
                "--stability-absolute-tolerance",
                "1e-9",
            ),
            stdout=output,
            stderr=errors,
        )
        == 0
    ), errors.getvalue()
    result = json.loads(output.getvalue())
    assert result["shape"] == [1, 1, 3]
    diagnostic = result["diagnostics"][0]
    assert diagnostic["status"] == "rescued" and diagnostic["accuracy_established"]
    attempts = diagnostic["verification"]["attempts"]
    assert len(attempts) == 3
    assert attempts[0]["failures"][0]["stage"] == "real model input type"
    assert attempts[-1]["agreement"] and attempts[-1]["conversion_agreement"]
    _close(result["born_values"][0][0], data["expected"]["born"][0][0])
    for coefficient, value in enumerate(result["values"][0][0]):
        _close(value, data["expected"]["virtual"][coefficient][0])


def test_cli_resolved_partial_json_preserves_mixed_points(retained, tmp_path):
    root, data = retained
    singular = [["0"] * 4 for _ in data["points"][0]]
    points = tmp_path / "mixed.json"
    points.write_text(json.dumps([data["points"][1], singular, data["points"][0]]))
    scales = tmp_path / "scales.json"
    scales.write_text(json.dumps([data["scales"][1], "1000000", data["scales"][0]]))
    output, errors = io.StringIO(), io.StringIO()
    assert (
        run_cli(
            (
                "evaluate",
                str(root / "source-colour-components/public-artifact"),
                "--momenta",
                str(points),
                "--mu-r2-file",
                str(scales),
                "--resolved",
                "--on-failure",
                "return",
                "--json",
            ),
            stdout=output,
            stderr=errors,
        )
        == 0
    ), errors.getvalue()
    result = json.loads(output.getvalue())
    assert result["shape"] == [3, 1, 16, 6, 3]
    assert result["values"][1] is result["born_values"][1] is None
    assert [lane["status"] for lane in result["diagnostics"]] == [
        "success",
        "failed",
        "success",
    ]
    assert {
        failure["point_id"] for failure in result["diagnostics"][1]["failures"]
    } == {1}
    with localcontext() as context:
        context.prec = 80
        for row, original in ((0, 1), (2, 0)):
            born = sum(
                Decimal(str(value))
                for helicity in result["born_values"][row][0]
                for value in helicity
            )
            _close(born, data["expected"]["born"][original][0])
            for coefficient in range(3):
                total = sum(
                    Decimal(str(component[coefficient]))
                    for helicity in result["values"][row][0]
                    for component in helicity
                )
                _close(
                    total, data["expected"]["virtual"][3 * original + coefficient][0]
                )
