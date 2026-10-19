# SPDX-License-Identifier: 0BSD
"""One saved joint-source CLI control, compared with the public grouped API."""

import io
import json
import os
from pathlib import Path

import pytest

from pyamplicol import CorrelatedRequest, Runtime
from pyamplicol.api.errors import CompatibilityError
from pyamplicol.cli import run_cli
from pyamplicol.cli.main import write_result

from .test_runtime_source_binding_rescue import POINT


@pytest.fixture(scope="module")
def artifact():
    path = os.environ.get("PYAMPLICOL_CLI_CORRELATED_ARTIFACT")
    if path is None:
        pytest.skip(
            "set PYAMPLICOL_CLI_CORRELATED_ARTIFACT to a saved joint-source artifact"
        )
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    assert load_loop_native() is _rusticol._loop
    verify_native_module(_rusticol)
    return Path(path)


def test_saved_grouped_cli_matches_api_with_independent_failure_masks(
    artifact, tmp_path
):
    a = [0, ["0.800000000000000000000000000000000001", "-0.16"], 0, ["-0.6", "0.12"]]
    b = [0, 0, ["1", "0.2"], 0]
    overflow = [0, ["1e400", "0"], 0, 0]
    requests = {
        "physical": CorrelatedRequest("born", {}),
        "single": CorrelatedRequest("born", {3: a}),
        "joint": CorrelatedRequest("born", {3: a, 4: b}),
        "zero": CorrelatedRequest("born", {3: [0, 0, 0, 0]}),
        "failing": CorrelatedRequest("born", {3: [a, overflow, a], 4: b}),
    }
    points, request_file = tmp_path / "points.json", tmp_path / "requests.json"
    points.write_text(json.dumps([POINT] * 3))
    request_file.write_text(
        json.dumps(
            {
                label: {
                    "color_correlation": request.color_correlation,
                    "spin_vectors": request.spin_vectors,
                }
                for label, request in requests.items()
            }
        )
    )
    output, errors = io.StringIO(), io.StringIO()
    arguments = (
        "evaluate",
        str(artifact),
        "--momenta",
        str(points),
        "--mu-r2",
        "1000000",
        "--correlated-requests",
        str(request_file),
        "--on-failure",
        "return",
        "--json",
    )
    assert run_cli(arguments, stdout=output, stderr=errors) == 0, errors.getvalue()
    actual = json.loads(output.getvalue())
    runtime = Runtime.load(artifact)
    expected = runtime.evaluate_correlated_many(
        [POINT] * 3, requests, mu_r2="1000000", on_failure="return"
    )
    reference = io.StringIO()
    write_result(expected, format="json", stream=reference)
    assert actual == json.loads(reference.getvalue())
    assert actual["failing"]["values"][1] is None
    assert {
        failure["point_id"]
        for failure in actual["failing"]["diagnostics"][1]["failures"]
    } == {1}
    assert all(row is not None for row in actual["physical"]["values"])
    assert all(
        part == 0
        for row in actual["zero"]["values"]
        for sector in row
        for value in sector
        for part in value.values()
    )
    assert actual["joint"]["shape"] == [3, 1, 3]
    for row in (0, 2):
        assert actual["failing"]["values"][row] == actual["joint"]["values"][row]
    with pytest.raises(CompatibilityError):
        runtime.evaluate_correlated_many(
            [], {"bad": CorrelatedRequest("missing")}, mu_r2=[]
        )
    points.write_text("[]")
    request_file.write_text(
        '{"physical": {}, "joint": {"spin_vectors": '
        '{"3": [0,"0.8",0,"-0.6"], "4": [0,0,1,0]}}}'
    )
    output, errors = io.StringIO(), io.StringIO()
    assert run_cli(arguments, stdout=output, stderr=errors) == 0, errors.getvalue()
    empty = json.loads(output.getvalue())
    assert set(empty) == {"physical", "joint"}
    assert all(
        value["shape"] == [0, 1, 3] and value["values"] == []
        for value in empty.values()
    )
