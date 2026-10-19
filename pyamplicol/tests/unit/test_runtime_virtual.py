# SPDX-License-Identifier: 0BSD
"""Public virtual artifact transport, exact ingress and arithmetic selection."""

import json
import os
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.api.errors import CompatibilityError
from pyamplicol.runtime.virtual import _inputs, _parameters


def test_exact_input_transport_and_rejection():
    precise = Decimal("0.118000000000000000000000000000000000001")
    assert _parameters({"aS": precise, "mass": ("1.2", "1e-80")}) == {
        "aS": [str(precise), "0"],
        "mass": ["1.2", "1E-80"],
    }
    assert _inputs([[[precise, "0", 0, 0.0]]], ["1e80"], 1) == (
        [[[str(precise), "0", "0", "0.0"]]],
        ["1E+80"],
    )
    assert _inputs([], "1", 1) == ([], [])
    assert _inputs(np.array([[[1, 0, 0, 1]]]), np.array(["1e5"]), 1) == (
        [[["1", "0", "0", "1"]]],
        ["1E+5"],
    )
    for bad in (True, complex(1, 0), "nan", "inf", "bad"):
        with pytest.raises((ValueError, TypeError)):
            _inputs([[[bad, 0, 0, 0]]], "1", 1)
    with pytest.raises(CompatibilityError, match="mu_r2"):
        _inputs([], None, 1)
    for bad in (0, -1, True, "NaN", "Infinity"):
        with pytest.raises((ValueError, TypeError)):
            _inputs([], bad, 1)
    with pytest.raises(ValueError, match="one entry"):
        _inputs([], ["1"], 1)
    with pytest.raises(ValueError, match="four-vectors"):
        _inputs([[[1, 2, 3]]], "1", 1)
    with pytest.raises(TypeError, match="mapping"):
        _inputs([[[1, 0, 0, 1]]], {1: "1000000"}, 1)


@pytest.fixture(scope="module")
def source_artifact():
    path = os.environ.get("PYAMPLICOL_SOURCE_PUBLIC_ARTIFACT")
    if not path:
        pytest.skip(
            "set PYAMPLICOL_SOURCE_PUBLIC_ARTIFACT to the saved gg source artifact"
        )
    native = pytest.importorskip("_loop_reduction")
    if not hasattr(native, "NativeVirtualRuntime"):
        pytest.skip("native source artifact consumer is not built")
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/gluon_scattering_qcd.json"
    )
    points = json.loads(fixture.read_text(), parse_int=str, parse_float=str)["points"]
    return Path(path), points


def test_virtual_manifest_selection_and_explicit_failures(source_artifact, monkeypatch):
    from pyamplicol import Runtime
    from pyamplicol.api.errors import EvaluationError

    path, _ = source_artifact

    def no_tree():
        raise AssertionError("virtual manifest entered tree loader")

    monkeypatch.setattr("pyamplicol.runtime.backend._load_native_module", no_tree)
    runtime = Runtime.load(path / "artifact.json", process="gg_second")
    assert runtime.physics.process_id == "gg_second"
    result = runtime.evaluate([], mu_r2=[], r2_strategy="model")
    assert result.shape == (0, 1, 3)
    assert runtime.inspect()["requested_r2_strategy"] == "model"
    assert runtime.inspect()["r2_selection_reason"] == "explicit strategy"
    with pytest.raises(EvaluationError):
        Runtime.load(path, process="unknown-process")


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance,bits",
    [
        ("double", 16, "3e-11", 53),
        ("double-double", 16, "3e-25", 106),
        ("arbitrary", 70, "3e-60", None),
    ],
)
def test_saved_source_artifact_public_all_arithmetic(
    source_artifact, monkeypatch, arithmetic, precision, tolerance, bits
):
    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualEvaluation
    from pyamplicol.config import R2Strategy

    from .test_loop_physical_gluons import _expected
    from .test_loop_source_portable import _forbid_source

    path, points = source_artifact
    _forbid_source(monkeypatch)

    # The virtual route must not depend on a tree numerical runtime extension.
    def no_tree():
        raise AssertionError("virtual artifact entered the tree loader")

    monkeypatch.setattr("pyamplicol.runtime.backend._load_native_module", no_tree)
    alpha = "0.118000000000000000000000000000000000001"
    runtime = Runtime.load(path, process="gg", model_parameters={"aS": alpha})
    assert runtime.execution_mode == "loop_recurrence"
    assert runtime.contribution_kind == "virtual"
    assert len(runtime.artifact_id) == 64
    assert runtime.physics.process_id == "gg"
    assert len(runtime.physics.helicities) == 16
    assert runtime.physics.selector_capabilities == ()
    options = dict(
        arithmetic=arithmetic,
        precision=precision,
        mu_r2=["1000000", "900000", "2100000"],
    )
    result = runtime.evaluate(points, **options)
    assert isinstance(result, VirtualEvaluation)
    assert result.shape == (3, 1, 3)
    assert result.born_shape == (3, 1)
    assert result.epsilon_powers == (0, -1, -2)
    assert result.sectors[0].order == (("QCD", 6),)
    assert result.for_order((("QCD", 6),)).values == result.values
    assert result.r2.requested == R2Strategy.AUTO
    assert result.r2.effective == R2Strategy.MODEL
    assert all(d.valid and not d.accuracy_established for d in result.diagnostics)
    if bits is not None:
        assert all(d.precision_bits == bits for d in result.diagnostics)
    expected_type = float if arithmetic == "double" else Decimal
    assert all(
        isinstance(v, expected_type)
        for row in result.values
        for order in row
        for v in order
    )
    with localcontext() as context:
        context.prec = 110
        for point, born, poles in zip(
            points, result.born_values, result.values, strict=True
        ):
            target_born, target_pole = _expected(point, alpha)
            for value, target in (
                (born[0], target_born),
                (poles[0].double_pole, target_pole),
            ):
                assert abs(Decimal(value) - target) < abs(target) * Decimal(tolerance)
    metadata = runtime.inspect()
    assert metadata["parameter_overrides"]["aS"] == [alpha, "0"]
    assert metadata["arithmetic"] == arithmetic
    assert metadata["scheme_status"] == "declared-not-certified"
    assert metadata["rescue_available"] is False
    empty = runtime.evaluate([], arithmetic=arithmetic, precision=precision, mu_r2=[])
    assert empty.shape == (0, 1, 3)
    assert empty.born_shape == (0, 1)
    assert empty.valid == ()
    for kwargs in (
        {"r2_strategy": "direct"},
        {"helicities": ["h0"]},
        {"color_flows": ["full"]},
    ):
        with pytest.raises(CompatibilityError):
            runtime.evaluate(points, mu_r2="1000000", **kwargs)
    with pytest.raises(CompatibilityError, match="resolved"):
        runtime.evaluate_resolved(points, mu_r2="1000000")
