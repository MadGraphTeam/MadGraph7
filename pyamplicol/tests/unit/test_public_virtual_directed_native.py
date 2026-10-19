# SPDX-License-Identifier: 0BSD
"""Graph-free publication and public complex transport of a real source plan."""

import json
import os
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from pyamplicol import CorrelatedRequest, CorrelatedValue, Runtime
from pyamplicol.api import VirtualArithmeticSpec, VirtualStabilityPolicy
from pyamplicol.api.errors import CompatibilityError, EvaluationError


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    plan_path = os.environ.get("PYAMPLICOL_NATIVE_DIRECTED_PACBIN")
    model_path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not plan_path or not model_path:
        pytest.skip("set the retained directed plan and original loop_sm UFO paths")
    native = pytest.importorskip("_loop_reduction")
    from pyamplicol.api.requests import ModelSource
    from pyamplicol.config import resolve_config
    from pyamplicol.generation.artifact_writer import (
        _GenerationConfigProvenance,
        write_schema_v3_artifact,
    )
    from pyamplicol.models.loading import compile_model_source

    from .test_loop_artifact_writer import source_process

    root = tmp_path_factory.mktemp("public-directed")
    # The retained source control uses this same supplied loop_sm/default model,
    # with MB fixed to zero by its saved source owner. Compilation here refreshes
    # only the ordinary model bundle; no graphs or numerators are regenerated.
    model = compile_model_source(model_path, cache_dir=root / "cache")
    process = source_process(Path(plan_path))
    process = replace(
        process,
        process_id="uudd",
        expression="u u~ > d d~",
        metadata=replace(process.metadata, correlations=True),
        validation_point=replace(
            process.validation_point, process_id="uudd", process="u u~ > d d~"
        ),
    )
    configuration = _GenerationConfigProvenance.from_config(
        resolve_config(
            {
                "action": "generate",
                "process": {"loop_coupling_orders": ["QCD"]},
                "generation": {"emit_api_bundle": False},
            }
        )
    )
    with pytest.MonkeyPatch.context() as patch:
        # Only the development consumer import/producer target is injected;
        # PACBIN/metadata integrity and numerical execution remain authoritative.
        patch.setattr("pyamplicol.loops.native.load_loop_native", lambda: native)
        patch.setattr("pyamplicol.runtime.virtual.load_loop_native", lambda: native)
        patch.setattr(
            "pyamplicol.generation.artifact_writer.active_native_build_inputs_sha256",
            lambda: "a" * 64,
        )
        patch.setattr(
            "pyamplicol.generation.artifact_writer._target_metadata",
            lambda _config: (
                {"triple": "x86_64-unknown-linux-gnu", "cpu_features": []},
                1,
            ),
        )
        output = root / "artifact"
        write_schema_v3_artifact(
            output,
            mode="error",
            source=ModelSource.from_path(model_path),
            compiled_model=model,
            configuration=configuration,
            processes=(process,),
            timings={"total": 0.0},
            staged_validation=lambda staged: Runtime.load(staged),
        )
        with pytest.raises(ValueError, match="correlation capability"):
            write_schema_v3_artifact(
                root / "mismatch",
                mode="error",
                source=ModelSource.from_path(model_path),
                compiled_model=model,
                configuration=configuration,
                processes=(
                    replace(
                        process, metadata=replace(process.metadata, correlations=False)
                    ),
                ),
                timings={"total": 0.0},
            )
        assert not (root / "mismatch").exists()
        fixture = (
            Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
        )
        points = json.loads(fixture.read_text(), parse_int=str, parse_float=str)[
            "points"
        ][:2]
        yield output, points


def _pair(value):
    if isinstance(value, CorrelatedValue):
        return value.real, value.imag
    return Decimal(str(value.real)), Decimal(str(value.imag))


def _close(left, right, tolerance):
    left_pair, right_pair = _pair(left), _pair(right)
    # A vanishing imaginary part is judged on the complete complex coefficient,
    # so harmless roundoff is not assigned an ill-conditioned relative error.
    scale = max(*(abs(x) for x in (*left_pair, *right_pair)), Decimal("1e-15"))
    for a, b in zip(left_pair, right_pair, strict=True):
        assert abs(a - b) <= Decimal(tolerance) * scale


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", 16, "4e-10"),
        ("double-double", 16, "4e-24"),
        ("arbitrary", 70, "4e-56"),
    ],
)
def test_public_saved_directed_all_carriers(
    artifact, monkeypatch, arithmetic, precision, tolerance
):
    from .test_loop_source_portable import _forbid_source

    path, points = artifact
    _forbid_source(monkeypatch)
    runtime = Runtime.load(path, arithmetic=arithmetic, precision=precision)
    assert [item.id for item in runtime.available_color_correlations()] == [
        "identity",
        "casimir",
        "ordered",
        "swapped",
        "born",
    ]
    assert runtime.available_color_correlations()[1].bra[0].emitter_label == 1
    assert runtime.inspect()["correlations"] is True
    assert runtime.physics.color_flow_ids == ()
    scales = ["1000000", "900000"]
    ordinary = runtime.evaluate(points, mu_r2=scales)
    result = runtime.evaluate_correlated_many(
        points,
        {
            name: CorrelatedRequest(name)
            for name in ("ordered", "identity", "swapped", "casimir")
        },
        mu_r2=scales,
    )
    expected = complex if arithmetic == "double" else CorrelatedValue
    with localcontext() as context:
        context.prec = 120
        for p in range(2):
            for b in range(len(ordinary.born_orders)):
                _close(
                    result["identity"].born_values[p][b],
                    ordinary.born_values[p][b],
                    tolerance,
                )
            for v in range(len(ordinary.sectors)):
                for e in range(3):
                    value = result["ordered"].values[p][v][e]
                    assert isinstance(value, expected)
                    real, imag = _pair(result["swapped"].values[p][v][e])
                    _close(value, CorrelatedValue(real, -imag), tolerance)
                    _close(
                        result["identity"].values[p][v][e],
                        ordinary.values[p][v][e],
                        tolerance,
                    )
        assert abs(_pair(result["ordered"].finite[0][0])[1]) > Decimal("1e-4")
    assert all(
        value.correlated and value.shape == ordinary.shape for value in result.values()
    )
    assert runtime.evaluate_correlated(
        [], color_correlation="ordered", mu_r2=[]
    ).shape == (0, 1, 3)
    with pytest.raises(CompatibilityError, match="resolved"):
        runtime.evaluate_resolved(points, mu_r2=scales)
    with pytest.raises(CompatibilityError, match="spin sources"):
        runtime.set_spin_correlation_vectors({1: (1, 0, 0, 1)})
    runtime.clear()
    runtime.set_model_parameters({"aS": "0.118000000000000000000000000000001"})
    assert runtime.inspect()["parameter_overrides"]["aS"] == [
        "0.118000000000000000000000000000001",
        "0",
    ]
    assert runtime.evaluate_correlated(
        points[:1], color_correlation="identity", mu_r2=scales[:1]
    ).shape == (1, 1, 3)


def test_public_directed_subset_rescue_preserves_complex_histories(artifact):
    path, points = artifact
    runtime = Runtime.load(path)
    points = [points[0]] * 3
    scales = ["1000000", "1e-400", "1000000"]
    with pytest.raises(EvaluationError) as caught:
        runtime.evaluate_correlated(points, color_correlation="ordered", mu_r2=scales)
    assert {failure.point_id for failure in caught.value.failures} == {1}
    result = runtime.evaluate_correlated_many(
        points,
        {"left": CorrelatedRequest("ordered"), "right": CorrelatedRequest("swapped")},
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
    for value in result.values():
        assert [d.status for d in value.diagnostics] == [
            "success",
            "rescued",
            "success",
        ]
        assert value.diagnostics[1].verification.attempts[-1].conversion_agreement
    for pole in range(3):
        left, right = (
            result["left"].values[1][0][pole],
            result["right"].values[1][0][pole],
        )
        assert left == pytest.approx(right.conjugate(), rel=1e-8, abs=1e-9)
