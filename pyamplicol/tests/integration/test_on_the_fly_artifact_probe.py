# SPDX-License-Identifier: 0BSD
"""Genuine-artifact canary for the hidden on-the-fly execution probe."""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import pyamplicol.generation.service as generation_service
from pyamplicol import Generator, ModelSource, ProcessRequest, Runtime
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RunConfig,
)

_MODEL = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "scalars"
    / "scalars.json"
)
_HELICITIES = (0, 0, 0, 0)
_S = 1_000_000.0
_FINAL_ONE_ENERGY = (_S + 1.0 - 4.0) / 2_000.0
_FINAL_TWO_ENERGY = (_S + 4.0 - 1.0) / 2_000.0
_FINAL_MOMENTUM = math.sqrt(_FINAL_ONE_ENERGY * _FINAL_ONE_ENERGY - 1.0)
_POINT_0012 = (
    (500.0, 0.0, 0.0, 500.0),
    (500.0, 0.0, 0.0, -500.0),
    (_FINAL_ONE_ENERGY, 0.0, 0.0, _FINAL_MOMENTUM),
    (_FINAL_TWO_ENERGY, 0.0, 0.0, -_FINAL_MOMENTUM),
)
_POINT_0000 = (
    (500.0, 0.0, 0.0, 500.0),
    (500.0, 0.0, 0.0, -500.0),
    (500.0, 0.0, 0.0, 500.0),
    (500.0, 0.0, 0.0, -500.0),
)


@dataclass(frozen=True)
class _ScalarContactCase:
    process: str
    process_id: str
    vertex: str
    point: tuple[tuple[float, ...], ...]
    normalization_factor: float


_CASES = (
    _ScalarContactCase(
        process="scalar_0 scalar_0 > scalar_0 scalar_0",
        process_id="scalars_0000",
        vertex="V_4_SCALAR_0000",
        point=_POINT_0000,
        normalization_factor=0.5,
    ),
    _ScalarContactCase(
        process="scalar_0 scalar_0 > scalar_1 scalar_2",
        process_id="scalars_0012",
        vertex="V_4_SCALAR_0012",
        point=_POINT_0012,
        normalization_factor=1.0,
    ),
)


def _analytic_scalar_contact(case: _ScalarContactCase, coupling: float) -> float:
    # Both selected V4 vertices have color 1, Lorentz structure 1, and raw
    # amplitude i*lam.  The only normalization difference is the factor 1/2
    # for the two identical scalar_0 particles in the 0000 final state.
    return case.normalization_factor * coupling * coupling


def _config() -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc", lc_flow_layout="topology-replay"),
        process=ProcessConfig(
            coupling_order_policy="explicit",
            max_coupling_orders={"QCD": 1, "QED": 0},
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode="recurrence",
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


def _flatten(points: tuple[tuple[tuple[float, ...], ...], ...]) -> list[float]:
    return [
        component
        for point in points
        for momentum in point
        for component in momentum
    ]


def _transverse(point: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    outgoing_one = point[2]
    outgoing_two = point[3]
    return (
        point[0],
        point[1],
        (outgoing_one[0], outgoing_one[3], 0.0, 0.0),
        (outgoing_two[0], outgoing_two[3], 0.0, 0.0),
    )


def _established_resolved_totals(
    runtime: Runtime, points: tuple[Any, ...]
) -> tuple[float, ...]:
    resolved = runtime.evaluate_resolved(points)
    assert resolved.helicity_ids == ("h:+0,+0,+0,+0",)
    assert resolved.color_ids == ("flow:singlet",)
    values = tuple(complex(value) for value in resolved.total())
    assert all(value.imag == pytest.approx(0.0, abs=1.0e-15) for value in values)
    return tuple(value.real for value in values)


def _canonical_direct_catalog_json(retained: dict[str, object]) -> bytes:
    direct_json = retained["direct_template_catalog_json"]
    assert isinstance(direct_json, bytes)
    assert direct_json == json.dumps(
        retained["direct_template_catalog"].to_dict(),  # type: ignore[union-attr]
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return direct_json


def _invoke_probe(
    probe: Any,
    artifact: Path,
    case: _ScalarContactCase,
    retained: dict[str, object],
    direct_json: bytes,
    points: tuple[tuple[tuple[float, ...], ...], ...],
    *,
    overrides: dict[str, list[float]] | None = None,
    pack_digest: str | None = None,
    tamper: bool = False,
    benchmark: bool = False,
    benchmark_warmup_repetitions: int = 0,
    benchmark_repetitions: int = 0,
    collect_current_diagnostics: bool = True,
) -> dict[str, Any]:
    return probe(
        str(artifact),
        case.process_id,
        retained["builder_input"],
        retained["prepared_template_input"],
        direct_json,
        pack_digest or retained["prepared_kernel_pack_digest"],
        0,
        list(_HELICITIES),
        _flatten(points),
        len(points),
        parameter_overrides=overrides,
        tamper_executor_key=tamper,
        benchmark=benchmark,
        benchmark_warmup_repetitions=benchmark_warmup_repetitions,
        benchmark_repetitions=benchmark_repetitions,
        collect_current_diagnostics=collect_current_diagnostics,
    )


def _assert_zero_poison_counts(report: dict[str, Any]) -> None:
    assert report["direct_plan_load_attempts"] == 0
    assert report["direct_plan_decode_attempts"] == 0
    assert report["direct_plan_materialization_attempts"] == 0
    assert report["established_builder_attempts"] == 0


def _assert_raw_i_lambda(report: dict[str, Any], coupling: float) -> None:
    for raw in report["raw_amplitudes"]:
        assert raw[0] == pytest.approx(0.0, abs=1.0e-14)
        assert raw[1] == pytest.approx(coupling)


def test_hidden_on_the_fly_probe_executes_genuine_scalar_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    rusticol = importlib.import_module("pyamplicol._rusticol")
    probe = getattr(rusticol, "_on_the_fly_artifact_probe_v1", None)
    if not callable(probe):
        pytest.skip("the native extension lacks on-the-fly test support")
    if importlib.util.find_spec("symbolica") is None:
        pytest.skip("Symbolica is unavailable")

    config = _config()
    prepared_output = tmp_path / "fresh-scalar-jit-o2.pyamplicol-model"
    assert not prepared_output.exists()
    prepared = ModelSource.from_path(_MODEL).compile(
        cache_dir=tmp_path / "fresh-model-cache",
        use_cache=False,
        prepared_output=prepared_output,
        evaluator=config.evaluator,
    )
    assert prepared_output.is_file()
    captured: list[dict[str, object]] = []
    original = generation_service._invoke_rust_recurrence_lowering_v2

    def capture_lowering(*args: object, **kwargs: object) -> object:
        captured.append(
            {
                "builder_input": args[0],
                "prepared_template_input": args[1],
                "direct_template_catalog": args[2],
                "prepared_kernel_pack_digest": args[3],
                "direct_template_catalog_json": kwargs.get(
                    "direct_template_catalog_json"
                ),
            }
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(
        generation_service,
        "_invoke_rust_recurrence_lowering_v2",
        capture_lowering,
    )
    generated: dict[
        str, tuple[_ScalarContactCase, Path, dict[str, object], bytes]
    ] = {}
    for case in _CASES:
        artifact = tmp_path / f"fresh-{case.process_id}-artifact"
        assert not artifact.exists()
        Generator(config).generate(
            ProcessRequest.parse(case.process, name=case.process_id),
            artifact,
            model=prepared,
        )
        assert artifact.exists()
        assert len(captured) == len(generated) + 1
        retained = captured[-1]
        # These are the exact canonical bytes supplied to genuine lowering,
        # not a reconstruction from the completed artifact.
        direct_json = _canonical_direct_catalog_json(retained)
        generated[case.process_id] = (case, artifact, retained, direct_json)

    diagnostics: dict[str, object] = {}
    for case, artifact, retained, direct_json in generated.values():
        runtime = Runtime.load(artifact, process=case.process_id)
        transverse = _transverse(case.point)
        single_points = ((case.point,), (transverse,))
        established_default = _established_resolved_totals(
            runtime, (case.point, transverse)
        )
        expected_default = _analytic_scalar_contact(case, 1.0)
        assert established_default == pytest.approx(
            (expected_default, expected_default)
        )

        singles = tuple(
            _invoke_probe(
                probe,
                artifact,
                case,
                retained,
                direct_json,
                points,
            )
            for points in single_points
        )
        batch_order = (0, 1, 1, 0, 1, 0, 0)
        batch_points = tuple(single_points[index][0] for index in batch_order)
        many = _invoke_probe(
            probe,
            artifact,
            case,
            retained,
            direct_json,
            batch_points,
        )
        one = singles[0]
        assert one["process_id"] == case.process_id
        assert one["point_count"] == 1
        assert many["point_count"] == len(batch_order)
        assert one["artifact_id"] == many["artifact_id"]
        assert all(
            report["seed_digest"] == one["seed_digest"]
            and report["query_digest"] == one["query_digest"]
            and report["trace_digest"] == one["trace_digest"]
            for report in (*singles, many)
        )
        assert one["normalization_factor"] == pytest.approx(
            case.normalization_factor
        )
        assert one["normalized_values"] == pytest.approx((expected_default,))
        assert many["normalized_values"] == pytest.approx(
            (expected_default,) * len(batch_order)
        )
        for report in (*singles, many):
            _assert_zero_poison_counts(report)
            assert report["trace_build_count"] == 1
            assert report["trace_cache_hit_count"] == 0
            assert report["momentum_fill_count"] == 1
            assert report["benchmark_warmup_repetitions"] == 0
            assert report["benchmark_repetitions"] == 0
            assert report["benchmark_elapsed_seconds"] is None
            assert report["benchmark_seconds_per_point"] is None
            _assert_raw_i_lambda(report, 1.0)
            for raw, normalized in zip(
                report["raw_amplitudes"],
                report["normalized_values"],
                strict=True,
            ):
                assert normalized == pytest.approx(
                    (raw[0] * raw[0] + raw[1] * raw[1])
                    * report["normalization_factor"]
                )

        assert one["currents"]
        one_digests = [current["semantic_digest"] for current in one["currents"]]
        assert all(
            [current["semantic_digest"] for current in report["currents"]]
            == one_digests
            for report in (*singles, many)
        )
        for current_index, many_current in enumerate(many["currents"]):
            width = many_current["component_count"]
            assert len(many_current["values"]) == len(batch_order) * width
            for point_index, single_index in enumerate(batch_order):
                single_current = singles[single_index]["currents"][current_index]
                assert single_current["component_count"] == width
                assert len(single_current["values"]) == width
                start = point_index * width
                assert many_current["values"][start : start + width] == single_current[
                    "values"
                ]
                assert many["raw_amplitudes"][point_index] == singles[single_index][
                    "raw_amplitudes"
                ][0]

        runtime.set_model_parameters({"lam": 3.0})
        established_mutated = _established_resolved_totals(runtime, (case.point,))[0]
        expected_mutated = _analytic_scalar_contact(case, 3.0)
        assert established_mutated == pytest.approx(expected_mutated)
        mutated = _invoke_probe(
            probe,
            artifact,
            case,
            retained,
            direct_json,
            (case.point,),
            overrides={"lam": [3.0, 0.0]},
        )
        _assert_zero_poison_counts(mutated)
        _assert_raw_i_lambda(mutated, 3.0)
        assert mutated["trace_digest"] == one["trace_digest"]
        assert mutated["normalized_values"] == pytest.approx((expected_mutated,))
        assert mutated["normalized_values"] == pytest.approx((established_mutated,))
        # `lam` is a runtime input while SCALAR_COUPLING is the prepared
        # derived slot consumed by the executor. This exact factor-three raw
        # scaling exercises the authenticated runtime-to-prepared projection.
        assert mutated["raw_amplitudes"][0][0] == pytest.approx(
            3.0 * one["raw_amplitudes"][0][0]
        )
        assert mutated["raw_amplitudes"][0][1] == pytest.approx(
            3.0 * one["raw_amplitudes"][0][1]
        )

        benchmark = _invoke_probe(
            probe,
            artifact,
            case,
            retained,
            direct_json,
            (case.point,),
            benchmark=True,
            benchmark_warmup_repetitions=2,
            benchmark_repetitions=5,
            collect_current_diagnostics=False,
        )
        _assert_zero_poison_counts(benchmark)
        _assert_raw_i_lambda(benchmark, 1.0)
        assert benchmark["normalized_values"] == pytest.approx((expected_default,))
        assert benchmark["currents"] == []
        assert benchmark["trace_build_count"] == 1
        assert benchmark["trace_cache_hit_count"] == 7
        assert benchmark["momentum_fill_count"] == 7
        assert benchmark["benchmark_warmup_repetitions"] == 2
        assert benchmark["benchmark_repetitions"] == 5
        assert math.isfinite(benchmark["benchmark_elapsed_seconds"])
        assert benchmark["benchmark_elapsed_seconds"] >= 0.0
        assert benchmark["benchmark_seconds_per_point"] == pytest.approx(
            benchmark["benchmark_elapsed_seconds"] / 5.0
        )

        with pytest.raises(ValueError, match="at least one timed repetition"):
            _invoke_probe(
                probe,
                artifact,
                case,
                retained,
                direct_json,
                (case.point,),
                benchmark=True,
            )

        with pytest.raises(ValueError, match="not used by the process"):
            _invoke_probe(
                probe,
                artifact,
                case,
                retained,
                direct_json,
                (case.point,),
                overrides={"unknown_parameter": [1.0, 0.0]},
            )
        with pytest.raises(
            rusticol.ArtifactError, match="prepared-kernel pack digest"
        ):
            _invoke_probe(
                probe,
                artifact,
                case,
                retained,
                direct_json,
                (case.point,),
                pack_digest="1" * 64,
            )
        with pytest.raises(
            rusticol.ArtifactError, match="executor|semantic|mapping"
        ):
            _invoke_probe(
                probe,
                artifact,
                case,
                retained,
                direct_json,
                (case.point,),
                tamper=True,
            )

        diagnostics[case.process_id] = {
            "vertex": case.vertex,
            "established_default": established_default,
            "probe_default_raw": one["raw_amplitudes"],
            "probe_default_normalized": one["normalized_values"],
            "probe_batch_normalized": many["normalized_values"],
            "normalization_factor": one["normalization_factor"],
            "poison_counts": (
                one["direct_plan_load_attempts"],
                one["direct_plan_decode_attempts"],
                one["direct_plan_materialization_attempts"],
                one["established_builder_attempts"],
            ),
            "established_lam3": established_mutated,
            "probe_lam3_raw": mutated["raw_amplitudes"],
            "probe_lam3_normalized": mutated["normalized_values"],
        }

    case_0000, artifact_0000, _, _ = generated["scalars_0000"]
    _, _, retained_0012, direct_json_0012 = generated["scalars_0012"]
    with pytest.raises(rusticol.ArtifactError, match="do not belong|selected artifact"):
        _invoke_probe(
            probe,
            artifact_0000,
            case_0000,
            retained_0012,
            direct_json_0012,
            (case_0000.point,),
        )

    print("genuine on-the-fly scalar contact diagnostics:", diagnostics)
