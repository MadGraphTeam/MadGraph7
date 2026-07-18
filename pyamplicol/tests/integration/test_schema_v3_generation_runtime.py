# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib.util
import json
import shutil
import warnings
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from pyamplicol import (
    BenchmarkConfig,
    BenchmarkRunner,
    Generator,
    ModelSource,
    ProcessSet,
    Runtime,
)
from pyamplicol.api.errors import EvaluationError
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    ModelConfig,
    RunConfig,
)
from pyamplicol.generation.phase_space import massive_rambo_final_state
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.base import Model
from tools.developer.analytic_oracles import (
    scalar_contact_2to2,
    scalar_gravity_2to2,
)
from tools.developer.reference_fixture import load_reference_fixture

REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "reference"
REFERENCE = REFERENCE_ROOT / "physics-v2.json"
_VALIDATED_REFERENCE_FIXTURE = load_reference_fixture(
    REFERENCE,
    (
        REFERENCE_ROOT / "legacy-fortran-v2.json",
        REFERENCE_ROOT / "analytic-oracles-v2.json",
    ),
)
REFERENCE_PAYLOAD = json.loads(REFERENCE.read_text(encoding="utf-8"))
CASE_NAMES = {
    "lc": "case:sm_ddbar_z:lc",
    "nlc": "case:sm_ddbar_z:nlc",
    "full": "case:sm_ddbar_z:full",
}
MODEL_ASSETS = Path(__file__).resolve().parents[2] / "src/pyamplicol/assets/models"
EXTERNAL_CASES = {
    "scalars_2to2_lc": MODEL_ASSETS / "json/scalars/scalars.json",
    "scalar_gravity_2to2_lc": (
        MODEL_ASSETS / "json/scalar_gravity/scalar_gravity.json"
    ),
}
EXTERNAL_SM_SOURCES = {
    "json": MODEL_ASSETS / "json/sm/sm.json",
    "ufo": MODEL_ASSETS / "ufo/sm",
}
EXTERNAL_SM_PROCESS_IDS = (
    "sm_ddbar_z",
    "sm_udbar_wplus",
    "sm_ddbar_zg",
    "sm_ddbar_ee",
    "sm_ddbar_uubar",
    "sm_ddbar_ddbar",
    "sm_gg_gg",
    "sm_gg_ttbar",
    "sm_ddbar_zgg",
)
# These cases intentionally use the current shared-ordering DAG rather than
# the larger topology retained in the independent pre-optimization fixture.
# Numerical expectations still come exclusively from that fixture.
CURRENT_BUILTIN_TOPOLOGY = {
    f"case:sm_gg_ttbar:{accuracy}": {
        "currents": 36,
        "interactions": 44,
        "reduction_groups": 32,
        "roots": 32,
    }
    for accuracy in ("nlc", "full")
} | {
    f"case:sm_ddbar_zgg:{accuracy}": {
        "currents": 117,
        "interactions": 242,
        "reduction_groups": 48,
        "roots": 48,
    }
    for accuracy in ("nlc", "full")
}
NAMED_CASE_IDS = {
    "builtin_sm_ddbar_zg_lc": "case:sm_ddbar_zg:lc",
    "scalars_2to2_lc": "case:scalars_2to2:lc",
    "scalar_gravity_2to2_lc": "case:scalar_gravity_2to2:lc",
}


@pytest.fixture(autouse=True)
def _discard_generated_artifacts(tmp_path: Path) -> Iterator[None]:
    """Keep parameterized generation tests from retaining every large artifact."""

    yield
    shutil.rmtree(tmp_path, ignore_errors=True)


def _case_payload(
    case_id: str,
) -> tuple[dict[str, Any], list[Any], dict[str, dict[str, float]]]:
    case = next(item for item in REFERENCE_PAYLOAD["cases"] if item["id"] == case_id)
    process = next(
        item
        for item in REFERENCE_PAYLOAD["processes"]
        if item["id"] == case["process_id"]
    )
    point_id = case["point_ids"][0]
    point = next(item for item in REFERENCE_PAYLOAD["points"] if item["id"] == point_id)
    observation = next(
        item for item in case["observations"] if item["point_id"] == point_id
    )
    helicity_ids = [axis["id"] for axis in case["axes"]["helicities"]]
    color_ids = [axis["id"] for axis in case["axes"]["colors"]]
    resolved = {
        helicity_id: {
            color_id: float(value)
            for color_id, value in zip(color_ids, row, strict=True)
        }
        for helicity_id, row in zip(helicity_ids, observation["values"], strict=True)
    }
    topology = case["topology"]
    reference = {
        "process": process["expression"],
        "process_id": process["id"],
        "color_accuracy": case["color_accuracy"],
        "total": float(observation["total"]),
        "resolved": resolved,
        "topology": {
            "current_count": topology["currents"],
            "interaction_count": topology["interactions"],
            "amplitude_root_count": topology["roots"],
            "reduction_group_count": topology["reduction_groups"],
        },
    }
    return reference, [point["momenta"]], resolved


def _reference_case(
    accuracy: str,
) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    return _case_payload(CASE_NAMES[accuracy])


def _named_reference_case(name: str) -> tuple[dict[str, Any], list[Any]]:
    case, momenta, _resolved = _case_payload(NAMED_CASE_IDS[name])
    return case, momenta


def _write_color_dummy_relabelled_sm(root: Path) -> Path:
    source = EXTERNAL_SM_SOURCES["json"]
    raw = json.loads(source.read_text(encoding="utf-8"))
    vertex = next(item for item in raw["vertex_rules"] if item["name"] == "V_37")
    vertex["color_structures"] = [
        "*".join(reversed(expression.replace("-1", "-97").split("*")))
        for expression in vertex["color_structures"]
    ]
    model_root = root / "model"
    model_root.mkdir()
    model_path = model_root / "sm.json"
    model_path.write_text(
        json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    restriction = source.with_name("restrict_default.json")
    (model_root / restriction.name).write_bytes(restriction.read_bytes())
    return model_path


@pytest.mark.parametrize("accuracy", ("lc", "nlc", "full"))
def test_current_source_generates_and_evaluates_schema_v3(
    tmp_path: Path,
    accuracy: str,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    reference, reference_momenta, expected_resolved = _reference_case(accuracy)
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy=accuracy),
    )
    artifact = tmp_path / accuracy

    result = Generator(config).generate(reference["process"], artifact)

    assert result.schema_version == 3
    outer = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    assert outer["schema_version"] == 3
    assert outer["processes"][0]["color_accuracy"] == accuracy
    process_root = artifact / "processes" / outer["processes"][0]["id"]
    execution = json.loads(
        (process_root / "execution.json").read_text(encoding="utf-8")
    )
    physics = json.loads((process_root / "physics.json").read_text(encoding="utf-8"))
    assert execution["schema_version"] == 3
    assert execution["kind"] == "pyamplicol-runtime-execution"
    assert execution["runtime_schema"]["kind"] == ("pyamplicol-runtime-execution-plan")
    assert all(
        parameter["name"] != "runtime.lc_sector_id"
        for parameter in execution["runtime_schema"]["model_parameters"]
    )
    assert (
        execution["dag_summary"]["current_count"]
        == reference["topology"]["current_count"]
    )
    assert (
        execution["dag_summary"]["interaction_count"]
        == reference["topology"]["interaction_count"]
    )
    assert (
        execution["dag_summary"]["amplitude_root_count"]
        == reference["topology"]["amplitude_root_count"]
    )
    assert (
        len(physics["reduction"]["groups"])
        == reference["topology"]["reduction_group_count"]
    )

    runtime = Runtime.load(artifact)
    momenta = tuple(
        tuple(tuple(float(component) for component in vector) for vector in point)
        for point in reference_momenta
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        total = runtime.evaluate(momenta)
        resolved = runtime.evaluate_resolved(momenta)

    assert recorded == []
    assert runtime.physics.color_accuracy == accuracy
    expected_capabilities = (
        ("helicity", "color_flow") if accuracy == "lc" else ("helicity",)
    )
    assert runtime.physics.selector_capabilities == expected_capabilities
    assert total[0].real == pytest.approx(reference["total"], rel=1.0e-12)
    assert total[0].imag == pytest.approx(0.0, abs=1.0e-15)
    assert resolved.total()[0] == pytest.approx(total[0], rel=1.0e-12)

    if accuracy == "lc":
        benchmark = BenchmarkRunner(
            BenchmarkConfig(
                target_runtime=1.0e-3,
                batch_size=2,
                warmup_runs=0,
                minimum_samples=2,
            )
        ).run(runtime, points=momenta)
        assert benchmark.wall_time_per_point > 0.0
        assert benchmark.evaluator_time_per_point >= 0.0
        assert benchmark.environment["wall_time_source"] == (
            "runtime_core_repeated_wall_time"
        )
        assert benchmark.timing_breakdown is not None

    exact = runtime.evaluate_resolved(momenta, precision=32)
    assert exact.helicity_ids == resolved.helicity_ids
    assert exact.color_ids == resolved.color_ids
    for jit_helicity, exact_helicity in zip(
        resolved.values[0], exact.values[0], strict=True
    ):
        for jit_value, exact_value in zip(jit_helicity, exact_helicity, strict=True):
            assert complex(jit_value).real == pytest.approx(
                complex(exact_value).real,
                rel=1.0e-12,
                abs=1.0e-15,
            )
            assert complex(jit_value).imag == pytest.approx(
                complex(exact_value).imag,
                rel=1.0e-12,
                abs=1.0e-15,
            )

    if accuracy != "lc":
        for precision in (16, 32):
            with pytest.raises(
                EvaluationError,
                match="color-flow selection is unavailable",
            ):
                runtime.evaluate_resolved(
                    momenta,
                    color_flows=[resolved.color_ids[0]],
                    precision=precision,
                )

    helicity_index = {value: index for index, value in enumerate(resolved.helicity_ids)}
    color_index = {value: index for index, value in enumerate(resolved.color_ids)}
    for helicity_id, expected_colors in expected_resolved.items():
        for color_id, expected_value in expected_colors.items():
            actual = resolved.values[0][helicity_index[helicity_id]][
                color_index[color_id]
            ]
            assert actual.real == pytest.approx(expected_value, rel=1.0e-12)
            assert actual.imag == pytest.approx(0.0, abs=1.0e-15)


def test_nlc_one_line_shared_orderings_match_sector_local_reference(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    artifact = tmp_path / "nlc-shared-one-line"
    Generator(
        RunConfig(
            action="generate",
            color=ColorConfig(accuracy="nlc"),
        )
    ).generate("g g > t t~ g", artifact)

    execution = json.loads(
        (
            artifact / "processes" / "g_g_to_t_tbar_g" / "execution.json"
        ).read_text(encoding="utf-8")
    )
    assert execution["dag_summary"] == {
        "amplitude_root_count": 192,
        "current_count": 250,
        "interaction_count": 624,
        "source_count": 10,
        "truncated": False,
    }

    runtime = Runtime.load(artifact)
    momenta = runtime._backend.validation_momenta()
    assert momenta is not None
    total = runtime.evaluate(momenta)

    # Captured with the previous exact sector-local NLC construction at the
    # same deterministic validation point.
    assert total[0].real == pytest.approx(5.285188765700242e-4, rel=1.0e-12)
    assert total[0].imag == pytest.approx(0.0, abs=1.0e-15)


@pytest.mark.parametrize(
    ("process", "color_accuracy"),
    (
        ("d d~ > t t~ g g", "full"),
        ("g g > g g g", "nlc"),
    ),
)
def test_recursive_current_reuse_matches_unshared_contracted_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    process: str,
    color_accuracy: str,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    from pyamplicol.generation import dag_compiler as dag_compiler_module

    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy=color_accuracy),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(samples=1),
        ),
        evaluator=EvaluatorConfig(
            output_chunk_size=512,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=1),
        ),
    )
    optimized_artifact = tmp_path / "recursive-current-reuse"
    external_artifact = tmp_path / "external-recursive-current-reuse"
    baseline_artifact = tmp_path / "attachment-local-reuse"
    Generator(config).generate(process, optimized_artifact)
    Generator(config).generate(
        process,
        external_artifact,
        model=ModelSource.from_path(EXTERNAL_SM_SOURCES["json"]),
    )

    monkeypatch.setattr(
        dag_compiler_module,
        "assign_recursive_current_evaluation_reuse",
        lambda dag, _model: dag,
    )
    monkeypatch.setattr(
        BuiltinSMModel,
        "vertex_evaluation_equivalence",
        Model.vertex_evaluation_equivalence,
    )
    Generator(config).generate(process, baseline_artifact)

    optimized = Runtime.load(optimized_artifact)
    external = Runtime.load(external_artifact)
    baseline = Runtime.load(baseline_artifact)
    momenta = optimized._backend.validation_momenta()
    assert momenta is not None
    optimized_total = optimized.evaluate(momenta)
    baseline_total = baseline.evaluate(momenta)
    assert optimized_total == pytest.approx(
        baseline_total,
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert external.evaluate(momenta) == pytest.approx(
        optimized_total,
        rel=1.0e-12,
        abs=1.0e-15,
    )

    optimized_resolved = optimized.evaluate_resolved(momenta)
    external_resolved = external.evaluate_resolved(momenta)
    baseline_resolved = baseline.evaluate_resolved(momenta)
    assert optimized_resolved.helicity_ids == baseline_resolved.helicity_ids
    assert optimized_resolved.color_ids == baseline_resolved.color_ids
    assert external_resolved.helicity_ids == optimized_resolved.helicity_ids
    assert external_resolved.color_ids == optimized_resolved.color_ids
    for optimized_helicities, baseline_helicities in zip(
        optimized_resolved.values,
        baseline_resolved.values,
        strict=True,
    ):
        for optimized_colors, baseline_colors in zip(
            optimized_helicities,
            baseline_helicities,
            strict=True,
        ):
            assert optimized_colors == pytest.approx(
                baseline_colors,
                rel=1.0e-12,
                abs=1.0e-15,
            )
    for external_helicities, optimized_helicities in zip(
        external_resolved.values,
        optimized_resolved.values,
        strict=True,
    ):
        for external_colors, optimized_colors in zip(
            external_helicities,
            optimized_helicities,
            strict=True,
        ):
            assert external_colors == pytest.approx(
                optimized_colors,
                rel=1.0e-12,
                abs=1.0e-15,
            )


def test_chunked_stage_evaluators_prune_inputs_and_preserve_precision(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    def config(output_chunk_size: int | None) -> RunConfig:
        return RunConfig(
            action="generate",
            color=ColorConfig(accuracy="lc"),
            generation=GenerationConfig(workers=1, emit_api_bundle=False),
            evaluator=EvaluatorConfig(
                output_chunk_size=output_chunk_size,
                optimization=EvaluatorOptimizationConfig(cores=1),
                jit=JITConfig(optimization_level=1),
            ),
        )

    artifact = tmp_path / "mapped-chunks"
    baseline_artifact = tmp_path / "unchunked"
    Generator(config(2)).generate("d d~ > z g", artifact)
    Generator(config(None)).generate("d d~ > z g", baseline_artifact)

    execution = json.loads(
        (
            artifact / "processes" / "d_dbar_to_z_g" / "execution.json"
        ).read_text(encoding="utf-8")
    )
    stages = execution["compiled"]["stage_evaluators"]
    evaluator_manifests = [
        *(stage["evaluator"] for stage in stages["stages"]),
        stages["amplitude_stage"]["evaluator"],
    ]
    chunked = [
        manifest
        for manifest in evaluator_manifests
        if manifest["kind"] == "chunked-symbolica-evaluator"
    ]
    assert chunked
    assert any(
        len(indices) < manifest["input_len"]
        for manifest in chunked
        for indices in manifest["chunk_input_indices"]
    )
    for manifest in chunked:
        assert len(manifest["chunks"]) == len(manifest["chunk_input_indices"])
        for child, indices in zip(
            manifest["chunks"], manifest["chunk_input_indices"], strict=True
        ):
            assert child["input_len"] == len(indices)
            assert indices == sorted(set(indices))

    runtime = Runtime.load(artifact)
    baseline = Runtime.load(baseline_artifact)
    momenta = runtime._backend.validation_momenta()
    assert momenta is not None
    resolved = runtime.evaluate_resolved(momenta)
    exact = runtime.evaluate_resolved(momenta, precision=32)
    assert resolved.total()[0] == pytest.approx(
        baseline.evaluate_resolved(momenta).total()[0], rel=1.0e-12
    )
    assert exact.total()[0] == baseline.evaluate_resolved(
        momenta, precision=32
    ).total()[0]

    final_state = massive_rambo_final_state(
        2,
        sqrt_s=1000.0,
        masses=(91.188, 0.0),
        seed=54321,
    )
    alternate = (
        (500.0, 0.0, 0.0, 500.0),
        (500.0, 0.0, 0.0, -500.0),
        *final_state,
    )
    mixed_batch = (momenta[0], alternate, momenta[0])
    assert runtime.evaluate(mixed_batch) == pytest.approx(
        Runtime.load(artifact).evaluate(mixed_batch),
        rel=1.0e-13,
        abs=1.0e-15,
    )
    assert runtime.evaluate((alternate,)) == pytest.approx(
        Runtime.load(artifact).evaluate((alternate,)),
        rel=1.0e-13,
        abs=1.0e-15,
    )


@pytest.mark.parametrize("case_name", tuple(EXTERNAL_CASES))
def test_current_source_external_models_match_reference(
    tmp_path: Path,
    case_name: str,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    reference, reference_momenta = _named_reference_case(case_name)
    config = RunConfig(
        action="generate",
        model=ModelConfig(cache=False),
        color=ColorConfig(accuracy=reference["color_accuracy"]),
    )
    artifact = tmp_path / case_name
    result = Generator(config).generate(
        reference["process"],
        artifact,
        model=ModelSource.from_path(EXTERNAL_CASES[case_name]),
    )

    assert result.schema_version == 3
    runtime = Runtime.load(artifact)
    momenta = tuple(
        tuple(tuple(float(component) for component in vector) for vector in point)
        for point in reference_momenta
    )
    total = runtime.evaluate(momenta)
    resolved = runtime.evaluate_resolved(momenta)

    assert runtime.physics.color_accuracy == "lc"
    assert runtime.physics.selector_capabilities == ("helicity", "color_flow")
    assert total[0].real == pytest.approx(reference["total"], rel=1.0e-12)
    assert total[0].imag == pytest.approx(0.0, abs=1.0e-15)
    assert resolved.total()[0] == pytest.approx(total[0], rel=1.0e-12)

    helicity_index = {value: index for index, value in enumerate(resolved.helicity_ids)}
    color_index = {value: index for index, value in enumerate(resolved.color_ids)}
    for helicity_id, expected_colors in reference["resolved"].items():
        for color_id, expected_value in expected_colors.items():
            actual = resolved.values[0][helicity_index[helicity_id]][
                color_index[color_id]
            ]
            assert actual.real == pytest.approx(expected_value, rel=1.0e-12)
            assert actual.imag == pytest.approx(0.0, abs=1.0e-15)

    precise_by_precision = {}
    for precision in (32, 80):
        precise = runtime.evaluate_resolved(momenta, precision=precision)
        precise_by_precision[precision] = precise
        assert precise.helicity_ids == resolved.helicity_ids
        assert precise.color_ids == resolved.color_ids
        for jit_helicity, precise_helicity in zip(
            resolved.values[0], precise.values[0], strict=True
        ):
            for jit_value, precise_value in zip(
                jit_helicity, precise_helicity, strict=True
            ):
                assert complex(precise_value).real == pytest.approx(
                    jit_value.real,
                    rel=1.0e-12,
                    abs=1.0e-15,
                )
                assert complex(precise_value).imag == pytest.approx(
                    jit_value.imag,
                    rel=1.0e-12,
                    abs=1.0e-15,
                )

    low_precision = precise_by_precision[32]
    high_precision = precise_by_precision[80]
    for low_helicity, high_helicity in zip(
        low_precision.values[0], high_precision.values[0], strict=True
    ):
        for low_value, high_value in zip(low_helicity, high_helicity, strict=True):
            assert isinstance(low_value, Decimal)
            assert isinstance(high_value, Decimal)
            assert abs(low_value - high_value) <= max(
                abs(high_value), Decimal(1)
            ) * Decimal("1e-28")

    if case_name == "scalars_2to2_lc":
        analytic = scalar_contact_2to2()
    else:
        analytic_momenta = tuple(
            tuple(Decimal(str(component)) for component in vector)
            for vector in reference_momenta[0]
        )
        analytic = scalar_gravity_2to2(analytic_momenta)
    assert high_precision.helicity_ids == analytic.helicity_ids
    precise_total = high_precision.total()[0]
    assert isinstance(precise_total, Decimal)
    assert abs(precise_total - analytic.total) <= max(
        abs(analytic.total), Decimal(1)
    ) * Decimal("1e-14")
    for row, expected in zip(high_precision.values[0], analytic.resolved, strict=True):
        assert isinstance(row[0], Decimal)
        assert abs(row[0] - expected) <= max(abs(expected), Decimal(1)) * Decimal(
            "1e-14"
        )


@pytest.mark.parametrize("source_kind", tuple(EXTERNAL_SM_SOURCES))
@pytest.mark.parametrize("accuracy", ("lc", "nlc", "full"))
def test_current_source_external_sm_matches_builtin_reference(
    tmp_path: Path,
    source_kind: str,
    accuracy: str,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    processes = {process["id"]: process for process in REFERENCE_PAYLOAD["processes"]}
    cases = {case["id"]: case for case in REFERENCE_PAYLOAD["cases"]}
    points = {point["id"]: point for point in REFERENCE_PAYLOAD["points"]}
    config = RunConfig(
        action="generate",
        model=ModelConfig(cache=False),
        color=ColorConfig(accuracy=accuracy),
    )
    artifact = tmp_path / f"external-sm-{source_kind}-{accuracy}"
    Generator(config).generate(
        ProcessSet.from_expressions(
            tuple(
                processes[process_id]["expression"]
                for process_id in EXTERNAL_SM_PROCESS_IDS
            ),
            names=EXTERNAL_SM_PROCESS_IDS,
        ),
        artifact,
        model=ModelSource.from_path(EXTERNAL_SM_SOURCES[source_kind]),
    )

    for process_id in EXTERNAL_SM_PROCESS_IDS:
        case = cases[f"case:{process_id}:{accuracy}"]
        execution = json.loads(
            (artifact / "processes" / process_id / "execution.json").read_text(
                encoding="utf-8"
            )
        )
        physics = json.loads(
            (artifact / "processes" / process_id / "physics.json").read_text(
                encoding="utf-8"
            )
        )
        topology = CURRENT_BUILTIN_TOPOLOGY.get(case["id"], case["topology"])
        assert execution["dag_summary"]["current_count"] == topology["currents"]
        assert execution["dag_summary"]["interaction_count"] == topology["interactions"]
        assert execution["dag_summary"]["amplitude_root_count"] == topology["roots"]
        assert len(physics["reduction"]["groups"]) == topology["reduction_groups"]

        runtime = Runtime.load(artifact, process=process_id)
        assert runtime.physics.helicity_coverage == "complete"
        assert runtime.physics.color_coverage == (
            "complete" if accuracy == "lc" else "contracted"
        )
        expected_helicities = tuple(item["id"] for item in case["axes"]["helicities"])
        expected_colors = tuple(item["id"] for item in case["axes"]["colors"])
        for observation in case["observations"]:
            point = points[observation["point_id"]]
            momenta = (
                tuple(
                    tuple(float(component) for component in vector)
                    for vector in point["momenta"]
                ),
            )
            total = runtime.evaluate(momenta)[0]
            resolved = runtime.evaluate_resolved(momenta)
            expected_total = float(observation["total"])

            assert total.real == pytest.approx(
                expected_total,
                rel=1.0e-10,
                abs=1.0e-12,
            )
            assert total.imag == pytest.approx(
                0.0,
                abs=max(1.0e-15, abs(expected_total) * 1.0e-12),
            )
            assert resolved.total()[0] == pytest.approx(
                total,
                rel=1.0e-12,
                abs=1.0e-12,
            )
            assert resolved.helicity_ids == expected_helicities
            assert resolved.color_ids == expected_colors
            for helicity_index, expected_row in enumerate(observation["values"]):
                for color_index, expected_text in enumerate(expected_row):
                    expected = float(expected_text)
                    actual = resolved.values[0][helicity_index][color_index]
                    assert actual.real == pytest.approx(
                        expected,
                        rel=1.0e-10,
                        abs=1.0e-12,
                    )
                    assert actual.imag == pytest.approx(
                        0.0,
                        abs=max(1.0e-15, abs(expected) * 1.0e-12),
                    )


def test_external_sm_color_dummy_relabeling_preserves_resolved_runtime(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    reference, momenta, expected_resolved = _case_payload("case:sm_gg_gg:lc")
    artifact = tmp_path / "color-dummy-relabelled-sm"
    config = RunConfig(
        action="generate",
        model=ModelConfig(cache=False),
        color=ColorConfig(accuracy="lc"),
    )
    Generator(config).generate(
        reference["process"],
        artifact,
        model=ModelSource.from_path(_write_color_dummy_relabelled_sm(tmp_path)),
    )

    outer = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    process_id = outer["processes"][0]["id"]
    execution = json.loads(
        (artifact / "processes" / process_id / "execution.json").read_text(
            encoding="utf-8"
        )
    )
    assert execution["dag_summary"]["current_count"] == reference["topology"][
        "current_count"
    ]
    assert execution["dag_summary"]["interaction_count"] == reference["topology"][
        "interaction_count"
    ]
    assert execution["dag_summary"]["amplitude_root_count"] == reference["topology"][
        "amplitude_root_count"
    ]

    runtime_momenta = tuple(
        tuple(tuple(float(component) for component in vector) for vector in point)
        for point in momenta
    )
    runtime = Runtime.load(artifact)
    resolved = runtime.evaluate_resolved(runtime_momenta)
    precise = runtime.evaluate_resolved(runtime_momenta, precision=80)
    expected_helicity_ids = tuple(expected_resolved)
    expected_color_ids = tuple(next(iter(expected_resolved.values())))
    assert resolved.helicity_ids == expected_helicity_ids
    assert resolved.color_ids == expected_color_ids
    assert precise.helicity_ids == expected_helicity_ids
    assert precise.color_ids == expected_color_ids
    for helicity_index, helicity_id in enumerate(expected_helicity_ids):
        for color_index, color_id in enumerate(expected_color_ids):
            expected = expected_resolved[helicity_id][color_id]
            actual = resolved.values[0][helicity_index][color_index]
            high_precision = precise.values[0][helicity_index][color_index]
            assert actual.real == pytest.approx(expected, rel=1.0e-10, abs=1.0e-12)
            assert actual.imag == pytest.approx(0.0, abs=1.0e-12)
            assert float(high_precision) == pytest.approx(
                expected,
                rel=1.0e-10,
                abs=1.0e-12,
            )
    assert resolved.total()[0].real == pytest.approx(
        reference["total"],
        rel=1.0e-10,
        abs=1.0e-12,
    )


def test_external_sm_mass_overrides_refresh_sources_and_derived_masses(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    config = RunConfig(
        action="generate",
        model=ModelConfig(cache=False),
        color=ColorConfig(accuracy="lc"),
    )
    artifact = tmp_path / "external-sm-runtime-mass"
    Generator(config).generate(
        "u d~ > w+",
        artifact,
        model=ModelSource.from_path(EXTERNAL_SM_SOURCES["json"]),
    )

    outer = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    process_id = outer["processes"][0]["id"]
    process_root = artifact / "processes" / process_id
    execution = json.loads(
        (process_root / "execution.json").read_text(encoding="utf-8")
    )
    w_record = next(
        particle
        for particle in execution["runtime_schema"]["model"]["particles"]
        if particle["pdg"] == 24
    )
    assert w_record["mass_parameter"] == "MW"
    assert {
        parameter.get("runtime_name")
        for parameter in execution["runtime_schema"]["model_parameters"]
        if parameter["kind"] == "derived_parameter_component"
    } >= {"MW"}

    validation = json.loads(
        (process_root / "validation-momenta.json").read_text(encoding="utf-8")
    )
    momenta = (
        tuple(
            tuple(float(component) for component in particle["momentum"])
            for particle in validation["points"][0]
        ),
    )
    runtime = Runtime.load(artifact)
    baseline = runtime.evaluate(momenta)[0]
    runtime.set_model_parameters({"MZ": 100.0})
    changed = runtime.evaluate(momenta)[0]
    runtime.set_model_parameters({"MZ": 91.188})
    restored = runtime.evaluate(momenta)[0]

    assert abs(changed - baseline) > max(1.0e-14, abs(baseline) * 1.0e-8)
    assert restored == pytest.approx(baseline, rel=1.0e-12, abs=1.0e-15)


def test_runtime_rejects_external_mass_class_changes_atomically(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    artifact = tmp_path / "external-sm-runtime-mass-class"
    Generator(
        RunConfig(
            action="generate",
            model=ModelConfig(cache=False),
            color=ColorConfig(accuracy="lc"),
        )
    ).generate(
        "d d~ > z",
        artifact,
        model=ModelSource.from_path(EXTERNAL_SM_SOURCES["json"]),
    )
    outer = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    process_id = outer["processes"][0]["id"]
    validation = json.loads(
        (artifact / "processes" / process_id / "validation-momenta.json").read_text(
            encoding="utf-8"
        )
    )
    momenta = (
        tuple(
            tuple(float(component) for component in particle["momentum"])
            for particle in validation["points"][0]
        ),
    )
    runtime = Runtime.load(artifact)
    baseline = runtime.evaluate(momenta)[0]

    with pytest.raises(EvaluationError, match=r"mass class.*regenerate"):
        runtime.set_model_parameters({"MZ": 0.0})

    assert runtime.evaluate(momenta)[0] == pytest.approx(
        baseline,
        rel=1.0e-12,
        abs=1.0e-15,
    )
