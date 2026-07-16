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

from pyamplicol import Generator, ModelSource, ProcessSet, Runtime
from pyamplicol.api.errors import EvaluationError
from pyamplicol.config import ColorConfig, ModelConfig, RunConfig
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
    assert execution["schema_version"] == 3
    assert execution["kind"] == "pyamplicol-runtime-execution"
    assert execution["runtime_schema"]["kind"] == ("pyamplicol-runtime-execution-plan")
    assert all(
        parameter["name"] != "runtime.lc_sector_id"
        for parameter in execution["runtime_schema"]["model_parameters"]
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
        topology = case["topology"]
        assert execution["dag_summary"]["current_count"] == topology["currents"]
        assert execution["dag_summary"]["interaction_count"] == topology["interactions"]
        assert execution["dag_summary"]["amplitude_root_count"] == topology["roots"]

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
