# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from pyamplicol import Generator, ModelSource, Runtime
from pyamplicol.api.errors import EvaluationError
from pyamplicol.config import ColorConfig, ModelConfig, RunConfig

REFERENCE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "reference" / "physics-v1.json"
)
CASE_NAMES = {
    "lc": "builtin_sm_ddbar_z_lc",
    "nlc": "builtin_sm_ddbar_z_nlc",
    "full": "builtin_sm_ddbar_z_full",
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


def _reference_case(
    accuracy: str,
) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    fixture = json.loads(REFERENCE.read_text(encoding="utf-8"))
    cases = fixture["cases"]
    case = cases[CASE_NAMES[accuracy]]
    momenta = (
        cases[case["momenta_from"]]["momenta"]
        if "momenta_from" in case
        else case["momenta"]
    )
    resolved = (
        cases[case["resolved_from"]]["resolved"]
        if "resolved_from" in case
        else case["resolved"]
    )
    return case, momenta, resolved


def _named_reference_case(name: str) -> tuple[dict[str, Any], list[Any]]:
    fixture = json.loads(REFERENCE.read_text(encoding="utf-8"))
    cases = fixture["cases"]
    case = cases[name]
    momenta = (
        cases[case["momenta_from"]]["momenta"]
        if "momenta_from" in case
        else case["momenta"]
    )
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
    process_root = artifact / "processes" / reference["process_id"]
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

    precise = runtime.evaluate_resolved(momenta, precision=32)
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


@pytest.mark.parametrize("source_kind", tuple(EXTERNAL_SM_SOURCES))
def test_current_source_external_sm_matches_builtin_reference(
    tmp_path: Path,
    source_kind: str,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    reference, reference_momenta = _named_reference_case("builtin_sm_ddbar_zg_lc")
    config = RunConfig(
        action="generate",
        model=ModelConfig(cache=False),
        color=ColorConfig(accuracy="lc"),
    )
    artifact = tmp_path / f"external-sm-{source_kind}"
    Generator(config).generate(
        reference["process"],
        artifact,
        model=ModelSource.from_path(EXTERNAL_SM_SOURCES[source_kind]),
    )

    execution = json.loads(
        (artifact / "processes" / reference["process_id"] / "execution.json").read_text(
            encoding="utf-8"
        )
    )
    for name, expected in reference["topology"].items():
        if name == "interaction_evaluation_count":
            continue
        assert execution["dag_summary"][name] == expected

    runtime = Runtime.load(artifact)
    momenta = tuple(
        tuple(tuple(float(component) for component in vector) for vector in point)
        for point in reference_momenta
    )
    total = runtime.evaluate(momenta)
    resolved = runtime.evaluate_resolved(momenta)

    assert total[0].real == pytest.approx(reference["total"], rel=1.0e-10)
    assert total[0].imag == pytest.approx(0.0, abs=1.0e-15)
    assert resolved.total()[0] == pytest.approx(total[0], rel=1.0e-12)
    helicity_index = {value: index for index, value in enumerate(resolved.helicity_ids)}
    color_index = {value: index for index, value in enumerate(resolved.color_ids)}
    for helicity_id, expected_colors in reference["resolved"].items():
        for color_id, expected_value in expected_colors.items():
            actual = resolved.values[0][helicity_index[helicity_id]][
                color_index[color_id]
            ]
            assert actual.real == pytest.approx(expected_value, rel=1.0e-10)
            assert actual.imag == pytest.approx(0.0, abs=1.0e-15)


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
