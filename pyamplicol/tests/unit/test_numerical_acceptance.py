# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools.developer import numerical_acceptance as acceptance_module
from tools.developer.numerical_acceptance import (
    ACCEPTANCE_LANES,
    CATALOG_CASE_COUNT,
    CATALOG_MAX_N_FINAL,
    EXTRA_FULL_COLOUR_CASES,
    FULL_CANDIDATE_PRECISION_DIGITS,
    NATIVE_PRECISION_DIGITS,
    RELATIVE_TOLERANCE,
    MadGraphReferenceIdentity,
    NumericalAcceptanceError,
    NumericalAcceptanceMismatch,
    assert_relative_match,
    build_fixture_payload,
    capture_acceptance_fixture,
    catalog_cases,
    compare_relative,
    current_model_identity,
    ingest_authenticated_extra_madgraph_root,
    ingest_authenticated_madgraph_wave_root,
    lane_case_specs,
    load_acceptance_fixture,
    parse_acceptance_fixture,
    write_acceptance_fixture,
)
from tools.performance_report.madgraph import (
    MADGRAPH_DRIVER_SOURCE_SHA256,
    madgraph_command_card,
)
from tools.performance_report.runner import point_digest


@pytest.fixture(autouse=True)
def synthetic_validation_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep wire-contract unit tests independent of a staged native runtime."""

    def point(process: str) -> tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...]:
        initial, final = process.split(">", maxsplit=1)
        width = len(initial.split()) + len(final.split())
        return tuple(
            (Decimal(index + 1), Decimal(0), Decimal(0), Decimal(0))
            for index in range(width)
        )

    monkeypatch.setattr(acceptance_module, "validation_momenta", point)


@pytest.fixture
def synthetic_payload() -> dict[str, object]:
    catalog = catalog_cases()
    all_cases = (*catalog, *EXTRA_FULL_COLOUR_CASES)
    full = {
        case.case_id: Decimal(index + 1) * Decimal("0.000000000000000000000000000001")
        for index, case in enumerate(all_cases)
    }
    lc = {
        case.case_id: Decimal(index + 101) * Decimal("0.000000000000000000000000000001")
        for index, case in enumerate(catalog)
    }
    nlc = {
        case.case_id: Decimal(index + 201) * Decimal("0.000000000000000000000000000001")
        for index, case in enumerate(catalog)
    }
    model = current_model_identity()
    return build_fixture_payload(
        full_values=full,
        lc_values=lc,
        nlc_values=nlc,
        full_reference=MadGraphReferenceIdentity(
            madgraph_version="synthetic-unit-test",
            model_source_sha256="1" * 64,
            driver_sha256=MADGRAPH_DRIVER_SOURCE_SHA256,
            external_parameters_sha256=model.external_parameters_sha256,
        ),
        captured_source_revision="a" * 40,
    )


def test_catalog_census_extra_cases_and_exact_lane_matrix() -> None:
    cases = catalog_cases()
    assert len(cases) == CATALOG_CASE_COUNT == 33
    assert Counter(case.n_final for case in cases) == {1: 2, 2: 8, 3: 9, 4: 14}
    assert all(case.n_final <= CATALOG_MAX_N_FINAL for case in cases)
    assert [case.case_id for case in EXTRA_FULL_COLOUR_CASES] == [
        "extra:identical-u-four-lines",
        "extra:identical-electron-three-pair",
    ]
    assert {case.process for case in cases}.isdisjoint(
        case.process for case in EXTRA_FULL_COLOUR_CASES
    )
    assert [
        (lane.accuracy, lane.mode, lane.lc_flow_layout)
        for lane in ACCEPTANCE_LANES
    ] == [
        ("full", "recurrence", "topology-replay"),
        ("full", "eager", "topology-replay"),
        ("full", "compiled", "topology-replay"),
        ("full", "on-the-fly", "topology-replay"),
        ("lc", "recurrence", "topology-replay"),
        ("lc", "eager", "topology-replay"),
        ("lc", "compiled", "topology-replay"),
        ("lc", "on-the-fly", "topology-replay"),
        ("lc", "recurrence", "all-flow-union"),
        ("lc", "eager", "all-flow-union"),
        ("lc", "compiled", "all-flow-union"),
        ("nlc", "recurrence", "topology-replay"),
        ("nlc", "eager", "topology-replay"),
        ("nlc", "compiled", "topology-replay"),
        ("nlc", "on-the-fly", "topology-replay"),
    ]
    assert len({lane.lane_id for lane in ACCEPTANCE_LANES}) == 15
    assert {
        lane.mode
        for lane in ACCEPTANCE_LANES
        if lane.lc_flow_layout == "all-flow-union"
    } == {"recurrence", "eager", "compiled"}
    with pytest.raises(ValueError, match="compact query-local LC artifact"):
        acceptance_module.AcceptanceLane(
            "lc", "on-the-fly", 2, "all-flow-union"
        )
    assert {
        lane.accuracy for lane in ACCEPTANCE_LANES if lane.mode == "on-the-fly"
    } == {"full", "lc", "nlc"}


def test_synthetic_fixture_round_trip_binds_model_points_and_provenance(
    tmp_path: Path,
    synthetic_payload: dict[str, object],
) -> None:
    path = tmp_path / "synthetic-numerical-acceptance.json"
    fixture = write_acceptance_fixture(synthetic_payload, path)
    loaded = load_acceptance_fixture(path)

    assert loaded == fixture
    assert loaded.model == current_model_identity()
    assert loaded.full_reference.external_parameters_sha256 == (
        loaded.model.external_parameters_sha256
    )
    assert loaded.captured_source_revision == "a" * 40
    assert len(loaded.catalog) == 33
    assert len(loaded.extra_full_colour) == 2
    for case in (*loaded.catalog, *loaded.extra_full_colour):
        assert case.momenta == acceptance_module.validation_momenta(case.spec.process)

    with pytest.raises(NumericalAcceptanceError, match="refusing to overwrite"):
        write_acceptance_fixture(synthetic_payload, path)


def test_fixture_loader_rejects_changed_points_model_and_wire_contract(
    synthetic_payload: dict[str, object],
) -> None:
    changed_point = copy.deepcopy(synthetic_payload)
    changed_point["catalog_cases"][0]["momenta"][0][0] = "999"  # type: ignore[index]
    with pytest.raises(NumericalAcceptanceError, match="momenta differ"):
        parse_acceptance_fixture(changed_point)

    changed_model = copy.deepcopy(synthetic_payload)
    changed_model["model"]["source_sha256"] = "f" * 64  # type: ignore[index]
    with pytest.raises(NumericalAcceptanceError, match="model identity differs"):
        parse_acceptance_fixture(changed_model)

    changed_comparison = copy.deepcopy(synthetic_payload)
    changed_comparison["comparison"]["absolute_tolerance"] = "0"  # type: ignore[index]
    with pytest.raises(NumericalAcceptanceError, match="relative-only"):
        parse_acceptance_fixture(changed_comparison)

    changed_driver = copy.deepcopy(synthetic_payload)
    changed_driver["references"]["full"]["driver_sha256"] = "f" * 64  # type: ignore[index]
    with pytest.raises(NumericalAcceptanceError, match="canonical adapter driver"):
        parse_acceptance_fixture(changed_driver)

    extra_field = copy.deepcopy(synthetic_payload)
    extra_field["unreviewed"] = True
    with pytest.raises(NumericalAcceptanceError, match="fields differ"):
        parse_acceptance_fixture(extra_field)


def test_relative_comparator_has_no_absolute_escape_hatch() -> None:
    reference = Decimal("0.000000000000000266945931894745456")
    inside = reference * (Decimal(1) + RELATIVE_TOLERANCE / 2)
    outside = reference * (Decimal(1) + RELATIVE_TOLERANCE * 2)

    assert compare_relative(inside, reference).passed
    assert not compare_relative(outside, reference).passed
    assert compare_relative(Decimal(0), Decimal(0)).passed
    assert not compare_relative(Decimal(0), reference).passed
    assert not compare_relative(-reference, reference).passed
    assert not compare_relative(
        complex(float(reference), float(reference)), reference
    ).passed
    with pytest.raises(NumericalAcceptanceMismatch, match="strict relative bound"):
        assert_relative_match(outside, reference, context="synthetic/sign-gate")


def test_lane_helpers_keep_ordered_process_sets_static_and_artifact_safe() -> None:
    for lane in ACCEPTANCE_LANES:
        specs = lane_case_specs(lane)
        assert len(specs) == 33
        assert tuple((spec.n_final, spec.family_id or 1000) for spec in specs) == tuple(
            sorted(
                ((spec.n_final, spec.family_id or 1000) for spec in specs),
            )
        )
        assert all(":" not in spec.artifact_name for spec in specs)
        assert all(spec.artifact_name[0].isalpha() for spec in specs)
        assert lane.jit_optimization_level == (3 if lane.mode == "compiled" else 2)
        config = acceptance_module.lane_run_config(lane)
        assert config.color.lc_flow_layout.value == lane.lc_flow_layout
        if lane.accuracy == "full":
            assert lane_case_specs(lane, group="extra") == EXTRA_FULL_COLOUR_CASES
        else:
            with pytest.raises(NumericalAcceptanceError, match="no 'extra'"):
                lane_case_specs(lane, group="extra")

    assert FULL_CANDIDATE_PRECISION_DIGITS == 200
    assert NATIVE_PRECISION_DIGITS == 16


def test_lane_precision_policy_keeps_on_the_fly_at_p16() -> None:
    for lane in ACCEPTANCE_LANES:
        expected = (NATIVE_PRECISION_DIGITS,)
        if lane.accuracy == "full" and lane.mode != "on-the-fly":
            expected += (FULL_CANDIDATE_PRECISION_DIGITS,)
        assert lane.evaluation_precisions == expected
        assert lane.evaluation_precisions_for("catalog") == expected
        assert lane.evaluation_precisions_for("extra") == (
            NATIVE_PRECISION_DIGITS,
        )


def test_runtime_lane_validation_binds_mode_and_color_accuracy() -> None:
    lane = acceptance_module.AcceptanceLane("lc", "recurrence", 2)
    runtime = SimpleNamespace(
        execution_mode="recurrence",
        physics=SimpleNamespace(color_accuracy="lc"),
    )
    acceptance_module._validate_runtime_lane(runtime, lane, context="synthetic")

    runtime.physics.color_accuracy = "full"
    with pytest.raises(NumericalAcceptanceError, match="color_accuracy='full'"):
        acceptance_module._validate_runtime_lane(runtime, lane, context="synthetic")
    runtime.physics.color_accuracy = "lc"
    runtime.execution_mode = "eager"
    with pytest.raises(NumericalAcceptanceError, match="mode='eager'"):
        acceptance_module._validate_runtime_lane(runtime, lane, context="synthetic")


def test_artifact_lane_validation_binds_all_flow_union(tmp_path: Path) -> None:
    artifact = tmp_path / "union"
    (artifact / "config").mkdir(parents=True)
    (artifact / "config" / "effective.toml").write_text(
        """
[color]
accuracy = "lc"
lc_flow_layout = "all-flow-union"
[evaluator]
execution_mode = "recurrence"
""".lstrip(),
        encoding="utf-8",
    )
    union = acceptance_module.AcceptanceLane(
        "lc", "recurrence", 2, "all-flow-union"
    )
    acceptance_module._validate_artifact_lane(artifact, union)

    topology = acceptance_module.AcceptanceLane("lc", "recurrence", 2)
    with pytest.raises(NumericalAcceptanceError, match="effective lane"):
        acceptance_module._validate_artifact_lane(artifact, topology)


def test_capture_runtime_identity_requires_repo_venv_site_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = acceptance_module.ROOT / ".venv"
    package_file = next(
        (prefix / "lib").glob("python*/site-packages/pyamplicol/__init__.py")
    )
    package = ModuleType("pyamplicol")
    package.__file__ = str(package_file)
    package.__path__ = []  # type: ignore[attr-defined]
    internal = ModuleType("pyamplicol._internal")
    internal.__path__ = []  # type: ignore[attr-defined]
    versions = ModuleType("pyamplicol._internal.versions")
    versions.active_native_source_identity = lambda: (  # type: ignore[attr-defined]
        "a" * 40,
        "b" * 64,
    )
    monkeypatch.setitem(sys.modules, "pyamplicol", package)
    monkeypatch.setitem(sys.modules, "pyamplicol._internal", internal)
    monkeypatch.setitem(sys.modules, "pyamplicol._internal.versions", versions)
    monkeypatch.setattr(acceptance_module.sys, "prefix", str(prefix))
    monkeypatch.setattr(
        acceptance_module.importlib.metadata,
        "version",
        lambda _name: "synthetic",
    )

    identity = acceptance_module._current_runtime_identity()
    assert identity.source_revision == "a" * 40
    assert identity.native_build_inputs_sha256 == "b" * 64

    package.__file__ = str(
        acceptance_module.ROOT / "src" / "pyamplicol" / "__init__.py"
    )
    with pytest.raises(NumericalAcceptanceError, match=r"\.venv site-packages"):
        acceptance_module._current_runtime_identity()


def test_json_loader_rejects_duplicate_fields(
    tmp_path: Path,
    synthetic_payload: dict[str, object],
) -> None:
    path = tmp_path / "duplicates.json"
    encoded = json.dumps(synthetic_payload)
    path.write_text(encoded[:-1] + ',"kind":"duplicate"}', encoding="ascii")
    with pytest.raises(NumericalAcceptanceError, match="duplicate JSON field"):
        load_acceptance_fixture(path)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _point_payload(process: str) -> list[list[str]]:
    return [
        [str(component) for component in row]
        for row in acceptance_module.validation_momenta(process)
    ]


def _measurement_payload(
    spec: acceptance_module.AcceptanceCaseSpec,
    *,
    driver_sha256: str = MADGRAPH_DRIVER_SOURCE_SHA256,
) -> dict[str, object]:
    point = (
        tuple(
            tuple(float(component) for component in row)
            for row in acceptance_module.validation_momenta(spec.process)
        ),
    )
    external_sha = current_model_identity().external_parameters_sha256
    command_card = madgraph_command_card(spec.process)
    card = {
        "binary64_exact_match": True,
        "external_parameter_count": 1,
        "external_parameters_sha256": external_sha,
        "format": "%.14e",
    }
    return {
        "status": "ok",
        "matrix_element": 1.25,
        "validation": {
            "status": "ok",
            "method": "independent-madgraph-tree-level-oracle",
            "point_digest": point_digest(point),
        },
        "provenance": {
            "method": "madgraph-standalone-custom-fortran-driver",
            "command_card": command_card,
            "command_card_sha256": hashlib.sha256(
                command_card.encode("utf-8")
            ).hexdigest(),
            "report_momenta": point,
            "model": {"name": "sm", "source_sha256": "1" * 64},
            "exact_param_card": dict(card),
            "default_restriction": dict(card),
            "version": "synthetic-madgraph",
            "driver_sha256": driver_sha256,
        },
    }


def test_madgraph_measurement_requires_canonical_driver_without_timing_census() -> None:
    spec = catalog_cases()[0]
    value, identity = acceptance_module._madgraph_measurement(
        _measurement_payload(spec),
        spec,
    )
    assert value == Decimal("1.25")
    assert identity.driver_sha256 == MADGRAPH_DRIVER_SOURCE_SHA256

    changed = _measurement_payload(spec, driver_sha256="f" * 64)
    with pytest.raises(NumericalAcceptanceError, match="authority contract"):
        acceptance_module._madgraph_measurement(changed, spec)


def _stub_raw_measurements(
    monkeypatch: pytest.MonkeyPatch,
    values: dict[str, Decimal],
) -> MadGraphReferenceIdentity:
    model = current_model_identity()
    identity = MadGraphReferenceIdentity(
        madgraph_version="synthetic-gate",
        model_source_sha256="1" * 64,
        driver_sha256=MADGRAPH_DRIVER_SOURCE_SHA256,
        external_parameters_sha256=model.external_parameters_sha256,
    )

    def parse(
        measurement: object,
        spec: acceptance_module.AcceptanceCaseSpec,
    ) -> tuple[Decimal, MadGraphReferenceIdentity]:
        assert measurement == {"case_id": spec.case_id}
        return values[spec.case_id], identity

    monkeypatch.setattr(acceptance_module, "_madgraph_measurement", parse)
    return identity


def _synthetic_wave_root(root: Path, values: dict[str, Decimal]) -> Path:
    specs = catalog_cases()
    _write_json(
        root / "manifest.json",
        {
            "kind": "pyamplicol-madgraph-ufo-sm-full-p200-wave-gate-v1",
            "status": "running",
            "seed": 101,
            "max_n": 6,
            "ordering": "n-final-then-family-id-then-mode",
            "plan": [
                {
                    "cell_ordinal": ordinal,
                    "family_id": spec.family_id,
                    "n_final": spec.n_final,
                    "process": spec.process,
                    "process_key": spec.family_key,
                }
                for ordinal, spec in enumerate(specs)
            ],
        },
    )
    for spec in specs:
        case_root = root / f"n{spec.n_final}" / f"family-{spec.family_id:02d}"
        _write_json(
            case_root / "point-seed-101.json",
            _point_payload(spec.process),
        )
        _write_json(
            case_root / "madgraph-measurement.json",
            {"case_id": spec.case_id},
        )
        _write_json(
            case_root / "madgraph-authority.json",
            {
                "process": spec.process,
                "n_final": spec.n_final,
                "seed": 101,
                "madgraph_value": str(values[spec.case_id]),
            },
        )
    return root


def _synthetic_stress_root(root: Path, values: dict[str, Decimal]) -> Path:
    keys = ("uu_four_identical_lines", "ee_four_identical_lines")
    _write_json(
        root / "manifest.json",
        {
            "kind": "pyamplicol-authenticated-noncatalog-n6-madgraph-stress-v1",
            "status": "ok",
            "stage": "complete",
            "seed": 101,
            "process_order": [
                {"key": key, "process": spec.process, "n_final": 6}
                for key, spec in zip(keys, EXTRA_FULL_COLOUR_CASES, strict=True)
            ],
            "catalog_policy": "direct synthetic CellSpec; REPORT_CATALOG not modified",
        },
    )
    for case_ordinal, (key, spec) in enumerate(
        zip(keys, EXTRA_FULL_COLOUR_CASES, strict=True)
    ):
        case_root = root / f"case-{case_ordinal + 1:02d}-{key}"
        point = _point_payload(spec.process)
        _write_json(case_root / "point-seed-101.json", point)
        _write_json(
            case_root / "madgraph-measurement.json",
            {"case_id": spec.case_id},
        )
        _write_json(
            case_root / "madgraph-authority-verification.json",
            {
                "process": spec.process,
                "seed": 101,
                "point": point,
                "matrix_element_binary64": str(values[spec.case_id]),
                "checks": {
                    "status": True,
                    "exact_seed_101_point": True,
                    "exact_command_card": True,
                    "custom_fortran_method": True,
                    "custom_driver_digest": True,
                    "exact_generated_process": True,
                },
            },
        )
    return root


def test_wave_ingestion_reuses_only_madgraph_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        spec.case_id: Decimal(index + 1) for index, spec in enumerate(catalog_cases())
    }
    identity = _stub_raw_measurements(monkeypatch, values)
    root = _synthetic_wave_root(tmp_path / "wave", values)

    ingested = ingest_authenticated_madgraph_wave_root(root)
    assert ingested.values == values
    assert ingested.identity == identity

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for terminal_status in ("validation-failed", "error"):
        manifest["status"] = terminal_status
        _write_json(manifest_path, manifest)
        with pytest.raises(NumericalAcceptanceError, match="cannot supply"):
            ingest_authenticated_madgraph_wave_root(root)
    manifest["status"] = "running"
    _write_json(manifest_path, manifest)

    last = catalog_cases()[-1]
    authority_path = (
        root / "n4" / f"family-{last.family_id:02d}" / "madgraph-authority.json"
    )
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["madgraph_value"] = "999"
    _write_json(authority_path, authority)
    with pytest.raises(NumericalAcceptanceError, match="authority"):
        ingest_authenticated_madgraph_wave_root(root)


def test_stress_ingestion_reuses_only_madgraph_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        spec.case_id: Decimal(index + 101)
        for index, spec in enumerate(EXTRA_FULL_COLOUR_CASES)
    }
    identity = _stub_raw_measurements(monkeypatch, values)
    root = _synthetic_stress_root(tmp_path / "stress", values)

    ingested = ingest_authenticated_extra_madgraph_root(root)
    assert ingested.values == values
    assert ingested.identity == identity

    authority_path = (
        root
        / "case-01-uu_four_identical_lines"
        / "madgraph-authority-verification.json"
    )
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["checks"]["custom_driver_digest"] = False
    _write_json(authority_path, authority)
    with pytest.raises(NumericalAcceptanceError, match="authority"):
        ingest_authenticated_extra_madgraph_root(root)


def test_capture_preflight_rejects_bidirectional_and_input_path_overlap(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "fixture.json"
    with pytest.raises(NumericalAcceptanceError, match=r"outside.*work root"):
        capture_acceptance_fixture(
            madgraph_wave_root=tmp_path / "wave",
            extra_madgraph_root=tmp_path / "stress",
            work_root=destination / "work",
            output=destination,
        )

    wave_root = tmp_path / "authenticated-wave"
    wave_root.mkdir()
    with pytest.raises(NumericalAcceptanceError, match="must not overlap"):
        capture_acceptance_fixture(
            madgraph_wave_root=wave_root,
            extra_madgraph_root=tmp_path / "stress",
            work_root=tmp_path / "work",
            output=wave_root / "fixture.json",
        )
