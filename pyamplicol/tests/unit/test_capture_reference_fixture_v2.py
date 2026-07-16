# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import subprocess
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from tools.developer import reference_capture as capture
from tools.developer.reference_capture import artifacts, evidence, pipeline, provenance

_REFERENCE_SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[2] / "schemas/reference-physics-v2.schema.json"
    ).read_text(encoding="utf-8")
)
_POINT_VALIDATOR = Draft202012Validator(
    {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": _REFERENCE_SCHEMA["$defs"],
        "$ref": "#/$defs/point",
    }
)


def _run_git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    )


def _masses(expression: str) -> tuple[Decimal, ...]:
    if expression == "d d~ > z":
        return (Decimal(0), Decimal(0), Decimal("91.188"))
    if expression == "d d~ > z g":
        return (Decimal(0), Decimal(0), Decimal("91.188"), Decimal(0))
    if expression == "d d~ > z g g":
        return (
            Decimal(0),
            Decimal(0),
            Decimal("91.188"),
            Decimal(0),
            Decimal(0),
        )
    return (Decimal(0), Decimal(0), Decimal(0), Decimal(0))


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (Decimal("1.2300"), "1.23"),
        (Decimal("-0.000"), "0"),
        (1000, "1000"),
        (0.1, "0.10000000000000001"),
        ("1e-12", "0.000000000001"),
    ),
)
def test_canonical_decimal_is_fixed_point_and_binary64_honest(
    value: Decimal | float | int | str,
    expected: str,
) -> None:
    assert capture.canonical_decimal(value) == expected


def test_compiled_model_contract_hash_excludes_only_timing_diagnostics() -> None:
    compiled = {
        "schema_version": 6,
        "kind": "pyamplicol-compiled-model",
        "source": {"digest": "abc"},
        "ir": {"particles": [{"pdg_code": 1}]},
        "parameter_defaults": {"ZERO": [Decimal("0.0"), Decimal("0.0")]},
        "conversion_seconds": Decimal("1.25"),
        "phase_timings": {"lower": Decimal("0.5")},
    }
    baseline = artifacts._compiled_model_contract_sha256(compiled)

    compiled["conversion_seconds"] = Decimal("99.0")
    compiled["phase_timings"] = {"lower": Decimal("42.0")}
    assert artifacts._compiled_model_contract_sha256(compiled) == baseline

    compiled["ir"] = {"particles": [{"pdg_code": 2}]}
    assert artifacts._compiled_model_contract_sha256(compiled) != baseline


def test_canonical_decimal_rejects_non_finite_values() -> None:
    with pytest.raises(capture.CaptureError, match="non-finite"):
        capture.canonical_decimal(float("inf"))


@pytest.mark.parametrize(
    ("process_id", "expression", "expected_count"),
    (
        ("sm_ddbar_z", "d d~ > z", 1),
        ("sm_ddbar_zg", "d d~ > z g", 4),
        ("sm_ddbar_zgg", "d d~ > z g g", 4),
        (
            "scalars_2to2",
            "scalar_0 scalar_0 > scalar_0 scalar_0",
            4,
        ),
        (
            "scalar_gravity_2to2",
            "scalar_0 scalar_0 > graviton graviton",
            4,
        ),
    ),
)
def test_reference_points_are_deterministic_canonical_and_valid(
    process_id: str,
    expression: str,
    expected_count: int,
) -> None:
    first = capture.build_reference_points(process_id, expression)
    second = capture.build_reference_points(process_id, expression)

    assert first == second
    assert len(first) == expected_count
    for point in first:
        capture.validate_point_kinematics(point, _masses(expression))
        payload = point.as_payload()
        _POINT_VALIDATOR.validate(payload)
        numeric_strings = [
            str(component) for momentum in payload["momenta"] for component in momentum
        ]
        assert all("e" not in value.lower() for value in numeric_strings)
        assert capture.point_input_sha256(payload) == capture.point_input_sha256(
            point.as_payload()
        )

    if expression == "d d~ > z":
        assert [point.point_class for point in first] == ["canonical"]
    else:
        assert [point.point_class for point in first] == [
            "generic",
            "generic",
            "generic",
            "stress",
        ]
        assert all(point.arithmetic_precision_bits == 53 for point in first[:3])
        assert all(point.round_trip_decimal_digits == 17 for point in first[:3])
        assert all(point.certified_decimal_digits == 12 for point in first[:3])
        assert all(point.stress_metric is None for point in first[:3])
        assert first[-1].arithmetic_precision_bits >= 256
        assert first[-1].round_trip_decimal_digits >= 80
        assert first[-1].certified_decimal_digits >= 80
        assert first[-1].stress_metric is not None
        assert first[-1].stress_metric.value <= Decimal("0.000001")

    for point in first:
        values = (
            point.sqrt_s,
            *point.masses,
            *(component for momentum in point.momenta for component in momentum),
        )
        assert all(
            len(value.as_tuple().digits) <= point.round_trip_decimal_digits
            for value in values
        )


def test_three_body_stress_point_is_deliberately_soft() -> None:
    point = capture.build_reference_points("sm_ddbar_zgg", "d d~ > z g g")[-1]
    incoming = point.momenta[0]
    soft = point.momenta[-1]
    incoming_dot_soft = incoming[0] * soft[0] - sum(
        (left * right for left, right in zip(incoming[1:], soft[1:], strict=True)),
        Decimal(0),
    )

    assert soft[0] == Decimal("0.00001")
    assert soft[1] != 0
    assert soft[3] < soft[0]
    assert incoming_dot_soft > 0
    assert point.stress_metric == capture.StressMetric(
        "minimum-final-energy-fraction",
        Decimal("0.00000001"),
    )


def test_stable_leg_ids_disambiguate_only_identical_role_particles() -> None:
    particles = (
        {"state": "incoming", "name": "d"},
        {"state": "incoming", "name": "d~"},
        {"state": "outgoing", "name": "z"},
        {"state": "outgoing", "name": "g"},
        {"state": "outgoing", "name": "g"},
    )
    assert capture.stable_external_leg_ids(particles) == (
        "leg:incoming-d",
        "leg:incoming-dbar",
        "leg:outgoing-z",
        "leg:outgoing-g-1",
        "leg:outgoing-g-2",
    )
    assert capture.stable_external_leg_ids(
        (
            {"role": "initial", "particle": "scalar_0"},
            {"role": "initial", "particle": "scalar_0"},
            {"role": "final", "particle": "scalar_0"},
            {"role": "final", "particle": "scalar_0"},
        )
    ) == (
        "leg:incoming-scalar_0-1",
        "leg:incoming-scalar_0-2",
        "leg:outgoing-scalar_0-1",
        "leg:outgoing-scalar_0-2",
    )


def test_external_model_metadata_is_derived_from_compiled_ir() -> None:
    compiled = {
        "ir": {
            "particles": [
                {
                    "pdg_code": 1,
                    "propagating": True,
                    "spin": 2,
                    "color": 3,
                    "mass": "ZERO",
                },
                {
                    "pdg_code": -1,
                    "propagating": True,
                    "spin": 2,
                    "color": -3,
                    "mass": "ZERO",
                },
                {
                    "pdg_code": 23,
                    "propagating": True,
                    "spin": 3,
                    "color": 1,
                    "mass": "MZ",
                },
            ]
        },
        "parameter_defaults": {"MZ": ["91.188", "0"]},
    }

    spins, colors, masses = artifacts._external_model_metadata(
        "process", (1, -1, 23), compiled
    )

    assert spins == (2, 2, 3)
    assert colors == (3, -3, 1)
    assert masses == (Decimal(0), Decimal(0), Decimal("91.188"))
    assert tuple(
        artifacts._physical_helicity_domain(spin, mass)
        for spin, mass in zip(spins, masses, strict=True)
    ) == ((-1, 1), (-1, 1), (-1, 0, 1))


def test_clean_tree_rejects_untracked_but_ignores_ignored_files(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _run_git(repository, "init")
    _run_git(repository, "config", "user.email", "fixture@example.invalid")
    _run_git(repository, "config", "user.name", "Fixture Test")
    tracked = repository / "tracked.txt"
    (repository / ".gitignore").write_text("*.ignored\n", encoding="ascii")
    tracked.write_text("first\n", encoding="ascii")
    _run_git(repository, "add", ".gitignore", "tracked.txt")
    _run_git(repository, "commit", "-m", "initial")

    revision = capture.require_clean_tracked_tree(repository)
    first_hash = capture.tracked_source_tree_sha256(repository, revision)
    untracked = repository / "untracked.txt"
    untracked.write_text("not ignored\n", encoding="ascii")

    with pytest.raises(capture.CaptureError, match="clean source tree"):
        capture.require_clean_tracked_tree(repository)
    untracked.rename(repository / "untracked.ignored")
    assert capture.require_clean_tracked_tree(repository) == revision
    assert capture.tracked_source_tree_sha256(repository, revision) == first_hash

    tracked.write_text("changed\n", encoding="ascii")
    with pytest.raises(capture.CaptureError, match="clean source tree"):
        capture.require_clean_tracked_tree(repository)
    _run_git(repository, "add", "tracked.txt")
    _run_git(repository, "commit", "-m", "change")
    second_revision = capture.require_clean_tracked_tree(repository)
    assert second_revision != revision
    assert capture.tracked_source_tree_sha256(repository, second_revision) != first_hash


def test_runtime_snapshot_requires_exact_clean_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "1" * 40
    fingerprint = "2" * 12
    version = f"0.1.0.dev0+candidate.{fingerprint}"
    site = tmp_path / "site-packages"
    package = site / "pyamplicol"
    package.mkdir(parents=True)
    module_path = package / "__init__.py"
    module_path.write_text("# installed candidate\n", encoding="ascii")
    build_info = package / "_build_info.json"
    build_info.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "candidate_fingerprint": fingerprint,
                "source_revision": revision,
                "version": version,
            }
        ),
        encoding="ascii",
    )
    dist_info = site / f"pyamplicol-{version}.dist-info"
    dist_info.mkdir()
    metadata_path = dist_info / "METADATA"
    metadata_path.write_text(
        f"Name: pyamplicol\nVersion: {version}\n", encoding="ascii"
    )

    distribution = SimpleNamespace(
        files=(
            Path("pyamplicol/__init__.py"),
            Path("pyamplicol/_build_info.json"),
            Path(f"pyamplicol-{version}.dist-info/METADATA"),
        ),
        version=version,
        locate_file=lambda path: site / str(path),
    )

    installed = SimpleNamespace(__file__=str(module_path), __version__=version)
    monkeypatch.setattr(
        provenance.importlib,
        "import_module",
        lambda name: installed if name == "pyamplicol" else None,
    )
    monkeypatch.setattr(provenance.metadata, "distribution", lambda name: distribution)
    source = capture.SourceSnapshot(
        "https://example.invalid/pyamplicol.git", revision, "3" * 64
    )

    snapshot = provenance.collect_runtime_snapshot(source)

    assert snapshot.version == version
    assert snapshot.candidate_fingerprint == fingerprint
    assert snapshot.source_revision == revision
    assert len(snapshot.distribution_sha256) == 64
    assert len(snapshot.build_info_sha256) == 64


def test_runtime_snapshot_rejects_candidate_from_another_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fingerprint = "2" * 12
    version = f"0.1.0.dev0+candidate.{fingerprint}"
    package = tmp_path / "site-packages" / "pyamplicol"
    package.mkdir(parents=True)
    module_path = package / "__init__.py"
    module_path.write_text("# candidate\n", encoding="ascii")
    (package / "_build_info.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "candidate_fingerprint": fingerprint,
                "source_revision": "4" * 40,
                "version": version,
            }
        ),
        encoding="ascii",
    )
    installed = SimpleNamespace(__file__=str(module_path), __version__=version)
    monkeypatch.setattr(provenance.importlib, "import_module", lambda name: installed)
    monkeypatch.setattr(
        provenance.metadata,
        "distribution",
        lambda name: SimpleNamespace(version=version, files=()),
    )

    with pytest.raises(capture.CaptureError, match="not built from the clean"):
        provenance.collect_runtime_snapshot(
            capture.SourceSnapshot(
                "https://example.invalid/pyamplicol.git", "1" * 40, "3" * 64
            )
        )


def test_github_ssh_origin_is_normalized_to_https() -> None:
    assert capture.github_https_uri("git@github.com:owner/repository.git") == (
        "https://github.com/owner/repository.git"
    )
    assert (
        capture.github_https_uri("ssh://git@github.com/owner/repository.git")
        == "https://github.com/owner/repository.git"
    )


def test_atomic_write_documents_never_overwrites_or_partially_adds(
    tmp_path: Path,
) -> None:
    output = tmp_path / "capture"
    output.mkdir()
    existing = output / capture.PHYSICS_FILENAME
    existing.write_text('{"retained":true}\n', encoding="ascii")

    with pytest.raises(capture.CaptureError, match="will not be overwritten"):
        capture.atomic_write_documents(
            output,
            {
                capture.PHYSICS_FILENAME: {"new": True},
                capture.FORTRAN_EVIDENCE_FILENAME: {"new": True},
            },
        )

    assert existing.read_text(encoding="ascii") == '{"retained":true}\n'
    assert not (output / capture.FORTRAN_EVIDENCE_FILENAME).exists()
    assert not tuple(output.glob("*.tmp"))


def test_atomic_write_documents_publishes_canonical_json(tmp_path: Path) -> None:
    output = tmp_path / "capture"
    paths = capture.atomic_write_documents(
        output,
        {"one.json": {"z": 1, "a": 2}, "two.json": {"ok": True}},
    )

    assert paths == (output / "one.json", output / "two.json")
    assert json.loads(paths[0].read_text(encoding="ascii")) == {"a": 2, "z": 1}
    with pytest.raises(capture.CaptureError, match="will not be overwritten"):
        capture.atomic_write_documents(output, {"one.json": {"z": 3}})


def test_validation_failure_prevents_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = False

    def fail_validation(*args: object) -> None:
        del args
        raise capture.CaptureError("invalid fixture")

    def publish(*args: object) -> tuple[Path, ...]:
        nonlocal published
        del args
        published = True
        return ()

    monkeypatch.setattr(evidence, "validate_capture_documents", fail_validation)
    monkeypatch.setattr(evidence, "atomic_write_documents", publish)

    with pytest.raises(capture.CaptureError, match="invalid fixture"):
        evidence.validate_and_publish(tmp_path, {}, {}, {})
    assert not published


def test_run_capture_uses_mocked_assembly_and_validation_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    source = capture.SourceSnapshot("https://example.invalid/repo", "1" * 40, "2" * 64)
    runtime = capture.RuntimeSnapshot(
        "0.1.0.dev0+candidate." + "3" * 12,
        "3" * 12,
        "1" * 40,
        "4" * 64,
        "5" * 64,
    )
    dependencies = capture.DependencySnapshot((), {}, {})
    process = capture.ProcessCaptureSpec("process", "a b > c", "model")
    spec = capture.ArtifactCaptureSpec(
        "artifact",
        "artifact",
        "model",
        None,
        "lc",
        (process,),
    )
    artifact_path = tmp_path / "artifacts" / "artifact"
    fixture = {"fixture": True}
    analytic = {"analytic": True}
    fortran = {"fortran": True}
    written = (
        tmp_path / capture.PHYSICS_FILENAME,
        tmp_path / capture.FORTRAN_EVIDENCE_FILENAME,
        tmp_path / capture.ANALYTIC_EVIDENCE_FILENAME,
    )

    monkeypatch.setattr(pipeline, "collect_source_snapshot", lambda root: source)
    monkeypatch.setattr(pipeline, "collect_runtime_snapshot", lambda value: runtime)
    monkeypatch.setattr(
        pipeline, "collect_dependency_snapshot", lambda value: dependencies
    )
    monkeypatch.setattr(pipeline, "artifact_capture_specs", lambda root: (spec,))

    def materialize(*args: object) -> dict[str, Path]:
        del args
        calls.append("materialize")
        return {spec.id: artifact_path}

    def assemble(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        calls.append("assemble")
        return fixture

    def analytic_evidence(value: object) -> dict[str, object]:
        assert value is fixture
        calls.append("analytic")
        return analytic

    def fortran_evidence(*args: object) -> dict[str, object]:
        del args
        calls.append("fortran")
        return fortran

    def unchanged(value: object, root: object) -> None:
        assert value is source
        del root
        calls.append("unchanged")

    def publish(*args: object) -> tuple[Path, ...]:
        del args
        calls.append("validate-publish")
        return written

    monkeypatch.setattr(pipeline, "materialize_artifacts", materialize)
    monkeypatch.setattr(pipeline, "assemble_fixture", assemble)
    monkeypatch.setattr(pipeline, "build_analytic_evidence", analytic_evidence)
    monkeypatch.setattr(pipeline, "build_fortran_evidence", fortran_evidence)
    monkeypatch.setattr(pipeline, "assert_source_snapshot_unchanged", unchanged)
    monkeypatch.setattr(
        pipeline, "assert_runtime_snapshot_unchanged", lambda *args: None
    )
    monkeypatch.setattr(pipeline, "validate_and_publish", publish)
    config = capture.CaptureConfig(
        output_directory=tmp_path,
        artifact_root=tmp_path / "artifacts",
        legacy_repository=tmp_path / "legacy",
        legacy_jobs=1,
        artifact_mode="reuse",
        capture_command=("capture",),
    )

    result = pipeline.run_capture(config)

    assert calls == [
        "materialize",
        "assemble",
        "analytic",
        "fortran",
        "unchanged",
        "validate-publish",
    ]
    assert result.fixture_path == written[0]
    assert result.evidence_paths == written[1:]
    assert result.artifact_paths == (artifact_path,)
