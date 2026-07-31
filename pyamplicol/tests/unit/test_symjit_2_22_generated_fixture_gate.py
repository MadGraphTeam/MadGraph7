# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from tools.developer import symjit_2_22_generated_fixture_gate as gate

_FIXTURE_NAMES = (
    "compiled_o3",
    "eager_o2",
    "recurrence_topology_o2",
    "recurrence_union_o2",
    "contracted_nlc",
)
_SOURCE_REVISION = "a" * 40
_NATIVE_BUILD_INPUTS_SHA256 = "b" * 64
_CANDIDATE_VERSION = "0.1.0.dev0+candidate.0123456789ab"


def _manifest(name: str) -> dict[str, Any]:
    return {
        "artifact_id": name,
        "default_process_id": gate._EXPECTED_PROCESS_ID,
        "processes": [
            {
                "id": gate._EXPECTED_PROCESS_ID,
                "expression": gate._EXPECTED_PROCESS_EXPRESSION,
                "external_pdgs": list(gate._EXPECTED_EXTERNAL_PDGS),
            }
        ],
        "producer": {
            "distribution": "pyamplicol",
            "git_revision": _SOURCE_REVISION,
            "native_build_inputs_sha256": _NATIVE_BUILD_INPUTS_SHA256,
            "version": _CANDIDATE_VERSION,
        },
        "extensions": {
            "artifact_identity": dict(gate._ARTIFACT_IDENTITY_CONTRACT),
        },
    }


def _build_info(
    *,
    source_revision: str = _SOURCE_REVISION,
    native_build_inputs_sha256: str = _NATIVE_BUILD_INPUTS_SHA256,
    version: str = _CANDIDATE_VERSION,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "publishable": False,
        "candidate_fingerprint": version.rsplit("+candidate.", maxsplit=1)[1],
        "native_build_inputs_sha256": native_build_inputs_sha256,
        "selftest_fixture_bootstrap": False,
        "source_checkout": "/workspace/source",
        "source_revision": source_revision,
        "version": version,
    }


def _preflight(
    raw: dict[str, Path],
    *,
    workspace_root: Path,
) -> gate.FixturePaths:
    return gate.preflight_fixtures(
        raw,
        workspace_root=workspace_root,
        source_revision=_SOURCE_REVISION,
        native_build_inputs_sha256=_NATIVE_BUILD_INPUTS_SHA256,
    )


def _identity_arguments() -> list[str]:
    return [
        "--source-revision",
        _SOURCE_REVISION,
        "--native-build-inputs-sha256",
        _NATIVE_BUILD_INPUTS_SHA256,
    ]


def _workspace(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
    fixtures: dict[str, Path] = {}
    for name in _FIXTURE_NAMES:
        artifact = root / "fixtures" / name
        artifact.mkdir(parents=True)
        (artifact / "artifact.json").write_text(
            json.dumps(_manifest(name)),
            encoding="utf-8",
        )
        fixtures[name] = artifact
    return root, fixtures


def _execution_paths(root: Path) -> dict[str, Path]:
    state = root / ".artifacts" / "gate-state"
    return {
        "cargo_home": state / "cargo-home",
        "cargo_target_dir": state / "cargo-target",
        "tmpdir": state / "tmp",
        "pip_cache_dir": state / "pip-cache",
        "xdg_cache_home": state / "xdg-cache",
        "python_cache_prefix": state / "python-cache",
    }


def test_preflight_requires_distinct_workspace_local_artifact_manifests(
    tmp_path: Path,
) -> None:
    root, fixtures = _workspace(tmp_path)
    resolved = _preflight(fixtures, workspace_root=root)
    assert resolved.as_dict() == fixtures

    missing_manifest = root / "fixtures" / "missing-manifest"
    missing_manifest.mkdir()
    invalid = dict(fixtures)
    invalid["compiled_o3"] = missing_manifest
    with pytest.raises(gate.GateError, match=r"missing artifact\.json"):
        _preflight(invalid, workspace_root=root)

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "artifact.json").write_text("{}", encoding="utf-8")
    invalid = dict(fixtures)
    invalid["compiled_o3"] = outside
    with pytest.raises(gate.GateError, match="workspace-local"):
        _preflight(invalid, workspace_root=root)

    duplicate = dict(fixtures)
    duplicate["compiled_o3"] = duplicate["eager_o2"]
    with pytest.raises(gate.GateError, match="distinct artifact directories"):
        _preflight(duplicate, workspace_root=root)


@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
def test_preflight_authenticates_the_producer_of_every_fixture_role(
    fixture_name: str,
    tmp_path: Path,
) -> None:
    root, fixtures = _workspace(tmp_path)
    manifest = fixtures[fixture_name] / "artifact.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["producer"]["git_revision"] = "c" * 40
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(gate.GateError, match="producer git_revision"):
        _preflight(fixtures, workspace_root=root)


def test_preflight_requires_exact_candidate_and_z6g_identity(
    tmp_path: Path,
) -> None:
    root, fixtures = _workspace(tmp_path)
    manifest = fixtures["compiled_o3"] / "artifact.json"

    invalid = _manifest("compiled_o3")
    invalid["producer"]["version"] = "0.1.0"
    manifest.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(gate.GateError, match="not a candidate build"):
        _preflight(fixtures, workspace_root=root)

    invalid = _manifest("compiled_o3")
    invalid["producer"]["native_build_inputs_sha256"] = "c" * 64
    manifest.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(
        gate.GateError,
        match="producer native_build_inputs_sha256",
    ):
        _preflight(fixtures, workspace_root=root)

    invalid = _manifest("compiled_o3")
    invalid["processes"][0]["expression"] = "u u~ > z g g g g g g"
    manifest.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(gate.GateError, match="exact Z\\+6g fixture"):
        _preflight(fixtures, workspace_root=root)

    invalid = _manifest("compiled_o3")
    invalid["processes"][0]["external_pdgs"] = [
        2,
        -2,
        23,
        21,
        21,
        21,
        21,
        21,
        21,
    ]
    manifest.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(gate.GateError, match="exact Z\\+6g fixture"):
        _preflight(fixtures, workspace_root=root)

    invalid = _manifest("compiled_o3")
    invalid["default_process_id"] = "not-the-z6g-process"
    manifest.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(gate.GateError, match="default_process_id"):
        _preflight(fixtures, workspace_root=root)


def test_preflight_rejects_unauthenticated_identity_arguments(
    tmp_path: Path,
) -> None:
    root, fixtures = _workspace(tmp_path)

    with pytest.raises(gate.GateError, match="lowercase 40-digit hex"):
        gate.preflight_fixtures(
            fixtures,
            workspace_root=root,
            source_revision="A" * 40,
            native_build_inputs_sha256=_NATIVE_BUILD_INPUTS_SHA256,
        )
    with pytest.raises(gate.GateError, match="lowercase 64-digit hex"):
        gate.preflight_fixtures(
            fixtures,
            workspace_root=root,
            source_revision=_SOURCE_REVISION,
            native_build_inputs_sha256="b" * 63,
        )


def test_execution_paths_are_workspace_local_and_preflight_does_not_create_them(
    tmp_path: Path,
) -> None:
    root, _ = _workspace(tmp_path)
    raw = _execution_paths(root)

    paths = gate.preflight_execution_paths(raw, workspace_root=root)

    assert paths.as_dict() == raw
    assert all(not path.exists() for path in raw.values())

    invalid = dict(raw)
    invalid["tmpdir"] = tmp_path / "outside-tmp"
    with pytest.raises(gate.GateError, match="workspace-local"):
        gate.preflight_execution_paths(invalid, workspace_root=root)


def test_environment_uses_every_audited_fixture_mapping(tmp_path: Path) -> None:
    root, raw = _workspace(tmp_path)
    fixtures = _preflight(raw, workspace_root=root)

    assert gate.artifact_environment(fixtures) == {
        "RUSTICOL_COMPILED_DIRECT_ARTIFACT": str(raw["compiled_o3"]),
        "RUSTICOL_COMPILED_ARTIFACT": str(raw["compiled_o3"]),
        "RUSTICOL_TEST_COMPILED_REPLAY_QUICK_ARTIFACT": str(raw["compiled_o3"]),
        "RUSTICOL_SELECTOR_ARTIFACT": str(raw["compiled_o3"]),
        "RUSTICOL_EAGER_ARTIFACT": str(raw["eager_o2"]),
        "RUSTICOL_RECURRENCE_ARTIFACT": str(raw["recurrence_topology_o2"]),
        "RUSTICOL_RECURRENCE_TOPOLOGY_ALLOCATION_ARTIFACT": str(
            raw["recurrence_topology_o2"]
        ),
        "RUSTICOL_RECURRENCE_UNION_ALLOCATION_ARTIFACT": str(
            raw["recurrence_union_o2"]
        ),
        "RUSTICOL_RECURRENCE_CONTRACTED_ALLOCATION_ARTIFACT": str(
            raw["contracted_nlc"]
        ),
        "RUSTICOL_CONTRACTED_SELECTOR_ARTIFACT": str(raw["contracted_nlc"]),
    }


def test_execution_creates_local_state_and_exports_the_required_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, raw_fixtures = _workspace(tmp_path)
    fixtures = _preflight(raw_fixtures, workspace_root=root)
    raw_execution = _execution_paths(root)
    execution = gate.preflight_execution_paths(
        raw_execution,
        workspace_root=root,
    )
    command = gate.GateCommand(
        label="mock focused gate",
        arguments=("cargo", "test"),
        expected_tests=("required_test",),
    )
    calls: list[dict[str, object]] = []

    class Completed:
        returncode = 0
        stdout = "test required_test ... ok\n"

    def run(*args: object, **kwargs: object) -> Completed:
        del args
        calls.append(kwargs)
        return Completed()

    monkeypatch.setattr(gate.subprocess, "run", run)

    gate.run_commands(
        (command,),
        environment=gate.artifact_environment(fixtures),
        execution_paths=execution,
        workspace_root=root,
    )

    assert all(path.is_dir() for path in raw_execution.values())
    child_environment = calls[0]["env"]
    assert isinstance(child_environment, dict)
    assert child_environment["PYAMPLICOL_REQUIRE_NATIVE_TESTS"] == "1"
    assert child_environment["CARGO_HOME"] == str(raw_execution["cargo_home"])
    assert child_environment["CARGO_TARGET_DIR"] == str(
        raw_execution["cargo_target_dir"]
    )
    assert child_environment["TMPDIR"] == str(raw_execution["tmpdir"])


def test_authenticated_candidate_overlay_is_local_and_reuses_the_gate_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, _ = _workspace(tmp_path)
    raw_execution = _execution_paths(root)
    execution = gate.preflight_execution_paths(
        raw_execution,
        workspace_root=root,
    )
    overlay = raw_execution["tmpdir"] / "pyamplicol-build-test" / "source"
    (overlay / ".cargo").mkdir(parents=True)
    (overlay / "Cargo.lock").write_text("version = 4\n", encoding="utf-8")
    (overlay / ".cargo" / "config.toml").write_text(
        "[patch.crates-io]\n",
        encoding="utf-8",
    )
    build_info = overlay / "src" / "pyamplicol" / "_build_info.json"
    build_info.parent.mkdir(parents=True)
    build_info.write_text(
        json.dumps(_build_info()),
        encoding="utf-8",
    )
    calls: list[tuple[str, dict[str, Path]]] = []
    dependency_checks: list[str] = []

    @contextmanager
    def fake_overlay(mode: str, **kwargs: Path):
        calls.append((mode, kwargs))
        yield overlay, kwargs["cargo_target_directory"]

    monkeypatch.setattr(
        gate.build_backend,
        "_check_dependencies",
        dependency_checks.append,
    )
    monkeypatch.setattr(gate.build_backend, "_overlay", fake_overlay)

    with gate.authenticated_candidate_overlay(
        execution,
        source_revision=_SOURCE_REVISION,
        native_build_inputs_sha256=_NATIVE_BUILD_INPUTS_SHA256,
        candidate_version=_CANDIDATE_VERSION,
    ) as observed:
        assert observed == overlay

    assert calls == [
        (
            "candidate",
            {
                "temporary_directory": raw_execution["tmpdir"],
                "cargo_target_directory": raw_execution["cargo_target_dir"],
            },
        )
    ]
    assert dependency_checks == ["candidate"]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"source_revision": "c" * 40}, "source_revision"),
        ({"native_build_inputs_sha256": "d" * 64}, "native_build_inputs_sha256"),
        (
            {"version": "0.1.0.dev0+candidate.fedcba987654"},
            "version",
        ),
        ({"selftest_fixture_bootstrap": True}, "selftest_fixture_bootstrap"),
    ],
)
def test_authenticated_candidate_overlay_rejects_mismatched_build_identity(
    updates: dict[str, object],
    message: str,
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "overlay"
    build_info = overlay / "src" / "pyamplicol" / "_build_info.json"
    build_info.parent.mkdir(parents=True)
    payload = _build_info()
    payload.update(updates)
    build_info.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(gate.GateError, match=message):
        gate._authenticate_candidate_overlay_build_info(
            overlay.resolve(),
            source_revision=_SOURCE_REVISION,
            native_build_inputs_sha256=_NATIVE_BUILD_INPUTS_SHA256,
            candidate_version=_CANDIDATE_VERSION,
        )


def test_preflight_requires_one_candidate_version(tmp_path: Path) -> None:
    root, raw = _workspace(tmp_path)
    fixtures = _preflight(raw, workspace_root=root)
    assert gate._common_fixture_candidate_version(fixtures) == _CANDIDATE_VERSION

    manifest = raw["compiled_o3"] / "artifact.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["producer"]["version"] = "0.1.0.dev0+candidate.fedcba987654"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(gate.GateError, match="one candidate build"):
        _preflight(raw, workspace_root=root)


def test_main_executes_every_command_in_one_authenticated_candidate_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, fixtures = _workspace(tmp_path)
    overlay = root / ".artifacts" / "authenticated-overlay"
    overlay.mkdir(parents=True)
    (overlay / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
    monkeypatch.setattr(gate, "ROOT", root)
    calls: list[tuple[tuple[gate.GateCommand, ...], Path]] = []

    @contextmanager
    def fake_candidate_overlay(
        _execution: gate.ExecutionPaths,
        *,
        source_revision: str,
        native_build_inputs_sha256: str,
        candidate_version: str,
    ):
        assert source_revision == _SOURCE_REVISION
        assert native_build_inputs_sha256 == _NATIVE_BUILD_INPUTS_SHA256
        assert candidate_version == _CANDIDATE_VERSION
        yield overlay

    def capture_commands(
        commands: tuple[gate.GateCommand, ...],
        *,
        environment: object,
        execution_paths: object,
        workspace_root: Path,
    ) -> None:
        del environment, execution_paths
        calls.append((commands, workspace_root))

    monkeypatch.setattr(
        gate,
        "authenticated_candidate_overlay",
        fake_candidate_overlay,
    )
    monkeypatch.setattr(gate, "run_commands", capture_commands)
    arguments = [
        "--compiled-o3-artifact",
        str(fixtures["compiled_o3"]),
        "--eager-o2-artifact",
        str(fixtures["eager_o2"]),
        "--recurrence-topology-o2-artifact",
        str(fixtures["recurrence_topology_o2"]),
        "--recurrence-union-o2-artifact",
        str(fixtures["recurrence_union_o2"]),
        "--contracted-nlc-artifact",
        str(fixtures["contracted_nlc"]),
        *_identity_arguments(),
    ]

    assert gate.main(arguments) == 0
    assert len(calls) == 1
    commands, workspace_root = calls[0]
    assert workspace_root == overlay
    assert len(commands) == 8
    assert all(
        str(overlay / "Cargo.toml") in command.arguments
        for command in commands
    )
    assert all(
        str(root / "Cargo.toml") not in command.arguments for command in commands
    )


def test_plan_contains_every_focused_gate_and_expected_marker(
    tmp_path: Path,
) -> None:
    root, _ = _workspace(tmp_path)
    commands = gate.gate_commands(cargo="/toolchain/bin/cargo", workspace_root=root)

    assert len(commands) == 8
    assert all(command.arguments[0] == "/toolchain/bin/cargo" for command in commands)
    assert all("--locked" in command.arguments for command in commands)
    assert all(str(root / "Cargo.toml") in command.arguments for command in commands)
    assert all(
        command.arguments.count(str(root / "Cargo.toml")) == 1
        for command in commands
    )
    assert all("--show-output" in command.arguments for command in commands)
    assert all("--test-threads=1" in command.arguments for command in commands)
    odd_tail_command = next(
        command
        for command in commands
        if command.expected_tests == gate._ODD_TAIL_TESTS
    )
    assert "generated_artifact_odd_tails" in odd_tail_command.arguments
    assert {test for command in commands for test in command.expected_tests} == {
        gate._COMPILED_FULL_TEST,
        gate._COMPILED_QUICK_TEST,
        gate._EAGER_ALLOCATION_TEST,
        gate._RECURRENCE_SMOKE_TEST,
        gate._RECURRENCE_BOUNDARY_TEST,
        *gate._RECURRENCE_ALLOCATION_TESTS,
        *gate._ODD_TAIL_TESTS,
        *gate._CAPI_ARTIFACT_TESTS,
        *gate._CAPI_SELECTOR_TESTS,
    }


def test_success_requires_an_ok_marker_for_every_requested_test(
    tmp_path: Path,
) -> None:
    root, _ = _workspace(tmp_path)
    command = gate.gate_commands(cargo="cargo", workspace_root=root)[4]
    output = "\n".join(
        f"test {test_name} ... ok" for test_name in command.expected_tests
    )
    gate.verify_test_markers(command, output)

    with pytest.raises(gate.GateError, match="did not run every required test"):
        gate.verify_test_markers(
            command,
            f"test {command.expected_tests[0]} ... ok\n"
            "test unrelated_test ... ok\n",
        )


def test_plan_mode_preflights_without_launching_cargo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, fixtures = _workspace(tmp_path)
    monkeypatch.setattr(gate, "ROOT", root)

    def unexpected_run(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("plan mode must not launch Cargo")

    monkeypatch.setattr(gate.subprocess, "run", unexpected_run)
    arguments = [
        "--compiled-o3-artifact",
        str(fixtures["compiled_o3"]),
        "--eager-o2-artifact",
        str(fixtures["eager_o2"]),
        "--recurrence-topology-o2-artifact",
        str(fixtures["recurrence_topology_o2"]),
        "--recurrence-union-o2-artifact",
        str(fixtures["recurrence_union_o2"]),
        "--contracted-nlc-artifact",
        str(fixtures["contracted_nlc"]),
        *_identity_arguments(),
        "--plan",
    ]

    assert gate.main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workspace_root"] == str(root)
    assert payload["cargo_input"] == {
        "mode": "authenticated-candidate-overlay",
        "manifest": gate._CANDIDATE_MANIFEST_PLACEHOLDER,
        "shared_target": payload["execution_paths"]["cargo_target_dir"],
    }
    assert payload["fixture_identity"] == {
        "producer": {
            "distribution": "pyamplicol",
            "git_revision": _SOURCE_REVISION,
            "native_build_inputs_sha256": _NATIVE_BUILD_INPUTS_SHA256,
            "version_pattern": gate._CANDIDATE_VERSION_PATTERN.pattern,
        },
        "process": {
            "id": gate._EXPECTED_PROCESS_ID,
            "expression": gate._EXPECTED_PROCESS_EXPRESSION,
            "external_pdgs": list(gate._EXPECTED_EXTERNAL_PDGS),
        },
    }
    assert len(payload["commands"]) == 8
    assert len(payload["execution_paths"]) == 6
    assert len(payload["environment"]) == 18
    assert payload["environment"]["PYAMPLICOL_REQUIRE_NATIVE_TESTS"] == "1"
    assert payload["environment"]["CARGO_TERM_COLOR"] == "never"
    default_state_root = (
        root
        / ".artifacts"
        / "symjit-2.22-migration"
        / "generated-fixture-gate"
    )
    assert all(
        Path(path).is_relative_to(default_state_root)
        for path in payload["execution_paths"].values()
    )
    assert payload["environment"]["CARGO_HOME"] == payload["execution_paths"][
        "cargo_home"
    ]
    assert payload["environment"]["CARGO_TARGET_DIR"] == payload[
        "execution_paths"
    ]["cargo_target_dir"]
    assert payload["environment"]["TMPDIR"] == payload["execution_paths"]["tmpdir"]
    assert payload["environment"]["PIP_CACHE_DIR"] == payload["execution_paths"][
        "pip_cache_dir"
    ]
    assert payload["environment"]["XDG_CACHE_HOME"] == payload[
        "execution_paths"
    ]["xdg_cache_home"]
    assert all(
        gate._CANDIDATE_MANIFEST_PLACEHOLDER in command["arguments"]
        for command in payload["commands"]
    )
    assert all(
        str(root / "Cargo.toml") not in command["arguments"]
        for command in payload["commands"]
    )
    assert not (root / ".artifacts").exists()
