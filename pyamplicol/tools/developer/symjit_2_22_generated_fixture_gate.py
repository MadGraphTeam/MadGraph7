#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Run the fail-closed SymJIT 2.22 generated-artifact fixture gates."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build_backend"))

import _pyamplicol_build as build_backend  # noqa: E402

_DEFAULT_EXECUTION_ROOT = (
    Path(".artifacts")
    / "symjit-2.22-migration"
    / "generated-fixture-gate"
)
_CANDIDATE_MANIFEST_PLACEHOLDER = "<authenticated-candidate-overlay>/Cargo.toml"
_EXPECTED_PROCESS_EXPRESSION = "d d~ > z g g g g g g"
_EXPECTED_PROCESS_ID = "d_dbar_to_z_g_g_g_g_g_g"
_EXPECTED_EXTERNAL_PDGS = (1, -1, 23, 21, 21, 21, 21, 21, 21)
_SOURCE_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_NATIVE_BUILD_INPUTS_PATTERN = re.compile(r"[0-9a-f]{64}")
_CANDIDATE_VERSION_PATTERN = re.compile(
    r"[0-9]+\.[0-9]+\.[0-9]+\.dev0\+candidate\.[0-9a-f]{12}"
)


class GateError(RuntimeError):
    """Raised when a fixture or focused gate fails closed."""


@dataclass(frozen=True)
class FixturePaths:
    compiled_o3: Path
    eager_o2: Path
    recurrence_topology_o2: Path
    recurrence_union_o2: Path
    contracted_nlc: Path

    def as_dict(self) -> dict[str, Path]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class ExecutionPaths:
    cargo_home: Path
    cargo_target_dir: Path
    tmpdir: Path
    pip_cache_dir: Path
    xdg_cache_home: Path
    python_cache_prefix: Path

    def as_dict(self) -> dict[str, Path]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def environment(self) -> dict[str, str]:
        return {
            "CARGO_HOME": str(self.cargo_home),
            "CARGO_TARGET_DIR": str(self.cargo_target_dir),
            "TMPDIR": str(self.tmpdir),
            "PIP_CACHE_DIR": str(self.pip_cache_dir),
            "XDG_CACHE_HOME": str(self.xdg_cache_home),
            "PYTHONPYCACHEPREFIX": str(self.python_cache_prefix),
            "PYAMPLICOL_REQUIRE_NATIVE_TESTS": "1",
            "CARGO_TERM_COLOR": "never",
        }

    def create(self) -> None:
        for path in self.as_dict().values():
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class GateCommand:
    label: str
    arguments: tuple[str, ...]
    expected_tests: tuple[str, ...]


_ARTIFACT_ENVIRONMENT_ROUTES = (
    ("RUSTICOL_COMPILED_DIRECT_ARTIFACT", "compiled_o3"),
    ("RUSTICOL_COMPILED_ARTIFACT", "compiled_o3"),
    ("RUSTICOL_TEST_COMPILED_REPLAY_QUICK_ARTIFACT", "compiled_o3"),
    ("RUSTICOL_SELECTOR_ARTIFACT", "compiled_o3"),
    ("RUSTICOL_EAGER_ARTIFACT", "eager_o2"),
    ("RUSTICOL_RECURRENCE_ARTIFACT", "recurrence_topology_o2"),
    (
        "RUSTICOL_RECURRENCE_TOPOLOGY_ALLOCATION_ARTIFACT",
        "recurrence_topology_o2",
    ),
    ("RUSTICOL_RECURRENCE_UNION_ALLOCATION_ARTIFACT", "recurrence_union_o2"),
    (
        "RUSTICOL_RECURRENCE_CONTRACTED_ALLOCATION_ARTIFACT",
        "contracted_nlc",
    ),
    ("RUSTICOL_CONTRACTED_SELECTOR_ARTIFACT", "contracted_nlc"),
)

_COMPILED_FULL_TEST = (
    "engine::compiled_direct_prototype::tests::"
    "retained_compiled_o3_artifact_native_runtime_direct_matches_legacy_contract"
)
_COMPILED_QUICK_TEST = (
    "engine::compiled_direct_prototype::tests::"
    "retained_compiled_replay_totals_quick_contract"
)
_EAGER_ALLOCATION_TEST = (
    "engine::eager_integration_tests::"
    "generated_eager_native_into_is_warmed_allocation_free_when_fixture_is_supplied"
)
_RECURRENCE_SMOKE_TEST = (
    "engine::recurrence_integration_tests::"
    "generated_recurrence_artifact_loads_when_fixture_is_supplied"
)
_RECURRENCE_BOUNDARY_TEST = (
    "engine::recurrence_integration_tests::"
    "generated_recurrence_odd_tails_report_zero_direct_boundary_traffic_when_fixtures_are_supplied"
)
_RECURRENCE_ALLOCATION_TESTS = (
    "genuine_topology_replay_artifact_warmed_loop_allocates_zero_heap_bytes",
    "genuine_all_flow_union_artifact_warmed_loop_allocates_zero_heap_bytes",
    "genuine_contracted_color_artifact_warmed_loop_allocates_zero_heap_bytes",
    "genuine_contracted_color_prepared_selector_allocates_zero_heap_bytes",
)
_ODD_TAIL_TESTS = (
    "genuine_compiled_o3_odd_tails_preserve_numerics_allocations_and_boundary_traffic",
    "genuine_eager_o2_odd_tails_preserve_numerics_allocations_and_boundary_traffic",
    "genuine_topology_recurrence_o2_odd_tails_preserve_numerics_and_allocations",
    "genuine_all_flow_union_recurrence_o2_odd_tails_preserve_numerics_and_allocations",
)
_CAPI_ARTIFACT_TESTS = (
    "generated_compiled_artifact_reports_compiled_execution_mode",
    "generated_eager_artifact_loads_and_evaluates_through_the_c_abi",
)
_CAPI_SELECTOR_TESTS = (
    "direct_total_outputs_preserve_bits_canaries_and_capacity_errors",
    "homogeneous_and_alternating_per_point_selectors_match_resolved_components",
    "malformed_selector_buffers_fail_without_touching_output",
    "contracted_color_rejects_global_and_per_point_flow_selectors",
)


def _workspace_artifact(
    raw_path: str | Path,
    *,
    label: str,
    workspace_root: Path,
    source_revision: str,
    native_build_inputs_sha256: str,
) -> Path:
    root = workspace_root.resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise GateError(f"{label} artifact does not exist: {candidate}") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise GateError(
            f"{label} artifact must be workspace-local: {resolved}"
        ) from error
    if not resolved.is_dir():
        raise GateError(f"{label} artifact is not a directory: {resolved}")

    manifest = resolved / "artifact.json"
    if not manifest.is_file():
        raise GateError(f"{label} artifact is missing artifact.json: {resolved}")
    try:
        manifest = manifest.resolve(strict=True)
    except OSError as error:
        raise GateError(
            f"{label} artifact.json could not be resolved: {manifest}"
        ) from error
    try:
        manifest.relative_to(root)
    except ValueError as error:
        raise GateError(
            f"{label} artifact.json must be workspace-local: {manifest}"
        ) from error
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(
            f"{label} artifact.json is not valid JSON: {manifest}"
        ) from error
    if not isinstance(payload, dict):
        raise GateError(f"{label} artifact.json must contain a JSON object: {manifest}")
    _validate_fixture_identity(
        payload,
        label=label,
        manifest=manifest,
        source_revision=source_revision,
        native_build_inputs_sha256=native_build_inputs_sha256,
    )
    return resolved


def _validated_identity_digest(
    value: str,
    *,
    label: str,
    length: int,
    pattern: re.Pattern[str],
) -> str:
    if pattern.fullmatch(value) is None:
        raise GateError(f"{label} must be a lowercase {length}-digit hex value")
    return value


def _validate_fixture_identity(
    payload: Mapping[str, object],
    *,
    label: str,
    manifest: Path,
    source_revision: str,
    native_build_inputs_sha256: str,
) -> None:
    producer = payload.get("producer")
    if not isinstance(producer, dict):
        raise GateError(f"{label} producer must be a JSON object: {manifest}")
    expected_producer = {
        "distribution": "pyamplicol",
        "git_revision": source_revision,
        "native_build_inputs_sha256": native_build_inputs_sha256,
    }
    for key, expected in expected_producer.items():
        observed = producer.get(key)
        if observed != expected:
            raise GateError(
                f"{label} producer {key} does not match the authenticated "
                f"candidate (expected {expected!r}, got {observed!r}): {manifest}"
            )
    version = producer.get("version")
    if (
        not isinstance(version, str)
        or _CANDIDATE_VERSION_PATTERN.fullmatch(version) is None
    ):
        raise GateError(
            f"{label} producer version is not a candidate build "
            f"(got {version!r}): {manifest}"
        )

    processes = payload.get("processes")
    if (
        not isinstance(processes, list)
        or len(processes) != 1
        or not isinstance(processes[0], dict)
    ):
        raise GateError(
            f"{label} must contain exactly one object-valued process: {manifest}"
        )
    process = processes[0]
    expected_process = {
        "id": _EXPECTED_PROCESS_ID,
        "expression": _EXPECTED_PROCESS_EXPRESSION,
        "external_pdgs": list(_EXPECTED_EXTERNAL_PDGS),
    }
    for key, expected in expected_process.items():
        observed = process.get(key)
        if observed != expected:
            raise GateError(
                f"{label} process {key} does not identify the exact Z+6g "
                f"fixture (expected {expected!r}, got {observed!r}): {manifest}"
            )
    default_process_id = payload.get("default_process_id")
    if default_process_id != _EXPECTED_PROCESS_ID:
        raise GateError(
            f"{label} default_process_id does not identify the exact Z+6g "
            f"fixture (expected {_EXPECTED_PROCESS_ID!r}, "
            f"got {default_process_id!r}): {manifest}"
        )


def preflight_fixtures(
    raw_paths: Mapping[str, str | Path],
    *,
    workspace_root: Path,
    source_revision: str,
    native_build_inputs_sha256: str,
) -> FixturePaths:
    source_revision = _validated_identity_digest(
        source_revision,
        label="source revision",
        length=40,
        pattern=_SOURCE_REVISION_PATTERN,
    )
    native_build_inputs_sha256 = _validated_identity_digest(
        native_build_inputs_sha256,
        label="native build inputs SHA-256",
        length=64,
        pattern=_NATIVE_BUILD_INPUTS_PATTERN,
    )
    expected_names = {field.name for field in fields(FixturePaths)}
    if set(raw_paths) != expected_names:
        missing = sorted(expected_names.difference(raw_paths))
        unexpected = sorted(set(raw_paths).difference(expected_names))
        raise GateError(
            "fixture arguments do not match the required roles "
            f"(missing={missing}, unexpected={unexpected})"
        )
    resolved = {
        name: _workspace_artifact(
            raw_paths[name],
            label=name.replace("_", " "),
            workspace_root=workspace_root,
            source_revision=source_revision,
            native_build_inputs_sha256=native_build_inputs_sha256,
        )
        for name in sorted(expected_names)
    }
    if len(set(resolved.values())) != len(resolved):
        raise GateError(
            "the five fixture roles must name distinct artifact directories"
        )
    fixtures = FixturePaths(**resolved)
    _common_fixture_candidate_version(fixtures)
    return fixtures


def _workspace_execution_path(
    raw_path: str | Path,
    *,
    label: str,
    workspace_root: Path,
) -> Path:
    root = workspace_root.resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as error:
        raise GateError(f"could not resolve {label}: {candidate}") from error
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise GateError(f"{label} must be workspace-local: {resolved}") from error
    if not relative.parts:
        raise GateError(f"{label} must be below the workspace root: {resolved}")
    if resolved.exists() and not resolved.is_dir():
        raise GateError(f"{label} exists but is not a directory: {resolved}")
    return resolved


def preflight_execution_paths(
    raw_paths: Mapping[str, str | Path],
    *,
    workspace_root: Path,
) -> ExecutionPaths:
    expected_names = {field.name for field in fields(ExecutionPaths)}
    if set(raw_paths) != expected_names:
        missing = sorted(expected_names.difference(raw_paths))
        unexpected = sorted(set(raw_paths).difference(expected_names))
        raise GateError(
            "execution path arguments do not match the required roles "
            f"(missing={missing}, unexpected={unexpected})"
        )
    resolved = {
        name: _workspace_execution_path(
            raw_paths[name],
            label=name.replace("_", " "),
            workspace_root=workspace_root,
        )
        for name in sorted(expected_names)
    }
    if len(set(resolved.values())) != len(resolved):
        raise GateError("cache and temporary directories must be distinct")
    return ExecutionPaths(**resolved)


def artifact_environment(fixtures: FixturePaths) -> dict[str, str]:
    paths = fixtures.as_dict()
    return {
        environment_name: str(paths[fixture_name])
        for environment_name, fixture_name in _ARTIFACT_ENVIRONMENT_ROUTES
    }


def _common_fixture_candidate_version(fixtures: FixturePaths) -> str:
    versions: dict[str, str] = {}
    for label, artifact in fixtures.as_dict().items():
        manifest = artifact / "artifact.json"
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            producer = payload["producer"]
            version = producer["version"]
        except (
            KeyError,
            OSError,
            TypeError,
            UnicodeError,
            json.JSONDecodeError,
        ) as error:
            raise GateError(
                f"{label.replace('_', ' ')} candidate version could not be "
                f"re-authenticated: {manifest}"
            ) from error
        if (
            not isinstance(version, str)
            or _CANDIDATE_VERSION_PATTERN.fullmatch(version) is None
        ):
            raise GateError(
                f"{label.replace('_', ' ')} producer version is not a "
                f"candidate build (got {version!r}): {manifest}"
            )
        versions[label] = version
    unique = set(versions.values())
    if len(unique) != 1:
        raise GateError(
            "the five authenticated fixtures were not generated by one "
            f"candidate build: {versions}"
        )
    return unique.pop()


def _authenticate_candidate_overlay_build_info(
    overlay: Path,
    *,
    source_revision: str,
    native_build_inputs_sha256: str,
    candidate_version: str,
) -> None:
    build_info_path = overlay / "src" / "pyamplicol" / "_build_info.json"
    if build_info_path.is_symlink():
        raise GateError(
            "authenticated candidate overlay build info must not be a symlink: "
            f"{build_info_path}"
        )
    try:
        resolved_build_info = build_info_path.resolve(strict=True)
        resolved_build_info.relative_to(overlay)
    except (OSError, ValueError) as error:
        raise GateError(
            "authenticated candidate overlay build info is missing or escaped "
            f"the overlay: {build_info_path}"
        ) from error
    if not resolved_build_info.is_file():
        raise GateError(
            "authenticated candidate overlay build info is not a regular file: "
            f"{resolved_build_info}"
        )
    try:
        payload = json.loads(resolved_build_info.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(
            "authenticated candidate overlay build info is not valid JSON: "
            f"{resolved_build_info}"
        ) from error
    if not isinstance(payload, dict):
        raise GateError(
            "authenticated candidate overlay build info must be a JSON object: "
            f"{resolved_build_info}"
        )

    schema_version = payload.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise GateError(
            "authenticated candidate overlay build identity does not match "
            "the generated fixtures: schema_version must be integer 1"
        )
    if payload.get("publishable") is not False:
        raise GateError(
            "authenticated candidate overlay build identity does not match "
            "the generated fixtures: publishable must be false"
        )
    if payload.get("selftest_fixture_bootstrap") is not False:
        raise GateError(
            "authenticated candidate overlay build identity does not match "
            "the generated fixtures: selftest_fixture_bootstrap must be false"
        )
    expected = {
        "source_revision": source_revision,
        "native_build_inputs_sha256": native_build_inputs_sha256,
        "version": candidate_version,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise GateError(
                "authenticated candidate overlay build identity does not match "
                f"the generated fixtures: {key} expected {value!r}, got "
                f"{payload.get(key)!r}"
            )
    fingerprint = payload.get("candidate_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{12}", fingerprint) is None
        or not candidate_version.endswith(f"+candidate.{fingerprint}")
    ):
        raise GateError(
            "authenticated candidate overlay fingerprint does not match its "
            f"candidate version: {resolved_build_info}"
        )
    if "release_prepared_model_bootstrap" in payload:
        raise GateError(
            "authenticated candidate overlay unexpectedly enables release "
            f"prepared-model bootstrap: {resolved_build_info}"
        )


@contextmanager
def authenticated_candidate_overlay(
    execution_paths: ExecutionPaths,
    *,
    source_revision: str,
    native_build_inputs_sha256: str,
    candidate_version: str,
) -> Iterator[Path]:
    """Stage one authenticated candidate graph shared by every Cargo command."""

    source_revision = _validated_identity_digest(
        source_revision,
        label="source revision",
        length=40,
        pattern=_SOURCE_REVISION_PATTERN,
    )
    native_build_inputs_sha256 = _validated_identity_digest(
        native_build_inputs_sha256,
        label="native build inputs SHA-256",
        length=64,
        pattern=_NATIVE_BUILD_INPUTS_PATTERN,
    )
    if _CANDIDATE_VERSION_PATTERN.fullmatch(candidate_version) is None:
        raise GateError(
            "candidate version must identify an authenticated candidate build"
        )
    try:
        execution_paths.create()
        build_backend._check_dependencies("candidate")
        with build_backend._overlay(
            "candidate",
            temporary_directory=execution_paths.tmpdir,
            cargo_target_directory=execution_paths.cargo_target_dir,
        ) as (overlay, cargo_target):
            resolved_overlay = overlay.resolve(strict=True)
            resolved_tmpdir = execution_paths.tmpdir.resolve(strict=True)
            resolved_target = cargo_target.resolve(strict=False)
            try:
                resolved_overlay.relative_to(resolved_tmpdir)
            except ValueError as error:
                raise GateError(
                    "authenticated candidate overlay escaped the workspace-local "
                    f"temporary root: {resolved_overlay}"
                ) from error
            if resolved_target != execution_paths.cargo_target_dir:
                raise GateError(
                    "authenticated candidate overlay selected the wrong Cargo "
                    f"target directory: {resolved_target}"
                )
            for required in (
                resolved_overlay / "Cargo.lock",
                resolved_overlay / ".cargo" / "config.toml",
            ):
                if not required.is_file():
                    raise GateError(
                        "authenticated candidate overlay is incomplete: "
                        f"{required}"
                    )
            _authenticate_candidate_overlay_build_info(
                resolved_overlay,
                source_revision=source_revision,
                native_build_inputs_sha256=native_build_inputs_sha256,
                candidate_version=candidate_version,
            )
            yield resolved_overlay
    except GateError:
        raise
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        raise GateError(
            f"could not stage the authenticated candidate Cargo overlay: {error}"
        ) from error


def _core_lib_command(
    cargo: str,
    manifest: Path,
    label: str,
    test_name: str,
) -> GateCommand:
    return GateCommand(
        label=label,
        arguments=(
            cargo,
            "test",
            "--locked",
            "--manifest-path",
            str(manifest),
            "-p",
            "rusticol-core",
            "--lib",
            test_name,
            "--",
            "--exact",
            "--show-output",
            "--test-threads=1",
        ),
        expected_tests=(test_name,),
    )


def gate_commands(
    *,
    cargo: str,
    workspace_root: Path,
) -> tuple[GateCommand, ...]:
    manifest = workspace_root.resolve() / "Cargo.toml"
    if not manifest.is_file():
        raise GateError(f"workspace Cargo.toml is missing: {manifest}")

    commands = [
        _core_lib_command(
            cargo,
            manifest,
            "compiled O3 full allocation and selector contract",
            _COMPILED_FULL_TEST,
        ),
        _core_lib_command(
            cargo,
            manifest,
            "compiled O3 quick replay allocation contract",
            _COMPILED_QUICK_TEST,
        ),
        _core_lib_command(
            cargo,
            manifest,
            "eager O2 allocation and selector contract",
            _EAGER_ALLOCATION_TEST,
        ),
        GateCommand(
            label="recurrence topology/union O2 generated-artifact contracts",
            arguments=(
                cargo,
                "test",
                "--locked",
                "--manifest-path",
                str(manifest),
                "-p",
                "rusticol-core",
                "--lib",
                "engine::recurrence_integration_tests::",
                "--",
                "--show-output",
                "--test-threads=1",
            ),
            expected_tests=(
                _RECURRENCE_SMOKE_TEST,
                _RECURRENCE_BOUNDARY_TEST,
            ),
        ),
        GateCommand(
            label="recurrence generated-artifact allocation contracts",
            arguments=(
                cargo,
                "test",
                "--locked",
                "--manifest-path",
                str(manifest),
                "-p",
                "rusticol-core",
                "--test",
                "recurrence_direct_arena_allocations",
                "--",
                "--show-output",
                "--test-threads=1",
            ),
            expected_tests=_RECURRENCE_ALLOCATION_TESTS,
        ),
        GateCommand(
            label="compiled, eager, and recurrence genuine-artifact odd-tail contracts",
            arguments=(
                cargo,
                "test",
                "--locked",
                "--manifest-path",
                str(manifest),
                "-p",
                "rusticol-core",
                "--test",
                "generated_artifact_odd_tails",
                "--",
                "--show-output",
                "--test-threads=1",
            ),
            expected_tests=_ODD_TAIL_TESTS,
        ),
        GateCommand(
            label="compiled and eager C ABI generated-artifact contracts",
            arguments=(
                cargo,
                "test",
                "--locked",
                "--manifest-path",
                str(manifest),
                "-p",
                "rusticol-capi",
                "--test",
                "eager_artifact",
                "--",
                "--show-output",
                "--test-threads=1",
            ),
            expected_tests=_CAPI_ARTIFACT_TESTS,
        ),
        GateCommand(
            label="compiled and contracted C ABI selector contracts",
            arguments=(
                cargo,
                "test",
                "--locked",
                "--manifest-path",
                str(manifest),
                "-p",
                "rusticol-capi",
                "--test",
                "runtime_selectors",
                "--",
                "--show-output",
                "--test-threads=1",
            ),
            expected_tests=_CAPI_SELECTOR_TESTS,
        ),
    ]
    return tuple(commands)


def verify_test_markers(command: GateCommand, output: str) -> None:
    missing = [
        test_name
        for test_name in command.expected_tests
        if re.search(
            rf"(?m)^test {re.escape(test_name)} \.\.\. ok(?:[ \t].*)?\r?$",
            output,
        )
        is None
    ]
    if missing:
        raise GateError(
            f"{command.label} did not run every required test; "
            f"missing successful markers: {missing}"
        )


def run_commands(
    commands: Sequence[GateCommand],
    *,
    environment: Mapping[str, str],
    execution_paths: ExecutionPaths,
    workspace_root: Path,
) -> None:
    required_environment = {
        environment_name
        for environment_name, _fixture_name in _ARTIFACT_ENVIRONMENT_ROUTES
    }
    if set(environment) != required_environment:
        missing = sorted(required_environment.difference(environment))
        unexpected = sorted(set(environment).difference(required_environment))
        raise GateError(
            "generated-fixture environment is incomplete "
            f"(missing={missing}, unexpected={unexpected})"
        )
    try:
        execution_paths.create()
    except OSError as error:
        raise GateError(
            f"could not create workspace-local execution directories: {error}"
        ) from error
    updates = {
        **environment,
        **execution_paths.environment(),
        **build_backend._macos_native_build_updates(),
    }
    inherited = {
        name: value
        for name, value in build_backend._clean_environment(updates).items()
        if not (name.startswith("RUSTICOL_") and name.endswith("_ARTIFACT"))
    }
    inherited.update(environment)

    print("[generated-fixture-gate] artifact mappings", flush=True)
    for name in sorted(environment):
        print(f"{name}={shlex.quote(environment[name])}", flush=True)
    print("[generated-fixture-gate] execution environment", flush=True)
    for name, value in sorted(execution_paths.environment().items()):
        print(f"{name}={shlex.quote(value)}", flush=True)

    for command in commands:
        print(f"[generated-fixture-gate] {command.label}", flush=True)
        print(f"$ {shlex.join(command.arguments)}", flush=True)
        try:
            completed = subprocess.run(
                command.arguments,
                cwd=workspace_root,
                env=inherited,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except OSError as error:
            raise GateError(f"could not launch {command.label}: {error}") from error
        sys.stdout.write(completed.stdout)
        sys.stdout.flush()
        if completed.returncode != 0:
            raise GateError(
                f"{command.label} failed with exit status {completed.returncode}"
            )
        verify_test_markers(command, completed.stdout)


def plan_payload(
    fixtures: FixturePaths,
    commands: Sequence[GateCommand],
    *,
    execution_paths: ExecutionPaths,
    workspace_root: Path,
    source_revision: str,
    native_build_inputs_sha256: str,
) -> dict[str, object]:
    environment = artifact_environment(fixtures)
    environment.update(execution_paths.environment())
    source_manifest = str(workspace_root.resolve() / "Cargo.toml")
    return {
        "schema_version": 1,
        "workspace_root": str(workspace_root.resolve()),
        "cargo_input": {
            "mode": "authenticated-candidate-overlay",
            "manifest": _CANDIDATE_MANIFEST_PLACEHOLDER,
            "shared_target": str(execution_paths.cargo_target_dir),
        },
        "fixture_identity": {
            "producer": {
                "distribution": "pyamplicol",
                "git_revision": source_revision,
                "native_build_inputs_sha256": native_build_inputs_sha256,
                "version_pattern": _CANDIDATE_VERSION_PATTERN.pattern,
            },
            "process": {
                "id": _EXPECTED_PROCESS_ID,
                "expression": _EXPECTED_PROCESS_EXPRESSION,
                "external_pdgs": list(_EXPECTED_EXTERNAL_PDGS),
            },
        },
        "artifacts": {
            name: str(path) for name, path in fixtures.as_dict().items()
        },
        "execution_paths": {
            name: str(path)
            for name, path in execution_paths.as_dict().items()
        },
        "environment": environment,
        "commands": [
            {
                "label": command.label,
                "arguments": [
                    (
                        _CANDIDATE_MANIFEST_PLACEHOLDER
                        if argument == source_manifest
                        else argument
                    )
                    for argument in command.arguments
                ],
                "expected_tests": list(command.expected_tests),
            }
            for command in commands
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--compiled-o3-artifact", required=True)
    parser.add_argument("--eager-o2-artifact", required=True)
    parser.add_argument("--recurrence-topology-o2-artifact", required=True)
    parser.add_argument("--recurrence-union-o2-artifact", required=True)
    parser.add_argument("--contracted-nlc-artifact", required=True)
    parser.add_argument(
        "--source-revision",
        required=True,
        help="authenticated candidate source revision (40 lowercase hex digits)",
    )
    parser.add_argument(
        "--native-build-inputs-sha256",
        required=True,
        help="authenticated candidate native-build-input digest",
    )
    parser.add_argument(
        "--cargo-home",
        default=str(_DEFAULT_EXECUTION_ROOT / "cargo-home"),
        help="workspace-local Cargo home",
    )
    parser.add_argument(
        "--cargo-target-dir",
        default=str(_DEFAULT_EXECUTION_ROOT / "cargo-target"),
        help="workspace-local Cargo target directory",
    )
    parser.add_argument(
        "--tmpdir",
        default=str(_DEFAULT_EXECUTION_ROOT / "tmp"),
        help="workspace-local temporary directory",
    )
    parser.add_argument(
        "--pip-cache-dir",
        default=str(_DEFAULT_EXECUTION_ROOT / "pip-cache"),
        help="workspace-local pip cache",
    )
    parser.add_argument(
        "--xdg-cache-home",
        default=str(_DEFAULT_EXECUTION_ROOT / "xdg-cache"),
        help="workspace-local XDG cache",
    )
    parser.add_argument(
        "--python-cache-prefix",
        default=str(_DEFAULT_EXECUTION_ROOT / "python-cache"),
        help="workspace-local Python bytecode cache",
    )
    parser.add_argument(
        "--cargo",
        default="cargo",
        help="Cargo executable to invoke from PATH or by explicit path",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="preflight fixtures and print the exact JSON plan without running Cargo",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    raw_paths = {
        "compiled_o3": arguments.compiled_o3_artifact,
        "eager_o2": arguments.eager_o2_artifact,
        "recurrence_topology_o2": arguments.recurrence_topology_o2_artifact,
        "recurrence_union_o2": arguments.recurrence_union_o2_artifact,
        "contracted_nlc": arguments.contracted_nlc_artifact,
    }
    raw_execution_paths = {
        "cargo_home": arguments.cargo_home,
        "cargo_target_dir": arguments.cargo_target_dir,
        "tmpdir": arguments.tmpdir,
        "pip_cache_dir": arguments.pip_cache_dir,
        "xdg_cache_home": arguments.xdg_cache_home,
        "python_cache_prefix": arguments.python_cache_prefix,
    }
    try:
        fixtures = preflight_fixtures(
            raw_paths,
            workspace_root=ROOT,
            source_revision=arguments.source_revision,
            native_build_inputs_sha256=arguments.native_build_inputs_sha256,
        )
        execution_paths = preflight_execution_paths(
            raw_execution_paths,
            workspace_root=ROOT,
        )
        candidate_version = _common_fixture_candidate_version(fixtures)
        if arguments.plan:
            commands = gate_commands(cargo=arguments.cargo, workspace_root=ROOT)
            print(
                json.dumps(
                    plan_payload(
                        fixtures,
                        commands,
                        execution_paths=execution_paths,
                        workspace_root=ROOT,
                        source_revision=arguments.source_revision,
                        native_build_inputs_sha256=(
                            arguments.native_build_inputs_sha256
                        ),
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        with authenticated_candidate_overlay(
            execution_paths,
            source_revision=arguments.source_revision,
            native_build_inputs_sha256=arguments.native_build_inputs_sha256,
            candidate_version=candidate_version,
        ) as overlay:
            commands = gate_commands(
                cargo=arguments.cargo,
                workspace_root=overlay,
            )
            run_commands(
                commands,
                environment=artifact_environment(fixtures),
                execution_paths=execution_paths,
                workspace_root=overlay,
            )
    except GateError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
