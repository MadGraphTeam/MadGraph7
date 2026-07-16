# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "release"))

import check_legal_inventory  # noqa: E402
import check_rust_licenses  # noqa: E402

CANDIDATE_LOCK = ROOT / "dependencies" / "candidate-Cargo.lock"


def _read_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _candidate_environment_available() -> bool:
    if not CANDIDATE_LOCK.is_file():
        return False
    config = _read_toml(ROOT / "config" / "release-dependencies.toml")
    return all(
        (ROOT / package["manifest"]).is_file()
        for package in config.get("candidate_path_package", [])
    )


@pytest.mark.skipif(
    not _candidate_environment_available(),
    reason="candidate legal projection requires the ignored managed candidate inputs",
)
def test_candidate_path_inventory_is_derived_from_the_active_lock() -> None:
    lock = _read_toml(CANDIDATE_LOCK)
    inventory = _read_toml(ROOT / "licenses" / "RUST_THIRD_PARTY.toml")
    source_less = {
        (package["name"], package["version"])
        for package in lock["package"]
        if "source" not in package
    }
    first_party = {
        (package["name"], package["version"]) for package in inventory["first_party"]
    }
    assert check_legal_inventory.candidate_package_keys(ROOT) == (
        source_less - first_party
    )


@pytest.mark.skipif(
    not _candidate_environment_available(),
    reason="candidate legal projection requires the ignored managed candidate inputs",
)
def test_candidate_projection_runs_all_legal_checks() -> None:
    issues = check_legal_inventory.check_repository(ROOT, mode="candidate")
    assert issues == []
    assert check_rust_licenses.blocking_issues(issues, mode="candidate") == []
    assert check_rust_licenses.blocking_issues(issues, mode="release") == []


def test_release_mode_delegates_to_strict_gate_without_candidate_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strict_issue = check_rust_licenses.LicenseIssue("strict-sentinel", "strict gate")
    strict_calls: list[Path] = []
    read_paths: list[Path] = []
    original_read_toml = check_legal_inventory._read_toml

    def strict_gate(root: Path) -> list[check_rust_licenses.LicenseIssue]:
        strict_calls.append(root)
        return [strict_issue]

    def guarded_read_toml(path: Path) -> dict:
        resolved = path.resolve()
        read_paths.append(resolved)
        if resolved == CANDIDATE_LOCK.resolve():
            raise AssertionError("release mode read ignored candidate state")
        return original_read_toml(path)

    monkeypatch.setattr(check_rust_licenses, "check_repository", strict_gate)
    monkeypatch.setattr(check_legal_inventory, "_read_toml", guarded_read_toml)

    issues = check_legal_inventory.check_repository(ROOT, mode="release")

    assert issues == [strict_issue]
    assert strict_calls == [ROOT.resolve()]
    assert CANDIDATE_LOCK.resolve() not in read_paths


def test_locked_cargo_fetch_commands_cover_release_targets_deterministically() -> None:
    commands = check_rust_licenses.cargo_fetch_commands(ROOT)

    assert tuple(command[-1] for command in commands) == (
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "x86_64-unknown-linux-gnu",
    )
    expected_manifest = str((ROOT / "Cargo.toml").resolve())
    for command in commands:
        assert command == (
            "cargo",
            "fetch",
            "--locked",
            "--manifest-path",
            expected_manifest,
            "--target",
            command[-1],
        )


def test_locked_cargo_fetch_executes_each_command_once() -> None:
    calls: list[tuple[tuple[str, ...], Path]] = []

    commands = check_rust_licenses.fetch_locked_release_targets(
        ROOT,
        runner=lambda command, root: calls.append((command, root)),
    )

    assert calls == [(command, ROOT.resolve()) for command in commands]
