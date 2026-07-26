# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools.developer import compiled_mode_matrix as matrix
from tools.developer import x86_performance_runtime_bundle as bundle


def test_embedded_frozen_install_state_has_accepted_digest() -> None:
    encoded = bundle.frozen_install_state_bytes()
    assert hashlib.sha256(encoded).hexdigest() == (
        bundle.FROZEN_INSTALL_STATE_SHA256
    )
    payload = json.loads(encoded)
    assert payload["created_utc"] == "2026-07-24T16:52:37.275744+00:00"
    assert payload["candidate_lock_sha256"] == (
        bundle.FROZEN_CANDIDATE_LOCK_SHA256
    )
    assert payload["cargo_config_sha256"] == bundle.FROZEN_CARGO_CONFIG_SHA256
    assert payload["sources"]["legacy-amplicol"]["revision"] == (
        "79c96cecf2a722e50c3d2030b6894d755f96518a"
    )


def test_freeze_baseline_requires_exact_path_revision_and_generated_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "baseline"
    dependencies = source / "dependencies"
    dependencies.mkdir(parents=True)
    lock = dependencies / "candidate-Cargo.lock"
    config = dependencies / "candidate-cargo-config.toml"
    lock.write_bytes(b"lock")
    config.write_bytes(b"config")
    for relative in bundle._BASELINE_BOOTSTRAP_ROOTS:
        (source / relative).mkdir(parents=True)
    monkeypatch.setattr(bundle, "FROZEN_SOURCE_ROOT", source.resolve())
    monkeypatch.setattr(
        bundle,
        "FROZEN_CANDIDATE_LOCK_SHA256",
        hashlib.sha256(b"lock").hexdigest(),
    )
    monkeypatch.setattr(
        bundle,
        "FROZEN_CARGO_CONFIG_SHA256",
        hashlib.sha256(b"config").hexdigest(),
    )
    frozen = dict(bundle.FROZEN_INSTALL_STATE)
    frozen["candidate_lock_sha256"] = bundle.FROZEN_CANDIDATE_LOCK_SHA256
    frozen["cargo_config_sha256"] = bundle.FROZEN_CARGO_CONFIG_SHA256
    monkeypatch.setattr(bundle, "FROZEN_INSTALL_STATE", frozen)
    encoded = bundle.frozen_install_state_bytes()
    (dependencies / "install-state.json").write_bytes(encoded)
    monkeypatch.setattr(
        bundle,
        "FROZEN_INSTALL_STATE_SHA256",
        hashlib.sha256(encoded).hexdigest(),
    )
    def git_output(_root: Path, *arguments: str) -> str:
        if arguments[:2] == ("rev-parse", "--verify"):
            return matrix.FROZEN_BASELINE_SOURCE_REVISION + "\n"
        if arguments[:1] == ("status",):
            return ""
        if arguments[:1] == ("ls-files",):
            return "\0".join(
                (
                    *bundle._BASELINE_GENERATED_FILES,
                    ".venv/bin/python",
                    "dependencies/checkouts/symbolica/file",
                    "dependencies/wheelhouse/dependency.whl",
                    "",
                )
            )
        raise AssertionError(arguments)

    monkeypatch.setattr(bundle, "_git_output", git_output)
    result = bundle.freeze_baseline(source)
    assert result["passes"] is True
    bundle._require_content_identity(result)
    assert (dependencies / "install-state.json").read_bytes() == encoded

    config.write_bytes(b"changed")
    with pytest.raises(bundle.BundleError, match="dependency inputs differ"):
        bundle.freeze_baseline(source)


def _installation(source: str, inputs: str, marker: str) -> dict[str, object]:
    return {
        "package_version": "0.1.0.dev0+candidate.test",
        "build_info": {
            "source_revision": source,
            "native_build_inputs_sha256": inputs,
            "publishable": False,
        },
        "build_info_sha256": marker * 64,
        "distribution_content": {
            "algorithm": "sha256-relative-path-size-content-v1",
            "sha256": marker * 64,
            "file_count": 10,
            "size_bytes": 100,
        },
        "native_module": {
            "relative_path": "pyamplicol/_rusticol.abi3.so",
            "sha256": marker * 64,
            "size_bytes": 50,
        },
    }


def _manifest_arguments(tmp_path: Path) -> argparse.Namespace:
    root = tmp_path / "bundle"
    for lane in ("baseline", "current"):
        directory = root / "wheels" / lane
        directory.mkdir(parents=True)
        (directory / f"{lane}.whl").write_bytes(lane.encode())
    prepared = root / "prepared-models" / "ufo-sm-jit-o2.pyamplicol-model"
    prepared.parent.mkdir(parents=True)
    prepared.write_bytes(b"prepared")
    attestation = bundle._attach_content_identity(
        {
            "kind": bundle.BASELINE_ATTESTATION_KIND,
            "schema_version": bundle.SCHEMA_VERSION,
            "source_root": str(bundle.FROZEN_SOURCE_ROOT),
            "source_revision": matrix.FROZEN_BASELINE_SOURCE_REVISION,
            "native_build_inputs_sha256": (
                matrix.FROZEN_BASELINE_NATIVE_INPUTS_SHA256
            ),
            "candidate_lock_sha256": bundle.FROZEN_CANDIDATE_LOCK_SHA256,
            "cargo_config_sha256": bundle.FROZEN_CARGO_CONFIG_SHA256,
            "install_state_sha256": bundle.FROZEN_INSTALL_STATE_SHA256,
            "source_inventory": {
                "tracked_and_untracked_status_clean": True,
                "unexpected_ignored_files": [],
            },
            "generated_identity": {"test": True},
            "passes": True,
        }
    )
    (root / "frozen-baseline-attestation.json").write_text(
        json.dumps(attestation, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return argparse.Namespace(
        bundle_root=root,
        baseline_python=Path("/runtime/baseline/python"),
        current_python=Path("/runtime/current/python"),
        dependency_site=Path("/runtime/dependencies"),
        workflow_run_id="4242",
        expected_current_revision="a" * 40,
        expected_current_native_inputs_sha256="b" * 64,
        output=tmp_path / "manifest.json",
    )


def test_manifest_binds_binaries_dependencies_and_prepared_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _manifest_arguments(tmp_path)
    installations = {
        str(arguments.baseline_python): _installation(
            matrix.FROZEN_BASELINE_SOURCE_REVISION,
            matrix.FROZEN_BASELINE_NATIVE_INPUTS_SHA256,
            "c",
        ),
        str(arguments.current_python): _installation(
            arguments.expected_current_revision,
            arguments.expected_current_native_inputs_sha256,
            "d",
        ),
    }
    monkeypatch.setattr(
        bundle,
        "_installation",
        lambda path: installations[str(path)],
    )
    dependency = {
        "algorithm": "deps-v1",
        "sha256": "e" * 64,
        "distributions": {},
    }
    monkeypatch.setattr(bundle, "_stable_dependency_site", lambda _path: dependency)
    monkeypatch.setattr(bundle.platform, "system", lambda: "Linux")
    monkeypatch.setattr(bundle.platform, "machine", lambda: "x86_64")
    payload = bundle.create_manifest(arguments)
    assert payload["passes"] is True
    assert payload["installations"]["baseline"] == installations[
        str(arguments.baseline_python)
    ]
    assert payload["dependency_site"] == dependency
    bundle._require_content_identity(payload)

    verify = argparse.Namespace(
        **{
            key: value
            for key, value in vars(arguments).items()
            if key
            not in {
                "expected_current_native_inputs_sha256",
                "output",
            }
        },
        manifest=arguments.output,
    )
    result = bundle.verify_manifest(verify)
    assert result["passes"] is True
    assert result["manifest_content_sha256"] == payload["content_identity"]["sha256"]

    wheel = arguments.bundle_root / "wheels" / "current" / "current.whl"
    wheel.write_bytes(b"tampered")
    with pytest.raises(bundle.BundleError, match="files differ"):
        bundle.verify_manifest(verify)


def test_manifest_rejects_wrong_frozen_or_current_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _manifest_arguments(tmp_path)
    monkeypatch.setattr(bundle.platform, "system", lambda: "Linux")
    monkeypatch.setattr(bundle.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        bundle,
        "_installation",
        lambda _path: _installation("f" * 40, "0" * 64, "1"),
    )
    with pytest.raises(bundle.BundleError, match="baseline installed build"):
        bundle.create_manifest(arguments)


def test_content_identity_rejects_manifest_tampering() -> None:
    payload = bundle._attach_content_identity({"passes": True, "value": 1})
    bundle._require_content_identity(payload)
    payload["value"] = 2
    with pytest.raises(bundle.BundleError, match="content identity"):
        bundle._require_content_identity(payload)


def test_file_identity_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.whl"
    target.write_bytes(b"wheel")
    link = tmp_path / "link.whl"
    link.symlink_to(target)
    with pytest.raises(bundle.BundleError, match="non-symlink"):
        bundle._file_identity(link)


def test_checked_manifest_rejects_duplicate_and_nonfinite_json(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"kind":"a","kind":"b"}', encoding="utf-8")
    with pytest.raises(bundle.BundleError, match="duplicate JSON key"):
        bundle._checked_manifest(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(bundle.BundleError, match="non-finite"):
        bundle._checked_manifest(nonfinite)


def test_wheel_inventory_rejects_untracked_extra_file(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    for lane in ("baseline", "current"):
        directory = root / "wheels" / lane
        directory.mkdir(parents=True)
        (directory / f"{lane}.whl").write_bytes(lane.encode())
    (root / "wheels" / "current" / "unexpected.txt").write_text(
        "unexpected\n",
        encoding="utf-8",
    )
    with pytest.raises(bundle.BundleError, match="exactly one file"):
        bundle._wheel_inventory(root)


def test_freeze_baseline_rejects_dirty_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "baseline"
    source.mkdir()
    monkeypatch.setattr(bundle, "FROZEN_SOURCE_ROOT", source.resolve())

    def git_output(_root: Path, *arguments: str) -> str:
        if arguments[:2] == ("rev-parse", "--verify"):
            return matrix.FROZEN_BASELINE_SOURCE_REVISION + "\n"
        if arguments[:1] == ("status",):
            return " M src/pyamplicol/runtime.py\n"
        return ""

    monkeypatch.setattr(bundle, "_git_output", git_output)
    with pytest.raises(bundle.BundleError, match="source changes"):
        bundle.freeze_baseline(source)


def test_materialize_runtime_reconstructs_both_lanes_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "bundle"
    dependencies = bundle_root / "dependency-site"
    dependencies.mkdir(parents=True)
    (dependencies / "dependency.py").write_text("VALUE = 1\n", encoding="utf-8")
    for lane in ("baseline", "current"):
        wheel = bundle_root / "wheels" / lane / f"{lane}.whl"
        wheel.parent.mkdir(parents=True)
        wheel.write_bytes(lane.encode())
    runtime_root = tmp_path / "runtime"

    class FakeBuilder:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def create(self, path: Path) -> None:
            (path / "bin").mkdir(parents=True)
            (path / "bin" / "python").write_bytes(b"python")

    def fake_purelib(python: Path) -> Path:
        site = python.parents[1] / "site-packages"
        site.mkdir()
        return site

    monkeypatch.setattr(bundle.venv, "EnvBuilder", FakeBuilder)
    monkeypatch.setattr(bundle, "_venv_purelib", fake_purelib)
    monkeypatch.setattr(
        bundle,
        "_stable_dependency_site",
        lambda _path: {"sha256": "a" * 64},
    )
    monkeypatch.setattr(
        bundle,
        "_installation",
        lambda python: {"python": str(python), "passes": True},
    )
    monkeypatch.setattr(
        bundle.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        ),
    )
    result = bundle.materialize_runtime(bundle_root, runtime_root)
    assert result["passes"] is True
    assert set(result["lanes"]) == {"baseline", "current"}
    for lane in ("baseline", "current"):
        site = runtime_root / lane / "site-packages"
        assert (site / "dependency.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_materialize_runtime_rejects_nonempty_destination(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "bundle"
    dependencies = bundle_root / "dependency-site"
    dependencies.mkdir(parents=True)
    for lane in ("baseline", "current"):
        wheel = bundle_root / "wheels" / lane / f"{lane}.whl"
        wheel.parent.mkdir(parents=True)
        wheel.write_bytes(lane.encode())
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "stale").write_text("stale\n", encoding="utf-8")
    with pytest.raises(bundle.BundleError, match="must be empty"):
        bundle.materialize_runtime(bundle_root, runtime_root)
