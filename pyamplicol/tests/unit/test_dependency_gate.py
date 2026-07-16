# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "release" / "check_dependencies.py"


def _module():
    spec = importlib.util.spec_from_file_location("dependency_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_gate_fails_closed_before_contributor_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "STATE_PATH", tmp_path / "install-state.json")
    monkeypatch.setattr(module, "CANDIDATE_LOCK_PATH", tmp_path / "Cargo.lock")
    monkeypatch.setattr(module, "CARGO_CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(module, "CHECKOUTS_PATH", tmp_path / "checkouts")
    codes = {issue.code for issue in module.check(candidate=True, online=False)}
    assert "candidate-input-missing" in codes


def test_release_gate_stays_closed_until_upstreams_are_verified() -> None:
    module = _module()
    codes = {issue.code for issue in module.check(candidate=False, online=False)}
    assert codes == {
        "python-artifact-missing",
        "symbolica-unverified",
        "ufo-loader-unverified",
    }


def test_release_cargo_lock_contains_only_published_registry_crates() -> None:
    module = _module()
    lock = module._load_lock()
    assert module._release_cargo_lock_issues(lock) == []


def test_release_toolchain_and_manylinux_image_are_exactly_pinned() -> None:
    module = _module()
    lock = module._load_lock()
    assert module._toolchain_issues(lock) == []


def test_python_runtime_lock_is_complete_and_covers_release_matrix() -> None:
    module = _module()
    release = module._load_lock()
    runtime, issues = module._python_runtime_lock(release)
    assert issues == []
    assert runtime is not None
    assert runtime.supported_python == ("3.11", "3.12", "3.13", "3.14")
    assert set(runtime.by_name) == {
        "colorama",
        "numpy",
        "platformdirs",
        "prettytable",
        "progressbar2",
        "python-utils",
        "six",
        "symbolica",
        "tomli-w",
        "typing-extensions",
        "ufo-model-loader",
        "wcwidth",
    }
    assert module._direct_python_contract_issues(release, runtime) == []
    assert (
        module._python_artifact_issues(
            release,
            runtime,
            candidate=True,
            online=False,
        )
        == []
    )


def test_python_runtime_lock_digest_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    release = module._load_lock()
    changed = tmp_path / "python-runtime-lock.toml"
    changed.write_bytes(module.PYTHON_LOCK_PATH.read_bytes() + b"\n# changed\n")
    monkeypatch.setattr(module, "PYTHON_LOCK_PATH", changed)
    runtime, issues = module._python_runtime_lock(release)
    assert runtime is None
    assert [issue.code for issue in issues] == ["python-lock-digest"]


def test_release_cargo_lock_rejects_candidate_path_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    lock = module._load_lock()
    contaminated = tmp_path / "Cargo.lock"
    text = module.CARGO_LOCK_PATH.read_text(encoding="utf-8")
    marker = 'name = "symbolica"\nversion = "2.1.0"\n'
    assert marker in text
    text = text.replace(
        marker
        + 'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        + f'checksum = "{lock["symbolica"]["rust_checksum"]}"\n',
        marker,
        1,
    )
    contaminated.write_text(text, encoding="utf-8")
    monkeypatch.setattr(module, "CARGO_LOCK_PATH", contaminated)
    codes = {issue.code for issue in module._release_cargo_lock_issues(lock)}
    assert codes == {"release-cargo-nonregistry", "release-cargo-pin"}


def test_cargo_metadata_validation_uses_clean_explicit_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    observed: dict[str, object] = {}

    def fake_run(command, *, cwd, env, text, capture_output, check):
        path = Path(cwd)
        observed.update(command=command, cwd=path, env=env)
        assert (path / "Cargo.lock").read_bytes() == (
            module.CARGO_LOCK_PATH.read_bytes()
        )
        assert not (path / ".cargo" / "config.toml").exists()
        assert "CARGO_HOME" not in env
        assert env["CARGO_TARGET_DIR"] == str(path / "target")
        return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/cargo")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert (
        module._cargo_metadata_issue(
            mode="release",
            lock_path=module.CARGO_LOCK_PATH,
            config_path=None,
        )
        is None
    )
    assert observed["command"][-3:] == ["--locked", "--format-version", "1"]


def test_candidate_cargo_metadata_projects_symjit_requirement(
    tmp_path: Path,
) -> None:
    module = _module()
    lock = module._load_lock()
    published = lock["symbolica"]["published_symjit_version"]
    candidate = lock["symjit"]["candidate_version"]
    manifest = tmp_path / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        f'[dependencies]\nsymjit = {{ version = "={published}", optional = true }}\n',
        encoding="utf-8",
    )

    module._rewrite_candidate_symjit_requirement(tmp_path)

    assert f'symjit = {{ version = "={candidate}"' in manifest.read_text(
        encoding="utf-8"
    )


def test_candidate_gate_accepts_only_exact_installer_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    lock = module._load_lock()
    state_path = tmp_path / "install-state.json"
    candidate_lock = tmp_path / "candidate-Cargo.lock"
    cargo_config = tmp_path / "config.toml"
    checkouts = tmp_path / "checkouts"
    candidate_lock.write_text("candidate lock\n", encoding="utf-8")
    cargo_config.write_text("[patch.crates-io]\n", encoding="utf-8")
    snapshot = module._source_tree_sha256(checkouts / "missing")
    revisions = module._candidate_revisions(lock)
    sources = {}
    for name, revision in revisions.items():
        (checkouts / name).mkdir(parents=True)
        sources[name] = {
            "revision": revision,
            "worktree_sha256": snapshot,
        }
    state = {
        "schema_version": 1,
        "publishable": False,
        "release_lock_sha256": hashlib.sha256(
            module.LOCK_PATH.read_bytes()
        ).hexdigest(),
        "python_runtime_lock_sha256": hashlib.sha256(
            module.PYTHON_LOCK_PATH.read_bytes()
        ).hexdigest(),
        "candidate_lock_sha256": hashlib.sha256(
            candidate_lock.read_bytes()
        ).hexdigest(),
        "cargo_config_sha256": hashlib.sha256(cargo_config.read_bytes()).hexdigest(),
        "sources": sources,
        "patches": [
            {
                "dependency": entry["dependency"],
                "path": entry["path"],
                "sha256": entry["sha256"],
            }
            for entry in lock["patches"]
        ],
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(module, "STATE_PATH", state_path)
    monkeypatch.setattr(module, "CANDIDATE_LOCK_PATH", candidate_lock)
    monkeypatch.setattr(module, "CARGO_CONFIG_PATH", cargo_config)
    monkeypatch.setattr(module, "CHECKOUTS_PATH", checkouts)
    monkeypatch.setattr(module, "_candidate_cargo_lock_issues", lambda: [])
    monkeypatch.setattr(module, "_cargo_metadata_issue", lambda **_kwargs: None)

    def git_output(path: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return revisions[path.name] + "\n"
        return ""

    monkeypatch.setattr(module, "_git_output", git_output)
    assert module._candidate_issues(lock) == []


def test_candidate_tree_fingerprint_covers_untracked_and_ignored_bytes(
    tmp_path: Path,
) -> None:
    module = _module()
    source = tmp_path / "source"
    source.mkdir()
    payload = source / "generated.rs"
    payload.write_text("one\n", encoding="utf-8")
    first = module._source_tree_sha256(source)
    payload.write_text("two\n", encoding="utf-8")
    second = module._source_tree_sha256(source)
    assert second != first
    target = source / "target"
    target.mkdir()
    (target / "build-output").write_text("ignored\n", encoding="utf-8")
    assert module._source_tree_sha256(source) == second
