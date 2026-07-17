# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import importlib.util
import json
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


def test_release_gate_reports_only_the_true_upstream_blocker_offline() -> None:
    module = _module()
    codes = [issue.code for issue in module.check(candidate=False, online=False)]
    assert codes == ["symbolica-unverified"]


def test_release_contract_is_lean_exact_and_schema_8() -> None:
    module = _module()
    lock = module._load_lock()
    assert lock["abis"]["compiled_model"] == 8
    assert "python_runtime_lock" not in lock
    assert "legal_status" not in lock
    assert module._locked_python_dependencies(lock) == (
        module._project_python_dependencies()
    )
    assert set(lock["symbolica"]) == {
        "python_distribution",
        "python_version",
        "rust_crate",
        "rust_version",
        "published_symjit_version",
        "serialization_abi",
        "release_status",
    }
    assert set(lock["ufo_model_loader"]) == {
        "python_distribution",
        "required_version",
        "latest_verified_published_version",
        "release_status",
    }


def test_release_cargo_lock_contains_only_published_registry_crates() -> None:
    module = _module()
    assert module._release_cargo_lock_issues(module._load_lock()) == []


def test_release_toolchain_and_manylinux_image_are_exactly_pinned() -> None:
    module = _module()
    assert module._toolchain_issues(module._load_lock()) == []


def test_release_cargo_lock_rejects_candidate_path_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    contaminated = tmp_path / "Cargo.lock"
    text = module.CARGO_LOCK_PATH.read_text(encoding="utf-8")
    marker = (
        'name = "symbolica"\n'
        'version = "2.1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
    )
    assert marker in text
    contaminated.write_text(text.replace(marker, marker.rsplit("source", 1)[0], 1))
    monkeypatch.setattr(module, "CARGO_LOCK_PATH", contaminated)
    codes = {
        issue.code for issue in module._release_cargo_lock_issues(module._load_lock())
    }
    assert codes == {"release-cargo-nonregistry", "release-cargo-pin"}


def test_candidate_gate_fails_closed_before_contributor_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "STATE_PATH", tmp_path / "install-state.json")
    monkeypatch.setattr(module, "CANDIDATE_LOCK_PATH", tmp_path / "Cargo.lock")
    monkeypatch.setattr(module, "CARGO_CONFIG_PATH", tmp_path / "config.toml")
    codes = {issue.code for issue in module.check(candidate=True, online=False)}
    assert "candidate-input-missing" in codes
    assert "symbolica-unverified" not in codes


def test_candidate_gate_uses_revisions_without_source_tree_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    contributor = module._load_contributor_lock()
    revisions = module._candidate_revisions(contributor)
    state_path = tmp_path / "install-state.json"
    candidate_lock = tmp_path / "candidate-Cargo.lock"
    cargo_config = tmp_path / "candidate-cargo-config.toml"
    checkouts = tmp_path / "checkouts"
    for name in revisions:
        (checkouts / name).mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "sources": {
                    name: {"revision": revision} for name, revision in revisions.items()
                },
            }
        ),
        encoding="utf-8",
    )
    candidate_lock.write_text(
        'version = 4\n\n[[package]]\nname = "rusticol-core"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    cargo_config.write_text(
        f'[patch.crates-io]\nsymbolica = {{ path = "{checkouts / "symbolica"}" }}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "STATE_PATH", state_path)
    monkeypatch.setattr(module, "CANDIDATE_LOCK_PATH", candidate_lock)
    monkeypatch.setattr(module, "CARGO_CONFIG_PATH", cargo_config)
    monkeypatch.setattr(module, "CHECKOUTS_PATH", checkouts)
    monkeypatch.setattr(
        module,
        "_git_head",
        lambda path: revisions[path.name],
    )
    assert module._candidate_issues(module._load_lock()) == []


def test_online_gate_checks_each_exact_published_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    requested: list[str] = []

    def published(url: str) -> bool:
        requested.append(url)
        return True

    monkeypatch.setattr(module, "_published", published)
    issues = module._published_dependency_issues(module._load_lock())
    assert issues == []
    assert len(requested) == len(module._project_python_dependencies()) + 2
    assert all("/json" in url for url in requested if "pypi.org" in url)
