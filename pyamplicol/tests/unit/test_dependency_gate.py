# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import copy
import hashlib
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


def test_release_gate_is_ready_without_network_preflights() -> None:
    module = _module()
    codes = [issue.code for issue in module.check(candidate=False)]
    assert codes == []


def test_release_contract_is_lean_exact_and_schema_8() -> None:
    module = _module()
    lock = module._load_lock()
    assert lock["abis"]["compiled_model"] == 9
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
        "serialization_abi",
    }
    assert set(lock["symjit"]) == {
        "version",
        "repository",
        "revision",
    }
    assert len(lock["symjit"]["revision"]) == 40
    assert set(lock["ufo_model_loader"]) == {
        "python_distribution",
        "required_version",
    }


def test_release_cargo_lock_contains_registry_crates_and_exact_symjit_git() -> None:
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
        'version = "2.2.0"\n'
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
    codes = {issue.code for issue in module.check(candidate=True)}
    assert "candidate-input-missing" in codes
    assert "symbolica-unverified" not in codes


def test_candidate_gate_uses_revisions_and_pinned_symjit_tree_fingerprint(
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
    for name in ("graphica", "numerica"):
        (checkouts / name).mkdir()
    source_state = {
        name: {"revision": revision} for name, revision in revisions.items()
    }
    source_state["symjit"].update(
        {
            "version": contributor["symjit"]["candidate_version"],
            "archive_sha256": contributor["symjit"]["archive_sha256"],
            "patch_sha256": module._EMPTY_PATCH_CLOSURE_SHA256,
            "worktree_sha256": contributor["symjit"]["candidate_tree_sha256"],
        }
    )
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "contributor_lock_sha256": hashlib.sha256(
                    module.CONTRIBUTOR_LOCK_PATH.read_bytes()
                ).hexdigest(),
                "sources": source_state,
                "patches": [],
            }
        ),
        encoding="utf-8",
    )
    candidate_lock.write_text(
        'version = 4\n\n[[package]]\nname = "rusticol-core"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    cargo_config.write_text(
        "[patch.crates-io]\n"
        + "\n".join(
            f'{name} = {{ path = "{checkouts / name}" }}'
            for name in ("graphica", "numerica", "symbolica", "symjit")
        )
        + "\n",
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
    monkeypatch.setattr(
        module,
        "_source_tree_sha256",
        lambda _path: contributor["symjit"]["candidate_tree_sha256"],
    )
    assert module._candidate_issues(module._load_lock()) == []


def test_candidate_contract_rejects_any_local_patch_contract() -> None:
    module = _module()
    contributor = copy.deepcopy(module._load_contributor_lock())
    contributor["patches"] = [
        {
            "name": "synthetic",
            "target": "symjit",
            "path": "patches/symjit/change.patch",
            "sha256": "0" * 64,
            "applies_to_revision": contributor["symjit"]["candidate_revision"],
        }
    ]

    issues = module._candidate_contributor_contract_issues(contributor)

    assert [issue.code for issue in issues] == ["candidate-patch-contract"]


def test_candidate_contributor_contract_pins_arena_abis_and_both_tree_states() -> None:
    module = _module()
    contributor = module._load_contributor_lock()

    assert module._candidate_contributor_contract_issues(contributor) == []

    wrong_abi = copy.deepcopy(contributor)
    wrong_abi["abis"]["symjit_direct_table_binding"] = "wrong"
    assert {
        issue.code for issue in module._candidate_contributor_contract_issues(wrong_abi)
    } == {"candidate-abi-contract"}

    wrong_tree = copy.deepcopy(contributor)
    wrong_tree["symjit"]["source_tree_sha256"] = "wrong"
    assert {
        issue.code
        for issue in module._candidate_contributor_contract_issues(wrong_tree)
    } == {"candidate-source-tree"}

    divergent_tree = copy.deepcopy(contributor)
    divergent_tree["symjit"]["source_tree_sha256"] = "0" * 64
    assert {
        issue.code
        for issue in module._candidate_contributor_contract_issues(divergent_tree)
    } == {"candidate-source-tree"}


def test_release_gate_has_no_live_package_or_repository_preflight() -> None:
    module = _module()
    source = SCRIPT.read_text(encoding="utf-8")
    assert not hasattr(module, "_published_dependency_issues")
    assert "urllib.request" not in source
    assert "pypi.org/pypi/" not in source
    assert "crates.io/api/" not in source
