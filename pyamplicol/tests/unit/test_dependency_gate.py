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
        "source_url",
        "archive_prefix",
        "archive_sha256",
        "source_tree_sha256",
        "configured_tree_sha256",
        "release_cargo_lock_sha256",
        "patches",
    }
    assert lock["symjit"]["version"] == "2.22.0"
    assert (
        lock["symjit"]["repository"]
        == "https://github.com/siravan/symjit-crate.git"
    )
    assert (
        lock["symjit"]["revision"]
        == "77789ff0f78232b1ea4608aceb397058df50b06d"
    )
    assert lock["symjit"]["configured_tree_sha256"] == (
        "4b4b791b0f2bbef33a7dbd2936d20dc722f7301e2e9e986b65b2a8b94d220b31"
    )
    assert len(lock["symjit"]["patches"]) == 1
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
    assert codes == {
        "release-cargo-lock-projection",
        "release-cargo-nonregistry",
        "release-cargo-pin",
    }


def test_release_gate_accepts_authenticated_local_symjit_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    lock = module._load_lock()
    source = module.CARGO_LOCK_PATH.read_text(encoding="utf-8")
    symjit = lock["symjit"]
    git_source = (
        f"git+{symjit['repository']}?rev={symjit['revision']}#{symjit['revision']}"
    )
    marker = (
        "[[package]]\n"
        'name = "symjit"\n'
        f'version = "{symjit["version"]}"\n'
        f'source = "{git_source}"\n'
    )
    replacement = (
        "[[package]]\n"
        'name = "symjit"\n'
        f'version = "{symjit["version"]}"\n'
    )
    assert source.count(marker) == 1
    local_lock = tmp_path / "Cargo.lock"
    local_lock.write_text(source.replace(marker, replacement, 1), encoding="utf-8")
    assert hashlib.sha256(local_lock.read_bytes()).hexdigest() == (
        symjit["release_cargo_lock_sha256"]
    )
    monkeypatch.setattr(module, "CARGO_LOCK_PATH", local_lock)

    assert module._release_cargo_lock_issues(lock) == []


def test_release_gate_authenticates_its_generic_symjit_patch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    release = copy.deepcopy(module._load_lock())
    dependencies = tmp_path / "dependencies"
    patch = dependencies / release["symjit"]["patches"][0]["path"]
    patch.parent.mkdir(parents=True)
    patch.write_bytes(
        (
            ROOT
            / "dependencies"
            / release["symjit"]["patches"][0]["path"]
        ).read_bytes()
        + b"\ntampered\n"
    )
    lock_path = dependencies / "release-lock.toml"
    lock_path.write_text("schema_version = 1\n", encoding="utf-8")
    monkeypatch.setattr(module, "LOCK_PATH", lock_path)

    assert {
        issue.code for issue in module._release_symjit_source_issues(release)
    } == {"release-symjit-patch"}


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
    patch_state, patch_issues = module._candidate_patch_contract(contributor)
    assert patch_issues == []
    state_path = tmp_path / "install-state.json"
    candidate_lock = tmp_path / "candidate-Cargo.lock"
    cargo_config = tmp_path / "candidate-cargo-config.toml"
    checkouts = tmp_path / "checkouts"
    for name in revisions:
        (checkouts / name).mkdir(parents=True)
    for name in ("graphica", "numerica"):
        (checkouts / "symbolica" / "lib" / name).mkdir(parents=True)
    source_state = {
        name: {"revision": revision} for name, revision in revisions.items()
    }
    source_state["symjit"].update(
        {
            "version": contributor["symjit"]["candidate_version"],
            "archive_sha256": contributor["symjit"]["archive_sha256"],
            "patch_sha256": module._patch_closure_sha256(patch_state),
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
                "patches": patch_state,
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
        f'graphica = {{ path = "{checkouts / "symbolica/lib/graphica"}" }}\n'
        f'numerica = {{ path = "{checkouts / "symbolica/lib/numerica"}" }}\n'
        f'symbolica = {{ path = "{checkouts / "symbolica"}" }}\n'
        f'symjit = {{ path = "{checkouts / "symjit"}" }}'
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


def test_candidate_contract_rejects_malformed_patch_contract() -> None:
    module = _module()
    contributor = copy.deepcopy(module._load_contributor_lock())
    contributor["patches"] = [
        {
            "name": "synthetic",
            "target": "pyamplicol",
            "path": "patches/symjit/change.patch",
            "sha256": "0" * 64,
            "applies_to_revision": contributor["symjit"]["candidate_revision"],
        }
    ]

    issues = module._candidate_contributor_contract_issues(contributor)

    assert {issue.code for issue in issues} == {
        "candidate-patch-contract",
        "candidate-source-tree",
    }


def test_candidate_contract_accepts_authenticated_generic_symjit_patch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    contributor = copy.deepcopy(module._load_contributor_lock())
    dependencies = tmp_path / "dependencies"
    patch = dependencies / "patches" / "symjit" / "generic.patch"
    patch.parent.mkdir(parents=True)
    patch.write_text("generic patch fixture\n", encoding="utf-8")
    lock_path = dependencies / "contributor-lock.toml"
    lock_path.write_text("schema_version = 1\n", encoding="utf-8")
    monkeypatch.setattr(module, "CONTRIBUTOR_LOCK_PATH", lock_path)
    contributor["patches"] = [
        {
            "name": "generic-plane-descriptor",
            "target": "symjit",
            "path": "patches/symjit/generic.patch",
            "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
            "applies_to_revision": contributor["symjit"]["candidate_revision"],
        }
    ]
    contributor["symjit"]["candidate_tree_sha256"] = "0" * 64

    assert module._candidate_contributor_contract_issues(contributor) == []


def test_candidate_contributor_contract_pins_plane_abi_and_both_tree_states() -> None:
    module = _module()
    contributor = module._load_contributor_lock()

    assert module._candidate_contributor_contract_issues(contributor) == []

    wrong_abi = copy.deepcopy(contributor)
    wrong_abi["abis"]["symjit_plane_application"] = "wrong"
    assert {
        issue.code for issue in module._candidate_contributor_contract_issues(wrong_abi)
    } == {"candidate-abi-contract"}

    wrong_tree = copy.deepcopy(contributor)
    wrong_tree["symjit"]["source_tree_sha256"] = "wrong"
    assert {
        issue.code
        for issue in module._candidate_contributor_contract_issues(wrong_tree)
    } == {"candidate-source-tree"}

    patchless_divergence = copy.deepcopy(contributor)
    patchless_divergence["patches"] = []
    assert {
        issue.code
        for issue in module._candidate_contributor_contract_issues(
            patchless_divergence
        )
    } == {"candidate-source-tree"}


def test_release_contract_pins_symjit_plane_application_abi() -> None:
    module = _module()
    release = module._load_lock()

    assert not {
        issue.code
        for issue in module._release_contract_issues(release)
        if issue.code == "release-abi-contract"
    }

    missing = copy.deepcopy(release)
    del missing["abis"]["symjit_plane_application"]
    assert "release-abi-contract" in {
        issue.code for issue in module._release_contract_issues(missing)
    }

    wrong = copy.deepcopy(release)
    wrong["abis"]["symjit_plane_application"] = "wrong"
    assert "release-abi-contract" in {
        issue.code for issue in module._release_contract_issues(wrong)
    }


def test_release_gate_has_no_live_package_or_repository_preflight() -> None:
    module = _module()
    source = SCRIPT.read_text(encoding="utf-8")
    assert not hasattr(module, "_published_dependency_issues")
    assert "urllib.request" not in source
    assert "pypi.org/pypi/" not in source
    assert "crates.io/api/" not in source
