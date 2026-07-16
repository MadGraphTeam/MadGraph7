# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "release"))

import check_rust_licenses as licenses  # type: ignore[import-not-found]  # noqa: E402


def _write_incomplete_policy(root: Path) -> dict[str, Any]:
    (root / "dependencies").mkdir(parents=True)
    (root / "licenses").mkdir(parents=True)
    (root / "dependencies" / "release-lock.toml").write_text(
        """
[[targets]]
triple = "aarch64-apple-darwin"

[legal_status]
candidate_builds_allowed = true
static_lgpl_compliance_manifest = "licenses/STATIC_LINK_COMPLIANCE.toml"
static_lgpl_release_ready = false
static_lgpl_status = "incomplete"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "licenses" / "STATIC_LINK_COMPLIANCE.toml").write_text(
        """
schema_version = 1
candidate_builds_allowed = true
release_ready = false
status = "incomplete"
coverage = []
evidence = []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return {
        "review": {
            "static_link_policy": "licenses/STATIC_LINK_COMPLIANCE.toml",
        },
        "package": [],
    }


def _metadata_payload(
    command: tuple[str, ...],
    *,
    package: str = "rusticol-python",
    contaminated: bool = False,
) -> str:
    manifest = command[command.index("--manifest-path") + 1]
    root_id = f"path+file://fixture#{package}@0.1.0"
    core_id = "path+file://fixture#rusticol-core@0.1.0"
    symjit_id = "registry+https://example.invalid#symjit@2.19.3"
    build_id = "registry+https://example.invalid#build-helper@1.0.0"
    dev_id = "registry+https://example.invalid#dev-helper@1.0.0"
    symbolica_id = "registry+https://example.invalid#symbolica@2.1.0"
    rug_id = "registry+https://example.invalid#rug@1.28.1"
    malachite_id = "registry+https://example.invalid#malachite-q@0.9.1"

    core_dependencies = [
        {
            "name": "symjit",
            "pkg": symjit_id,
            "dep_kinds": [{"kind": None, "target": None}],
        }
    ]
    if contaminated:
        core_dependencies.append(
            {
                "name": "symbolica",
                "pkg": symbolica_id,
                "dep_kinds": [{"kind": None, "target": None}],
            }
        )

    packages = [
        {
            "id": root_id,
            "name": package,
            "version": "0.1.0",
            "source": None,
            "manifest_path": manifest,
        },
        {
            "id": core_id,
            "name": "rusticol-core",
            "version": "0.1.0",
            "source": None,
            "manifest_path": "/fixture/rusticol-core/Cargo.toml",
        },
        {
            "id": symjit_id,
            "name": "symjit",
            "version": "2.19.3",
            "source": "registry+https://example.invalid",
            "manifest_path": "/fixture/symjit/Cargo.toml",
        },
        {
            "id": build_id,
            "name": "build-helper",
            "version": "1.0.0",
            "source": "registry+https://example.invalid",
            "manifest_path": "/fixture/build-helper/Cargo.toml",
        },
        {
            "id": dev_id,
            "name": "dev-helper",
            "version": "1.0.0",
            "source": "registry+https://example.invalid",
            "manifest_path": "/fixture/dev-helper/Cargo.toml",
        },
        {
            "id": symbolica_id,
            "name": "symbolica",
            "version": "2.1.0",
            "source": "registry+https://example.invalid",
            "manifest_path": "/fixture/symbolica/Cargo.toml",
        },
        {
            "id": rug_id,
            "name": "rug",
            "version": "1.28.1",
            "source": "registry+https://example.invalid",
            "manifest_path": "/fixture/rug/Cargo.toml",
        },
        {
            "id": malachite_id,
            "name": "malachite-q",
            "version": "0.9.1",
            "source": "registry+https://example.invalid",
            "manifest_path": "/fixture/malachite-q/Cargo.toml",
        },
    ]
    nodes = [
        {
            "id": root_id,
            "features": ["extension-module"],
            "deps": [
                {
                    "name": "rusticol_core",
                    "pkg": core_id,
                    "dep_kinds": [{"kind": None, "target": None}],
                },
                {
                    "name": "build_helper",
                    "pkg": build_id,
                    "dep_kinds": [{"kind": "build", "target": None}],
                },
                {
                    "name": "dev_helper",
                    "pkg": dev_id,
                    "dep_kinds": [{"kind": "dev", "target": None}],
                },
            ],
        },
        {"id": core_id, "features": ["f64-symjit"], "deps": core_dependencies},
        {"id": symjit_id, "features": [], "deps": []},
        {"id": build_id, "features": [], "deps": []},
        {"id": dev_id, "features": [], "deps": []},
        {
            "id": symbolica_id,
            "features": ["gmp"],
            "deps": [
                {
                    "name": "rug",
                    "pkg": rug_id,
                    "dep_kinds": [{"kind": None, "target": None}],
                },
                {
                    "name": "malachite_q",
                    "pkg": malachite_id,
                    "dep_kinds": [{"kind": "dev", "target": None}],
                },
            ],
        },
        {"id": rug_id, "features": [], "deps": []},
        {"id": malachite_id, "features": [], "deps": []},
    ]
    return json.dumps(
        {
            "packages": packages,
            "resolve": {"root": root_id, "nodes": nodes},
        }
    )


def test_shipped_artifact_commands_match_release_feature_selection(
    tmp_path: Path,
) -> None:
    by_root = {spec.cargo_root: spec for spec in licenses.SHIPPED_CARGO_ARTIFACTS}
    capi = licenses.cargo_metadata_command(
        tmp_path, by_root["rusticol-capi"], "aarch64-apple-darwin"
    )
    python = licenses.cargo_metadata_command(
        tmp_path, by_root["rusticol-python"], "aarch64-apple-darwin"
    )

    for command in (capi, python):
        assert command[:4] == ("cargo", "metadata", "--format-version", "1")
        assert "--locked" in command
        assert "--offline" in command
        assert command[command.index("--filter-platform") + 1] == (
            "aarch64-apple-darwin"
        )
    assert "--features" not in capi
    assert "--no-default-features" not in capi
    assert python[python.index("--features") + 1] == "extension-module"
    assert "--no-default-features" not in python

    optional = licenses.cargo_metadata_command(
        tmp_path,
        licenses.OPTIONAL_SYMBOLICA_CLOSURES[0],
        "x86_64-unknown-linux-gnu",
    )
    assert "--no-default-features" in optional
    assert optional[optional.index("--features") + 1] == "symbolica-runtime"


def test_feature_closure_keeps_normal_and_build_edges_but_excludes_dev_and_lock_only(
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[str, ...], Path]] = []

    def runner(command: tuple[str, ...], root: Path) -> str:
        calls.append((command, root))
        return _metadata_payload(command)

    spec = next(
        spec
        for spec in licenses.SHIPPED_CARGO_ARTIFACTS
        if spec.cargo_root == "rusticol-python"
    )
    closure = licenses.cargo_dependency_closure(
        tmp_path,
        spec,
        "x86_64-apple-darwin",
        runner=runner,
    )

    assert len(calls) == 1
    assert calls[0][1] == tmp_path.resolve()
    assert {package.name for package in closure.packages} == {
        "build-helper",
        "rusticol-core",
        "rusticol-python",
        "symjit",
    }
    assert licenses.sensitive_dependency_families(closure) == {}


def test_symbolica_bearing_closure_is_target_specific_and_detects_known_families(
    tmp_path: Path,
) -> None:
    spec = next(
        spec
        for spec in licenses.SHIPPED_CARGO_ARTIFACTS
        if spec.cargo_root == "rusticol-python"
    )
    closure = licenses.cargo_dependency_closure(
        tmp_path,
        spec,
        "x86_64-unknown-linux-gnu",
        runner=lambda command, _root: _metadata_payload(command, contaminated=True),
    )

    assert licenses.sensitive_dependency_families(closure) == {
        "gmp": ("rug@1.28.1",),
        "symbolica": ("symbolica@2.1.0",),
    }
    issues = licenses._shipped_closure_issues((closure,))
    assert len(issues) == 1
    assert issues[0].code == "static-native-forbidden-dependency"
    assert "pyamplicol._rusticol" in issues[0].message
    assert "x86_64-unknown-linux-gnu" in issues[0].message
    assert "symbolica@2.1.0" in issues[0].message
    assert "rug@1.28.1" in issues[0].message


def test_known_malachite_family_remains_available_for_optional_closure() -> None:
    spec = licenses.OPTIONAL_SYMBOLICA_CLOSURES[0]
    closure = licenses.CargoClosure(
        spec=spec,
        target="aarch64-apple-darwin",
        command=("cargo", "metadata"),
        packages=(
            licenses.CargoClosurePackage(
                name="malachite-q",
                version="0.9.1",
                source="registry+fixture",
                package_id="malachite-q@0.9.1",
            ),
            licenses.CargoClosurePackage(
                name="symbolica",
                version="2.1.0",
                source="registry+fixture",
                package_id="symbolica@2.1.0",
            ),
        ),
        activated_features=(),
    )

    assert licenses.sensitive_dependency_families(closure) == {
        "malachite": ("malachite-q@0.9.1",),
        "symbolica": ("symbolica@2.1.0",),
    }
    assert licenses._shipped_closure_issues((closure,)) == []


def test_clean_shipped_closures_still_require_final_binary_scan(tmp_path: Path) -> None:
    inventory = _write_incomplete_policy(tmp_path)
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...], _root: Path) -> str:
        calls.append(command)
        manifest = command[command.index("--manifest-path") + 1]
        package = "rusticol-capi" if "rusticol-capi" in manifest else "rusticol-python"
        return _metadata_payload(command, package=package)

    issues = licenses._static_link_compliance_issues(
        tmp_path,
        {"package": []},
        inventory,
        runner=runner,
        verify_corpus=False,
    )

    assert len(calls) == 2
    assert {issue.code for issue in issues} == {"static-native-binary-audit-pending"}
    assert all(issue.release_only for issue in issues)


def test_corpus_package_set_is_exact_union_of_release_feature_closures() -> None:
    source = "registry+https://example.invalid"
    root = licenses.CargoClosurePackage(
        name="rusticol-core",
        version="0.1.0",
        source="",
        package_id="path+fixture#rusticol-core@0.1.0",
    )
    common = licenses.CargoClosurePackage(
        name="common",
        version="1.0.0",
        source=source,
        package_id=f"{source}#common@1.0.0",
    )
    platform = licenses.CargoClosurePackage(
        name="platform",
        version="2.0.0",
        source=source,
        package_id=f"{source}#platform@2.0.0",
    )
    spec = licenses.SHIPPED_CARGO_ARTIFACTS[0]
    closures = (
        licenses.CargoClosure(
            spec=spec,
            target="aarch64-apple-darwin",
            command=("cargo", "metadata"),
            packages=(root, common),
            activated_features=(),
        ),
        licenses.CargoClosure(
            spec=spec,
            target="x86_64-unknown-linux-gnu",
            command=("cargo", "metadata"),
            packages=(root, common, platform),
            activated_features=(),
        ),
    )
    inventory = {
        "rust_license_corpus": {
            "package": [
                {"name": "common", "version": "1.0.0", "source": source},
                {"name": "platform", "version": "2.0.0", "source": source},
            ]
        }
    }

    assert licenses._rust_license_corpus_closure_issues(inventory, closures) == []

    inventory["rust_license_corpus"]["package"].pop()
    issues = licenses._rust_license_corpus_closure_issues(inventory, closures)
    assert {issue.code for issue in issues} == {"rust-license-corpus-package-missing"}
    assert not issues[0].release_only


def test_unresolvable_corpus_closure_blocks_candidate_mode(tmp_path: Path) -> None:
    inventory = _write_incomplete_policy(tmp_path)

    def runner(_command: tuple[str, ...], _root: Path) -> str:
        raise licenses.CargoClosureError("offline registry source is absent")

    issues = licenses._static_link_compliance_issues(
        tmp_path,
        {"package": []},
        inventory,
        runner=runner,
    )

    by_code = {issue.code: issue for issue in issues}
    assert "rust-license-corpus-closure" in by_code
    assert by_code["rust-license-corpus-closure"].release_only is False
    assert licenses.blocking_issues(issues, mode="candidate") == [
        by_code["rust-license-corpus-closure"]
    ]
