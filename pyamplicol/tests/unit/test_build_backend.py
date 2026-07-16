# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build_backend"))

import _pyamplicol_build as backend  # noqa: E402
import distribution_sbom as distribution  # noqa: E402
import sdk  # noqa: E402


def _cargo_distribution_sbom(version: str, *, reverse: bool = False) -> bytes:
    cargo_version = version.replace(".dev0+", "-dev.0+")
    root_ref = f"path+file:///private/build/rusticol-python#{cargo_version}"
    core_ref = f"path+file:///Users/builder/rusticol-core#{cargo_version}"
    build_ref = (
        "registry+https://github.com/rust-lang/crates.io-index#build-helper@1.2.3"
    )
    components = [
        {
            "type": "library",
            "bom-ref": core_ref,
            "name": "rusticol-core",
            "version": cargo_version,
            "scope": "required",
            "purl": (
                f"pkg:cargo/rusticol-core@{cargo_version}"
                "?download_url=file:///Users/builder/rusticol-core"
            ),
            "licenses": [{"expression": "0BSD"}],
        },
        {
            "type": "library",
            "bom-ref": build_ref,
            "name": "build-helper",
            "version": "1.2.3",
            "scope": "required",
            "purl": "pkg:cargo/build-helper@1.2.3",
            "hashes": [{"alg": "SHA-256", "content": "c" * 64}],
            "licenses": [{"expression": "MIT"}],
        },
    ]
    dependencies = [
        {"ref": root_ref, "dependsOn": [core_ref, build_ref]},
        {"ref": core_ref, "dependsOn": []},
        {"ref": build_ref, "dependsOn": []},
    ]
    if reverse:
        components.reverse()
        dependencies.reverse()
        dependencies[-1]["dependsOn"].reverse()
    return json.dumps(
        {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "version": 1,
            "metadata": {
                "timestamp": "2026-07-16T00:00:00Z",
                "component": {
                    "type": "library",
                    "bom-ref": root_ref,
                    "name": "rusticol-python",
                    "version": cargo_version,
                    "scope": "required",
                    "purl": (
                        f"pkg:cargo/rusticol-python@{cargo_version}"
                        "?download_url=file:///private/build/rusticol-python"
                    ),
                    "licenses": [{"expression": "0BSD"}],
                    "components": [
                        {
                            "type": "library",
                            "bom-ref": f"{root_ref} bin-target-0",
                            "name": "_rusticol",
                            "version": cargo_version,
                            "purl": (
                                f"pkg:cargo/rusticol-python@{cargo_version}"
                                "?download_url=file://.#src/lib.rs"
                            ),
                        }
                    ],
                },
                "properties": [
                    {"name": "cdx:rustc:sbom:target:all_targets", "value": "true"}
                ],
            },
            "components": components,
            "dependencies": dependencies,
        }
    ).encode()


def _python_locks(*, complete: bool = True) -> tuple[bytes, bytes]:
    beta_artifact = (
        """
  [[packages.artifacts]]
  filename = "beta-2.0-py3-none-any.whl"
  sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
"""
        if complete
        else ""
    )
    runtime_lock = f"""\
schema_version = 1
requires_python = ">=3.11"
supported_python = ["3.11"]

[[packages]]
distribution = "alpha-runtime"
version = "1.0"
license = "MIT"
direct = true
dependencies = ["beta"]

  [[packages.artifacts]]
  filename = "alpha_runtime-1.0-py3-none-any.whl"
  sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

[[packages]]
distribution = "beta"
version = "2.0"
license = "BSD-3-Clause"
direct = false
dependencies = []
{beta_artifact}
""".encode()
    digest = hashlib.sha256(runtime_lock).hexdigest()
    release_lock = f"""\
schema_version = 1

[project]
distribution = "pyamplicol"
version = "0.1.0"
python_requires = ">=3.11"
license = "0BSD"

[toolchain]

[python_runtime_lock]
path = "dependencies/python-runtime-lock.toml"
sha256 = "{digest}"

[[python_dependencies]]
distribution = "alpha-runtime"
version = "1.0"
license = "MIT"
""".encode()
    return release_lock, runtime_lock


def _distribution_document(
    *,
    mode: str = "release",
    complete: bool = True,
    reverse: bool = False,
) -> bytes:
    version = "0.1.0" if mode == "release" else "0.1.0.dev0+candidate.123456789abc"
    release_lock, runtime_lock = _python_locks(complete=complete)
    return distribution.build_distribution_sbom(
        _cargo_distribution_sbom(version, reverse=reverse),
        release_lock,
        runtime_lock,
        distribution_name="pyamplicol",
        distribution_version=version,
        runtime_requirements=("alpha-runtime==1.0",),
        mode=mode,
    )


def test_cyclonedx_normalization_removes_build_and_candidate_paths() -> None:
    root_ref = "path+file:///private/build/source/rusticol-python#0.1.0"
    core_ref = "path+file:///private/build/source/rusticol-core#0.1.0"
    symjit_ref = "path+file:///Users/developer/checkouts/symjit#2.19.3"
    payload = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": "urn:uuid:11111111-1111-4111-8111-111111111111",
        "metadata": {
            "timestamp": "2026-07-15T00:00:00Z",
            "component": {
                "bom-ref": root_ref,
                "name": "rusticol-python",
                "version": "0.1.0",
                "purl": "pkg:cargo/rusticol-python@0.1.0?download_url=file://.",
            },
        },
        "components": [
            {
                "bom-ref": core_ref,
                "name": "rusticol-core",
                "version": "0.1.0",
                "purl": "pkg:cargo/rusticol-core@0.1.0?download_url=file://../core",
            },
            {
                "bom-ref": symjit_ref,
                "name": "symjit",
                "version": "2.19.3",
                "purl": (
                    "pkg:cargo/symjit@2.19.3?download_url="
                    "file:///Users/developer/checkouts/symjit"
                ),
            },
        ],
        "dependencies": [
            {"ref": root_ref, "dependsOn": [core_ref]},
            {"ref": core_ref, "dependsOn": [symjit_ref]},
        ],
    }

    normalized = backend._normalize_cyclonedx(json.dumps(payload).encode())
    decoded = json.loads(normalized)

    assert b"file://" not in normalized
    assert b"/Users/" not in normalized
    assert b"/private/" not in normalized
    assert decoded["metadata"]["timestamp"] == "1970-01-01T00:00:00Z"
    assert decoded["dependencies"] == [
        {
            "ref": "pkg:cargo/rusticol-python@0.1.0",
            "dependsOn": ["pkg:cargo/rusticol-core@0.1.0"],
        },
        {
            "ref": "pkg:cargo/rusticol-core@0.1.0",
            "dependsOn": ["pkg:cargo/symjit@2.19.3"],
        },
    ]
    assert backend._normalize_cyclonedx(normalized) == normalized


def test_distribution_sbom_merges_complete_rust_and_locked_python_graphs() -> None:
    encoded = _distribution_document()
    document = json.loads(encoded)
    root = document["metadata"]["component"]
    components = {component["purl"]: component for component in document["components"]}
    edges = {
        dependency["ref"]: dependency["dependsOn"]
        for dependency in document["dependencies"]
    }

    assert root == {
        "type": "library",
        "bom-ref": "pkg:pypi/pyamplicol@0.1.0",
        "name": "pyamplicol",
        "version": "0.1.0",
        "scope": "required",
        "purl": "pkg:pypi/pyamplicol@0.1.0",
        "licenses": [{"expression": "0BSD"}],
    }
    assert {
        "pkg:pypi/alpha-runtime@1.0",
        "pkg:pypi/beta@2.0",
    } <= components.keys()
    alpha = components["pkg:pypi/alpha-runtime@1.0"]
    assert alpha["hashes"] == [{"alg": "SHA-256", "content": "a" * 64}]
    artifact_properties = [
        json.loads(item["value"])
        for item in alpha["properties"]
        if item["name"] == "pyamplicol:pypi:artifact"
    ]
    assert artifact_properties == [
        {
            "filename": "alpha_runtime-1.0-py3-none-any.whl",
            "sha256": "a" * 64,
        }
    ]
    build = components["pkg:cargo/build-helper@1.2.3"]
    assert build["hashes"] == [{"alg": "SHA-256", "content": "c" * 64}]
    assert build["licenses"] == [{"expression": "MIT"}]

    assert edges[root["bom-ref"]] == [
        "pkg:cargo/rusticol-python@0.1.0",
        "pkg:pypi/alpha-runtime@1.0",
    ]
    assert edges["pkg:pypi/alpha-runtime@1.0"] == ["pkg:pypi/beta@2.0"]
    assert edges["pkg:pypi/beta@2.0"] == []
    assert edges["pkg:cargo/rusticol-python@0.1.0"] == [
        "pkg:cargo/build-helper@1.2.3",
        "pkg:cargo/rusticol-core@0.1.0",
    ]
    assert set(edges) == {root["bom-ref"], *components}
    assert b"file://" not in encoded
    assert b"path+file:" not in encoded
    assert b"/Users/" not in encoded
    assert b"/private/" not in encoded


def test_distribution_sbom_is_deterministic_for_shuffled_cargo_graph() -> None:
    assert _distribution_document() == _distribution_document(reverse=True)


def test_distribution_sbom_distinguishes_candidate_and_release_locks() -> None:
    candidate = json.loads(_distribution_document(mode="candidate", complete=False))
    root = candidate["metadata"]["component"]
    assert root["purl"] == ("pkg:pypi/pyamplicol@0.1.0.dev0%2Bcandidate.123456789abc")
    assert any(
        component["purl"]
        == "pkg:cargo/rusticol-python@0.1.0-dev.0+candidate.123456789abc"
        for component in candidate["components"]
    )
    beta = next(
        component
        for component in candidate["components"]
        if component.get("purl") == "pkg:pypi/beta@2.0"
    )
    assert {item["name"]: item["value"] for item in beta["properties"]}[
        "pyamplicol:pypi:hashes-verified"
    ] == "false"

    with pytest.raises(RuntimeError, match="lacks verified Python artifact"):
        _distribution_document(mode="release", complete=False)


def test_distribution_sbom_rejects_a_tampered_python_runtime_lock() -> None:
    release_lock, runtime_lock = _python_locks()
    with pytest.raises(RuntimeError, match="fails its SHA-256 binding"):
        distribution.build_distribution_sbom(
            _cargo_distribution_sbom("0.1.0"),
            release_lock,
            runtime_lock + b"\n# tampered\n",
            distribution_name="pyamplicol",
            distribution_version="0.1.0",
            runtime_requirements=("alpha-runtime==1.0",),
            mode="release",
        )


def test_wheel_normalization_only_rewrites_distribution_sbom(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "pyamplicol-0.1.0-cp311-abi3-macosx_11_0_arm64.whl"
    distribution_sbom = json.dumps(
        {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "serialNumber": "urn:uuid:11111111-1111-4111-8111-111111111111",
            "metadata": {"timestamp": "2026-07-15T00:00:00Z"},
            "components": [],
            "dependencies": [],
        }
    ).encode()
    sdk_sbom = b'{"bomFormat":"CycloneDX","sdk":true}\n'
    distribution_path = (
        "pyamplicol-0.1.0.dist-info/sboms/rusticol-python.cyclonedx.json"
    )
    sdk_path = "pyamplicol/_sdk/sboms/rusticol-capi.cyclonedx.json"
    metadata_path = "pyamplicol-0.1.0.dist-info/METADATA"
    record_path = "pyamplicol-0.1.0.dist-info/RECORD"
    release_lock = b"release lock\n"
    runtime_lock = b"runtime lock\n"
    merged_sbom = b'{"bomFormat":"CycloneDX","merged":true}\n'
    calls: list[tuple[object, ...]] = []

    def fake_builder(*args: object, **kwargs: object) -> bytes:
        calls.append((*args, kwargs))
        return merged_sbom

    monkeypatch.setattr(backend, "build_distribution_sbom", fake_builder)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(distribution_path, distribution_sbom)
        archive.writestr(sdk_path, sdk_sbom)
        archive.writestr(
            metadata_path,
            b"Metadata-Version: 2.4\n"
            b"Name: pyamplicol\n"
            b"Version: 0.1.0\n"
            b"Requires-Dist: alpha-runtime==1.0\n\n",
        )
        archive.writestr(backend._DISTRIBUTION_LOCK_MEMBER, release_lock)
        archive.writestr(backend._PYTHON_RUNTIME_LOCK_MEMBER, runtime_lock)
        archive.writestr(record_path, b"")

    backend._normalize_built_wheel(wheel)

    with zipfile.ZipFile(wheel) as archive:
        assert archive.read(sdk_path) == sdk_sbom
        assert archive.read(distribution_path) == merged_sbom
        assert archive.read(record_path)
    assert calls == [
        (
            distribution_sbom,
            release_lock,
            runtime_lock,
            {
                "distribution_name": "pyamplicol",
                "distribution_version": "0.1.0",
                "runtime_requirements": ("alpha-runtime==1.0",),
                "mode": "release",
            },
        )
    ]


def _candidate_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    candidate_lock = tmp_path / "Cargo.lock"
    candidate_config = tmp_path / "config.toml"
    installer_state = tmp_path / "install-state.json"
    candidate_lock.write_bytes((ROOT / "Cargo.lock").read_bytes())
    candidate_config.write_text("[patch.crates-io]\n", encoding="utf-8")
    sources = {
        name: {
            "revision": hashlib.sha256(name.encode()).hexdigest()[:40],
            "worktree_sha256": hashlib.sha256(f"tree:{name}".encode()).hexdigest(),
        }
        for name in backend._CANDIDATE_SOURCES
    }
    installer_state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "release_lock_sha256": hashlib.sha256(
                    (ROOT / "dependencies" / "release-lock.toml").read_bytes()
                ).hexdigest(),
                "python_runtime_lock_sha256": hashlib.sha256(
                    (ROOT / "dependencies" / "python-runtime-lock.toml").read_bytes()
                ).hexdigest(),
                "candidate_lock_sha256": hashlib.sha256(
                    candidate_lock.read_bytes()
                ).hexdigest(),
                "cargo_config_sha256": hashlib.sha256(
                    candidate_config.read_bytes()
                ).hexdigest(),
                "sources": sources,
            }
        ),
        encoding="utf-8",
    )
    return candidate_lock, candidate_config, installer_state


def test_candidate_overlay_is_versioned_without_mutating_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_cargo = (ROOT / "Cargo.toml").read_bytes()
    source_lock = (ROOT / "Cargo.lock").read_bytes()
    source_core = (
        ROOT / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
    ).read_bytes()
    candidate_inputs = _candidate_inputs(tmp_path)
    monkeypatch.setattr(
        backend,
        "_candidate_inputs",
        lambda: candidate_inputs,
    )

    with backend._overlay("candidate") as (overlay, target):
        assert target.parent == overlay.parent
        cargo = (overlay / "Cargo.toml").read_text(encoding="utf-8")
        lock = (overlay / "Cargo.lock").read_text(encoding="utf-8")
        assert (overlay / ".cargo" / "config.toml").read_bytes() == (
            candidate_inputs[1].read_bytes()
        )
        assert 'version = "0.1.0-dev.0+candidate.' in cargo
        assert lock.count('version = "0.1.0-dev.0+candidate.') == 3
        candidate_core = (
            overlay / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
        ).read_text(encoding="utf-8")
        assert 'symjit = { version = "=2.19.3"' in candidate_core
        build_info = json.loads(
            (overlay / "src" / "pyamplicol" / "_build_info.json").read_text(
                encoding="utf-8"
            )
        )
        assert build_info["publishable"] is False
        assert len(build_info["candidate_fingerprint"]) == 12
        target = "aarch64-apple-darwin"
        backend._stage_selftest_fixture(overlay, target)
        selftest_manifest = json.loads(
            (
                overlay
                / "src"
                / "pyamplicol"
                / "assets"
                / "selftest"
                / target
                / "artifact"
                / "artifact.json"
            ).read_text(encoding="utf-8")
        )
        assert selftest_manifest["producer"]["version"] == build_info["version"]
        assert selftest_manifest["runtime"]["engine_version"] == build_info["version"]
        assert selftest_manifest["producer"]["target"] == {
            "cpu_features": [],
            "triple": target,
        }
        target_payloads = [
            payload["target"]
            for payload in selftest_manifest["payloads"]
            if "target" in payload
        ]
        assert target_payloads
        assert target_payloads == [
            {"cpu_features": [], "triple": target} for _payload in target_payloads
        ]
        expected = json.loads(
            (
                overlay
                / "src"
                / "pyamplicol"
                / "assets"
                / "selftest"
                / target
                / "expected.json"
            ).read_text(encoding="utf-8")
        )
        assert expected["target"] == target
        assert expected["compatible_targets"] == sorted(
            backend._PORTABLE_SELFTEST_TARGETS
        )
        assert not (
            overlay
            / "src"
            / "pyamplicol"
            / "assets"
            / "selftest"
            / backend._PORTABLE_SELFTEST_TEMPLATE
        ).exists()
        content = dict(selftest_manifest)
        claimed_id = content.pop("artifact_id")
        canonical = (
            json.dumps(content, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        assert claimed_id == hashlib.sha256(canonical).hexdigest()

    assert (ROOT / "Cargo.toml").read_bytes() == source_cargo
    assert (ROOT / "Cargo.lock").read_bytes() == source_lock
    assert (
        ROOT / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
    ).read_bytes() == source_core


def test_release_overlay_uses_only_canonical_registry_lock() -> None:
    with backend._overlay("release") as (overlay, _target):
        assert (overlay / "Cargo.lock").read_bytes() == (
            ROOT / "Cargo.lock"
        ).read_bytes()
        assert not (overlay / ".cargo" / "config.toml").exists()
        assert not (overlay / "dependencies" / "candidate-Cargo.lock").exists()


def test_selftest_staging_rejects_an_unavailable_target() -> None:
    with (
        backend._overlay("release") as (overlay, _target),
        pytest.raises(RuntimeError, match="unsupported self-test target"),
    ):
        backend._stage_selftest_fixture(overlay, "x86_64-unknown-test")


@pytest.mark.parametrize("target", sorted(backend._PORTABLE_SELFTEST_TARGETS))
def test_portable_selftest_stages_for_every_release_target(target: str) -> None:
    with backend._overlay("release") as (overlay, _target):
        backend._stage_selftest_fixture(overlay, target)
        fixture_root = overlay / "src" / "pyamplicol" / "assets" / "selftest"
        assert tuple(path.name for path in fixture_root.iterdir()) == (target,)
        manifest = json.loads(
            (fixture_root / target / "artifact" / "artifact.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["producer"]["target"]["triple"] == target
        assert {
            payload["target"]["triple"]
            for payload in manifest["payloads"]
            if "target" in payload
        } == {target}


def test_cargo_overlay_rejects_unknown_resolution_mode(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unsupported Cargo input mode"):
        backend._stage_cargo_inputs(tmp_path, "mixed")


def test_candidate_inputs_fail_closed_when_installer_has_not_run() -> None:
    expected = (
        ROOT / "dependencies" / "candidate-Cargo.lock",
        ROOT / "dependencies" / "candidate-cargo-config.toml",
        ROOT / "dependencies" / "install-state.json",
    )
    if all(path.is_file() for path in expected):
        pytest.skip("candidate environment is installed")
    with pytest.raises(RuntimeError, match="dev-install"):
        backend._candidate_inputs()


def test_overlay_excludes_managed_dependencies_and_includes_licenses() -> None:
    with backend._overlay("release") as (overlay, _target):
        assert (overlay / "config" / "release-dependencies.toml").is_file()
        assert (overlay / "licenses" / "MadGraph5_aMCatNLO.txt").is_file()
        assert (overlay / "licenses" / "RUST_THIRD_PARTY.toml").is_file()
        assert (overlay / "licenses" / "STATIC_LINK_COMPLIANCE.toml").is_file()
        assert (overlay / "rust-toolchain.toml").is_file()
        assert (overlay / "tools" / "typing" / "check_public_typing.py").is_file()
        assert not (overlay / "dependencies" / "checkouts").exists()
        assert not (overlay / "dependencies" / "install-state.json").exists()


def test_wheel_examples_are_staged_from_the_single_canonical_tree() -> None:
    with backend._overlay("release") as (overlay, _target):
        packaged = overlay / "src" / "pyamplicol" / "_examples"
        assert not packaged.exists()
        backend._stage_packaged_examples(overlay)
        assert (packaged / "all_options.toml").is_file()
        assert (packaged / "native" / "runtime.cpp").is_file()
        assert (packaged / "all_options.toml").read_bytes() == (
            overlay / "examples" / "all_options.toml"
        ).read_bytes()
        with pytest.raises(RuntimeError, match="already contains"):
            backend._stage_packaged_examples(overlay)


def test_wheel_stages_the_single_maintained_rusticol_stub() -> None:
    with backend._overlay("release") as (overlay, _target):
        source = (
            overlay
            / "rust"
            / "crates"
            / "rusticol-python"
            / "stubs"
            / "pyamplicol"
            / "_rusticol.pyi"
        )
        target = overlay / "src" / "pyamplicol" / "_rusticol.pyi"
        assert source.is_file()
        assert not target.exists()
        backend._stage_python_stub(overlay)
        assert target.read_bytes() == source.read_bytes()
        with pytest.raises(RuntimeError, match="already contains"):
            backend._stage_python_stub(overlay)


def test_wheel_stages_runtime_resources_inside_package_namespace(
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "overlay"
    for relative in (
        Path("dependencies/release-lock.toml"),
        Path("dependencies/python-runtime-lock.toml"),
        Path("schemas/README.md"),
        Path("schemas/artifact-manifest-v3.schema.json"),
        Path("schemas/runtime-physics-v1.schema.json"),
    ):
        target = overlay / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())

    backend._stage_runtime_resources(overlay)
    package_assets = overlay / "src" / "pyamplicol" / "assets"
    assert (package_assets / "release" / "release-lock.toml").read_bytes() == (
        overlay / "dependencies" / "release-lock.toml"
    ).read_bytes()
    assert (package_assets / "release" / "python-runtime-lock.toml").read_bytes() == (
        overlay / "dependencies" / "python-runtime-lock.toml"
    ).read_bytes()
    assert (
        package_assets / "schemas" / "artifact-manifest-v3.schema.json"
    ).read_bytes() == (
        overlay / "schemas" / "artifact-manifest-v3.schema.json"
    ).read_bytes()
    assert (package_assets / "schemas" / "runtime-physics-v1.schema.json").is_file()
    with pytest.raises(RuntimeError, match="already contains"):
        backend._stage_runtime_resources(overlay)


def test_overlay_rejects_symlinked_inputs(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("content", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(RuntimeError, match="symlinks"):
        backend._reject_symlinks(link)


def test_overlay_prunes_excluded_directories_before_symlink_walk(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    excluded = source / "checkouts"
    excluded.mkdir(parents=True)
    (excluded / "escape").symlink_to(tmp_path)
    backend._reject_symlinks(source)


def test_git_overlay_copies_only_tracked_files_and_prunes_ignored_symlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", source], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "Test"],
        check=True,
    )
    tracked = source / "build_backend" / "tracked.py"
    tracked.parent.mkdir()
    tracked.write_text("tracked = True\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(source), "add", "build_backend/tracked.py"],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "tracked"], check=True)
    (tracked.parent / "untracked.py").write_text("leak = True\n", encoding="utf-8")
    excluded = source / "dependencies" / "checkouts"
    excluded.mkdir(parents=True)
    (excluded / "escape").symlink_to(tmp_path)

    monkeypatch.setattr(backend, "ROOT", source)
    destination = tmp_path / "destination"
    backend._copy_allowlisted_source(destination)
    assert (destination / "build_backend" / "tracked.py").is_file()
    assert not (destination / "build_backend" / "untracked.py").exists()
    assert not (destination / "dependencies" / "checkouts").exists()


def test_archive_overlay_without_git_history_uses_pruned_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "archive"
    retained = {
        Path("build_backend/backend.py"): "archive = True\n",
        Path("docs/pyAmpliCol.tex"): "maintained TeX\n",
        Path("src/pyamplicol/_sdk/config.py"): "maintained SDK config\n",
        Path("tests/fixtures/candidate-Cargo.lock"): "fixture lock\n",
    }
    excluded = (
        Path("dependencies/candidate-Cargo.lock"),
        Path("dependencies/candidate-cargo-config.toml"),
        Path("dependencies/install-state.json"),
        Path("docs/.result_outputs/cache.json"),
        Path("docs/archive/retired.md"),
        Path("docs/pyAmpliCol.aux"),
        Path("docs/pyAmpliCol.synctex.gz"),
        Path("docs/pyAmpliCol.toc"),
        Path("src/pyamplicol.egg-info/PKG-INFO"),
        Path("src/pyamplicol/_rusticol.abi3.so"),
        Path("src/pyamplicol/_rusticol.dylib"),
        Path("src/pyamplicol/_rusticol.pyd"),
        Path("src/pyamplicol/_sdk/fortran/rusticol.f90"),
        Path("src/pyamplicol/_sdk/include/rusticol.h"),
        Path("src/pyamplicol/_sdk/lib/librusticol_capi.a"),
        Path("src/pyamplicol/_sdk/sboms/rusticol-capi.cyclonedx.json"),
        Path("src/pyamplicol/_sdk/link.json"),
        Path("src/pyamplicol/_sdk/metadata.json"),
        Path("tests/.artifacts/build.json"),
        Path("tests/build/output.txt"),
        Path("tests/htmlcov/index.html"),
    )
    for relative, content in retained.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for relative in excluded:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated\n", encoding="utf-8")
    ignored = source / "build_backend" / "__pycache__"
    ignored.mkdir()
    (ignored / "escape").symlink_to(tmp_path)
    monkeypatch.setattr(backend, "ROOT", source)

    destination = tmp_path / "destination"
    backend._copy_allowlisted_source(destination)
    for relative, content in retained.items():
        assert (destination / relative).read_text(encoding="utf-8") == content
    for relative in excluded:
        assert not (destination / relative).exists()
    assert not (destination / "build_backend" / "__pycache__").exists()

    copy_ignore = backend._copy_ignore(source)
    assert copy_ignore(str(source / "docs"), ["pyAmpliCol.aux", "pyAmpliCol.tex"]) == {
        "pyAmpliCol.aux"
    }
    assert copy_ignore(
        str(source / "src" / "pyamplicol" / "_sdk"),
        ["config.py", "lib", "metadata.json"],
    ) == {"lib", "metadata.json"}


def test_candidate_digest_covers_state_sources_and_exact_lock_inputs(
    tmp_path: Path,
) -> None:
    inputs = _candidate_inputs(tmp_path)
    overlay = tmp_path / "overlay"
    dependencies = overlay / "dependencies"
    dependencies.mkdir(parents=True)
    for name in ("release-lock.toml", "python-runtime-lock.toml"):
        (dependencies / name).write_bytes((ROOT / "dependencies" / name).read_bytes())
    first = backend._candidate_digest(overlay, *inputs)

    source = overlay / "src" / "pyamplicol" / "api.py"
    source.parent.mkdir(parents=True)
    source.write_text("# changed source\n", encoding="utf-8")
    source_tree_changed = backend._candidate_digest(overlay, *inputs)
    assert source_tree_changed != first

    state = json.loads(inputs[2].read_text(encoding="utf-8"))
    state["python_runtime_lock_sha256"] = "0" * 64
    inputs[2].write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="python_runtime_lock_sha256"):
        backend._candidate_digest(overlay, *inputs)
    state["python_runtime_lock_sha256"] = hashlib.sha256(
        (dependencies / "python-runtime-lock.toml").read_bytes()
    ).hexdigest()

    state["sources"]["symbolica"]["worktree_sha256"] = "f" * 64
    inputs[2].write_text(json.dumps(state), encoding="utf-8")
    source_changed = backend._candidate_digest(overlay, *inputs)
    assert source_changed not in {first, source_tree_changed}

    inputs[0].write_bytes(inputs[0].read_bytes() + b"\n# identity change\n")
    state["candidate_lock_sha256"] = hashlib.sha256(inputs[0].read_bytes()).hexdigest()
    inputs[2].write_text(json.dumps(state), encoding="utf-8")
    lock_changed = backend._candidate_digest(overlay, *inputs)
    assert lock_changed not in {first, source_changed}

    runtime_lock = dependencies / "python-runtime-lock.toml"
    runtime_lock.write_bytes(runtime_lock.read_bytes() + b"\n# identity change\n")
    state["python_runtime_lock_sha256"] = hashlib.sha256(
        runtime_lock.read_bytes()
    ).hexdigest()
    inputs[2].write_text(json.dumps(state), encoding="utf-8")
    runtime_changed = backend._candidate_digest(overlay, *inputs)
    assert runtime_changed not in {first, source_changed, lock_changed}

    inputs[1].write_text("[patch.crates-io]\n# changed\n", encoding="utf-8")
    state["cargo_config_sha256"] = hashlib.sha256(inputs[1].read_bytes()).hexdigest()
    inputs[2].write_text(json.dumps(state), encoding="utf-8")
    assert backend._candidate_digest(overlay, *inputs) not in {
        first,
        source_changed,
        lock_changed,
        runtime_changed,
    }


@pytest.mark.parametrize(
    ("hook", "arguments", "expected", "with_sdk"),
    [
        ("build_wheel", ("wheel",), "delegated", True),
        ("build_sdist", ("sdist",), "delegated", False),
        ("get_requires_for_build_wheel", (), ["delegated"], False),
        ("get_requires_for_build_sdist", (), ["delegated"], False),
        ("prepare_metadata_for_build_wheel", ("metadata",), "delegated", False),
    ],
)
def test_retained_pep517_hooks_use_gate_overlay_and_clean_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hook: str,
    arguments: tuple[str, ...],
    expected: str | list[str],
    with_sdk: bool,
) -> None:
    # This test exercises the publishable hook contract independently of the
    # candidate mode used by the enclosing contributor source gate.
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "release")
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    target = tmp_path / "cargo-target"
    gates: list[str] = []
    sdk_stages: list[Path] = []
    selftest_stages: list[tuple[Path, str]] = []
    normalized_wheels: list[Path] = []
    injected_bins = (tmp_path / "injected-bin", tmp_path / "injected-tools")
    for directory in injected_bins:
        directory.mkdir()
    injected = {
        "CARGO": "/attacker/cargo",
        "CARGO_BUILD_TARGET": "attacker-target",
        "CPATH": "/opt/local/include",
        "DYLD_LIBRARY_PATH": "/opt/local/lib",
        "GIT_INDEX_FILE": "/attacker/index",
        "LIBRARY_PATH": "/opt/local/lib",
        "MATURIN_PEP517_ARGS": "--target attacker-target",
        "PYO3_PYTHON": "/attacker/python",
        "PYTHONPATH": "/attacker/pythonpath",
        "RUSTC_WRAPPER": "/attacker/wrapper",
        "RUSTFLAGS": "-Clink-arg=/attacker/library",
    }
    for name, value in injected.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(str(path) for path in injected_bins)
        + os.pathsep
        + os.environ["PATH"],
    )

    @contextlib.contextmanager
    def fake_overlay(mode: str):
        assert mode == "release"
        yield overlay, target

    def fake_sdk(root: Path, target_dir: Path) -> Path:
        assert root == overlay and target_dir == target
        assert not set(injected) & set(os.environ)
        assert not {str(path) for path in injected_bins} & set(
            os.environ["PATH"].split(os.pathsep)
        )
        assert "CARGO_ENCODED_RUSTFLAGS" in os.environ
        staging = tmp_path / "sdk"
        staging.mkdir()
        (staging / "metadata.json").write_text(
            json.dumps({"target": "aarch64-apple-darwin"}), encoding="utf-8"
        )
        sdk_stages.append(staging)
        return staging

    def delegated(*_args, **_kwargs):
        assert Path.cwd() == overlay
        assert not set(injected) & set(os.environ)
        assert not {str(path) for path in injected_bins} & set(
            os.environ["PATH"].split(os.pathsep)
        )
        remaps = os.environ["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
        assert len(remaps) == 4
        assert all(flag.startswith("--remap-path-prefix=") for flag in remaps)
        assert os.environ["CARGO_HOME"] == str(tmp_path / "cargo-home")
        assert os.environ["CARGO_TARGET_DIR"] == str(target)
        assert os.environ["PYAMPLICOL_BUILD_OVERLAY"] == str(overlay)
        if sys.platform == "darwin":
            assert os.environ["MACOSX_DEPLOYMENT_TARGET"] == "11.0"
        if with_sdk:
            assert os.environ["PYAMPLICOL_SDK_STAGING"] == str(tmp_path / "sdk")
        return ["delegated"] if hook.startswith("get_requires") else "delegated"

    monkeypatch.setattr(backend, "_overlay", fake_overlay)
    monkeypatch.setattr(backend, "_check_dependencies", gates.append)
    monkeypatch.setattr(backend, "_stage_packaged_examples", lambda path: None)
    monkeypatch.setattr(backend, "_stage_python_stub", lambda path: None)
    monkeypatch.setattr(backend, "_stage_runtime_resources", lambda path: None)
    monkeypatch.setattr(
        backend,
        "_stage_selftest_fixture",
        lambda path, target_name: selftest_stages.append((path, target_name)),
    )
    monkeypatch.setattr(backend, "build_sdk", fake_sdk)
    monkeypatch.setattr(backend, "_normalize_built_wheel", normalized_wheels.append)
    monkeypatch.setattr(backend.maturin, hook, delegated)

    result = getattr(backend, hook)(*arguments)
    assert result == expected
    assert gates == ["release"]
    assert bool(sdk_stages) is with_sdk
    assert selftest_stages == ([(overlay, "aarch64-apple-darwin")] if with_sdk else [])
    assert normalized_wheels == ([Path(arguments[0]) / "delegated"] if with_sdk else [])
    for name, value in injected.items():
        assert os.environ[name] == value


def test_backend_does_not_advertise_pep660_hooks() -> None:
    editable_hooks = (
        "build_editable",
        "get_requires_for_build_editable",
        "prepare_metadata_for_build_editable",
    )
    assert not [name for name in editable_hooks if hasattr(backend, name)]


def test_dependency_gate_subprocess_ignores_inherited_build_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CARGO", "/attacker/cargo")
    monkeypatch.setenv("PYTHONPATH", "/attacker/pythonpath")
    monkeypatch.setenv("RUSTFLAGS", "-Clink-arg=/attacker/library")
    observed: dict[str, object] = {}

    def fake_run(command, *, cwd, env, check):
        observed.update(command=command, cwd=cwd, env=env, check=check)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(backend.subprocess, "run", fake_run)
    backend._check_dependencies("candidate")
    command = observed["command"]
    environment = observed["env"]
    assert isinstance(command, list) and command[1] == "-I"
    assert command[-1] == "--candidate"
    assert isinstance(environment, dict)
    assert not {"CARGO", "PYTHONPATH", "RUSTFLAGS"} & set(environment)


def test_build_tool_path_retains_isolated_interpreter_bin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment_bin = tmp_path / "build-env" / "bin"
    environment_bin.mkdir(parents=True)
    interpreter = environment_bin / "python"
    interpreter.symlink_to(Path(sys.executable).resolve())
    tool_bin = tmp_path / "rust" / "bin"
    tool_bin.mkdir(parents=True)

    monkeypatch.setattr(backend.sys, "executable", str(interpreter))
    monkeypatch.setattr(
        backend.shutil,
        "which",
        lambda executable, *, path: str(tool_bin / executable),
    )

    result = backend._build_tool_path("").split(os.pathsep)

    assert str(environment_bin) in result


def test_pep517_backend_rejects_recursive_delegation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "overlay"
    overlay.mkdir()

    @contextlib.contextmanager
    def fake_overlay(_mode: str):
        yield overlay, tmp_path / "target"

    monkeypatch.setattr(backend, "_overlay", fake_overlay)
    monkeypatch.setattr(backend, "_check_dependencies", lambda _mode: None)
    monkeypatch.setattr(
        backend.maturin,
        "build_sdist",
        lambda *_args: backend.get_requires_for_build_sdist(),
    )
    with pytest.raises(RuntimeError, match="recursive PEP 517"):
        backend.build_sdist(str(tmp_path / "dist"))


def test_native_link_arguments_are_typed_and_allowlisted() -> None:
    macos = sdk._typed_link_arguments(
        ["-lgcc_s", "-lSystem", "-lc", "-framework", "Security"],
        "aarch64-apple-darwin",
    )
    assert "gcc_s" in macos["system_libraries"]
    assert macos["system_libraries"] == ["gcc_s", "System", "c"]
    assert macos["frameworks"] == ["Security"]

    linux = sdk._typed_link_arguments(
        ["-lgcc_s", "-lutil", "-lrt", "-lpthread", "-lm", "-ldl", "-lc"],
        "x86_64-unknown-linux-gnu",
    )
    assert linux["frameworks"] == []
    assert "gcc_s" in linux["system_libraries"]


def test_static_archive_byte_scan_rejects_python_markers(tmp_path: Path) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"archive rusticol_runtime_load native-static-libs")
    sdk._scan_archive(archive)

    archive.write_bytes(b"archive rusticol_runtime_load PyGILState_Ensure")
    with pytest.raises(RuntimeError, match="PyGILState"):
        sdk._scan_archive(archive)


@pytest.mark.parametrize(
    "symbol",
    ["_PyErr_SetString", "PyLong_FromLong", "_Py_Dealloc"],
)
def test_static_archive_symbol_scan_rejects_python_symbols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(sdk.shutil, "which", lambda _name: "/usr/bin/nm")
    monkeypatch.setattr(
        sdk.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["nm"], 0, stdout=f"                 U {symbol}\n", stderr=""
        ),
    )

    with pytest.raises(RuntimeError, match="undefined Python"):
        sdk._scan_archive_symbols(archive)


def test_static_archive_symbol_scan_accepts_non_python_symbols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(sdk.shutil, "which", lambda _name: "/usr/bin/nm")
    monkeypatch.setattr(
        sdk.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["nm"], 0, stdout="                 U _malloc\n", stderr=""
        ),
    )

    assert sdk._scan_archive_symbols(archive)


def test_static_archive_symbol_scan_defers_incompatible_llvm_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"archive")
    monkeypatch.delenv("LLVM_NM", raising=False)
    monkeypatch.setattr(sdk.shutil, "which", lambda _name: "/usr/bin/nm")
    monkeypatch.setattr(
        sdk.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["nm"],
            1,
            stdout="",
            stderr=(
                "Unknown attribute kind (91) "
                "(Producer: 'LLVM20.1.7' Reader: 'LLVM APPLE_1_1600')"
            ),
        ),
    )

    assert not sdk._scan_archive_symbols(archive)


@pytest.mark.skipif(os.name == "nt", reason="native SDK targets macOS and Linux")
def test_static_archive_link_probe_rejects_missing_public_export(
    tmp_path: Path,
) -> None:
    target = (
        "aarch64-apple-darwin"
        if sys.platform == "darwin"
        else "x86_64-unknown-linux-gnu"
    )
    try:
        compiler = sdk._native_c_compiler(target)
    except RuntimeError as error:
        pytest.skip(str(error))
    archiver = shutil.which("ar")
    if archiver is None:
        pytest.skip("ar is unavailable")

    header = tmp_path / "rusticol.h"
    header.write_text(
        """\
#include <stdint.h>
#define RUSTICOL_ABI_VERSION 1u
uint32_t rusticol_abi_version(void);
int rusticol_required_export(void);
""",
        encoding="utf-8",
    )
    implementation = tmp_path / "partial.c"
    implementation.write_text(
        "#include <stdint.h>\nuint32_t rusticol_abi_version(void) { return 1u; }\n",
        encoding="utf-8",
    )
    object_file = tmp_path / "partial.o"
    archive = tmp_path / "librusticol_capi.a"
    subprocess.run(
        [*compiler, "-c", str(implementation), "-o", str(object_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [archiver, "rcs", str(archive), str(object_file)],
        check=True,
        capture_output=True,
        text=True,
    )

    with pytest.raises(RuntimeError, match="complete C ABI link probe"):
        sdk._validate_archive_linkage(
            archive,
            header=header,
            link={"target": target, "system_libraries": [], "frameworks": []},
            target=target,
        )


@pytest.mark.parametrize(
    "token",
    [
        "-L/opt/local/lib",
        "-Wl,-rpath,/opt/local/lib",
        "/usr/local/lib/libfoo.a",
        "-lnot_allowlisted",
    ],
)
def test_native_link_arguments_reject_nonportable_tokens(token: str) -> None:
    with pytest.raises(RuntimeError):
        sdk._typed_link_arguments([token], "aarch64-apple-darwin")
