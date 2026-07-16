# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "release"))
_LOCK_BYTES = (ROOT / "dependencies" / "release-lock.toml").read_bytes()
_PYTHON_LOCK_BYTES = (ROOT / "dependencies" / "python-runtime-lock.toml").read_bytes()
_LOCK = tomllib.loads(_LOCK_BYTES.decode("utf-8"))
_DEFAULT_REQUIREMENTS = [
    f"{entry['distribution']}=={entry['version']}"
    for entry in _LOCK["python_dependencies"]
]
_LEGAL_FILES = tuple(
    tomllib.loads(
        (ROOT / "licenses" / "RUST_THIRD_PARTY.toml").read_text(encoding="utf-8")
    )["required_release_files"]
)

import _artifacts as artifacts  # noqa: E402
from _artifacts import (  # noqa: E402
    MAX_WHEEL_BYTES,
    RELEASE_TARGETS,
    ArtifactError,
    WheelReport,
    _canonical_sdist_members,
    _canonical_wheel_package_members,
    _scan_capi_archive,
    _scan_static_runtime_families,
    _target_cargo_inventory,
    audit_sdist,
    audit_wheel,
    compare_wheels,
    verify_manifest,
    write_manifest,
)


def _record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _cargo_version(version: str) -> str:
    return version.replace(".dev0+", "-dev.0+")


def _sbom_inventory(
    root_name: str, rust_target: str, version: str, *, candidate: bool
) -> set[tuple[str, str]]:
    inventory = set(_target_cargo_inventory(root_name, rust_target))
    if not candidate:
        return inventory
    first_party = {"rusticol-capi", "rusticol-core", "rusticol-python"}
    inventory = {
        (name, _cargo_version(version) if name in first_party else item_version)
        for name, item_version in inventory
    }
    candidate_symjit = _LOCK["symjit"]["candidate_version"]
    inventory = {
        (name, candidate_symjit if name == "symjit" else item_version)
        for name, item_version in inventory
    }
    return inventory


def _cyclonedx_sbom(
    root_name: str,
    rust_target: str,
    version: str,
    *,
    candidate: bool,
    include_target: bool,
) -> bytes:
    inventory = _sbom_inventory(root_name, rust_target, version, candidate=candidate)
    root_identity = (root_name, _cargo_version(version))
    assert root_identity in inventory

    def component(identity: tuple[str, str]) -> dict[str, str]:
        name, item_version = identity
        reference = f"pkg:cargo/{name}@{item_version}"
        return {
            "type": "library",
            "bom-ref": reference,
            "name": name,
            "version": item_version,
            "scope": "required",
            "purl": reference,
        }

    root_reference = f"pkg:cargo/{root_identity[0]}@{root_identity[1]}"
    children = sorted(inventory - {root_identity})
    metadata: dict[str, object] = {"component": component(root_identity)}
    if include_target:
        metadata["properties"] = [
            {"name": "pyamplicol:rust-target", "value": rust_target}
        ]
    document = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": metadata,
        "components": [component(identity) for identity in children],
        "dependencies": [
            {
                "ref": root_reference,
                "dependsOn": [
                    f"pkg:cargo/{name}@{item_version}"
                    for name, item_version in children
                ],
            },
            *[
                {
                    "ref": f"pkg:cargo/{name}@{item_version}",
                    "dependsOn": [],
                }
                for name, item_version in children
            ],
        ],
    }
    return (json.dumps(document, sort_keys=True) + "\n").encode()


def _distribution_cyclonedx_sbom(
    rust_target: str,
    version: str,
    *,
    candidate: bool,
    lock_bytes: bytes,
) -> bytes:
    cargo = json.loads(
        _cyclonedx_sbom(
            "rusticol-python",
            rust_target,
            version,
            candidate=candidate,
            include_target=False,
        )
    )
    cargo_metadata = cargo["metadata"]
    assert isinstance(cargo_metadata, dict)
    cargo_root = cargo_metadata["component"]
    assert isinstance(cargo_root, dict)
    runtime_lock = tomllib.loads(_PYTHON_LOCK_BYTES.decode("utf-8"))
    packages = runtime_lock["packages"]

    def license_record(value: str) -> dict[str, object]:
        if " " not in value or any(
            operator in value for operator in (" AND ", " OR ", " WITH ")
        ):
            return {"expression": value}
        return {"license": {"name": value}}

    python_components: list[dict[str, object]] = []
    python_edges: list[dict[str, object]] = []
    direct_references: list[str] = []
    for package in packages:
        name = artifacts.canonicalize_name(package["distribution"])
        reference = artifacts._pypi_purl(name, str(package["version"]))
        properties = [
            {
                "name": "pyamplicol:pypi:hashes-verified",
                "value": str(bool(package.get("artifacts", []))).lower(),
            },
            *[
                {
                    "name": "pyamplicol:pypi:artifact",
                    "value": json.dumps(
                        {
                            "filename": artifact["filename"],
                            "sha256": artifact["sha256"],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
                for artifact in package.get("artifacts", [])
            ],
        ]
        python_components.append(
            {
                "type": "library",
                "bom-ref": reference,
                "name": package["distribution"],
                "version": package["version"],
                "scope": "required",
                "purl": reference,
                "licenses": [license_record(package["license"])],
                "properties": properties,
            }
        )
        python_edges.append(
            {
                "ref": reference,
                "dependsOn": sorted(
                    artifacts._pypi_purl(
                        artifacts.canonicalize_name(dependency),
                        str(
                            next(
                                item["version"]
                                for item in packages
                                if artifacts.canonicalize_name(item["distribution"])
                                == artifacts.canonicalize_name(dependency)
                            )
                        ),
                    )
                    for dependency in package["dependencies"]
                ),
            }
        )
        if package["direct"]:
            direct_references.append(reference)

    root_reference = artifacts._pypi_purl("pyamplicol", version)
    document = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "library",
                "bom-ref": root_reference,
                "name": "pyamplicol",
                "version": version,
                "scope": "required",
                "purl": root_reference,
                "licenses": [{"expression": "0BSD"}],
            },
            "properties": [
                {
                    "name": "pyamplicol:build-mode",
                    "value": "candidate" if candidate else "release",
                },
                {
                    "name": "pyamplicol:release-lock:sha256",
                    "value": hashlib.sha256(lock_bytes).hexdigest(),
                },
                {
                    "name": "pyamplicol:python-runtime-lock:sha256",
                    "value": hashlib.sha256(_PYTHON_LOCK_BYTES).hexdigest(),
                },
            ],
        },
        "components": [
            cargo_root,
            *cargo["components"],
            *python_components,
        ],
        "dependencies": [
            *cargo["dependencies"],
            *python_edges,
            {
                "ref": root_reference,
                "dependsOn": sorted([cargo_root["bom-ref"], *direct_references]),
            },
        ],
    }
    return (json.dumps(document, sort_keys=True) + "\n").encode()


def _selftest_files(rust_target: str, version: str) -> dict[str, bytes]:
    payload_path = "processes/smoke/evaluator.symjit"
    payload = b"synthetic trusted SymJIT application"
    manifest = {
        "schema_version": 3,
        "artifact_id": "0" * 64,
        "producer": {
            "distribution": "pyamplicol",
            "version": version,
            "target": {"triple": rust_target, "cpu_features": []},
        },
        "runtime": {
            "engine": "rusticol",
            "engine_version": version,
            "required_runtime_capabilities": ["symjit.application.complex-f64.v1"],
        },
        "payloads": [
            {
                "path": payload_path,
                "role": "evaluator-state",
                "media_type": "application/vnd.symjit.application",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "target": {"triple": rust_target, "cpu_features": []},
            }
        ],
    }
    identity = dict(manifest)
    identity.pop("artifact_id")
    canonical = (
        json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    manifest["artifact_id"] = hashlib.sha256(canonical).hexdigest()
    prefix = f"pyamplicol/assets/selftest/{rust_target}"
    return {
        f"{prefix}/expected.json": json.dumps(
            {
                "schema_version": 1,
                "target": rust_target,
                "artifact_path": "artifact",
            }
        ).encode(),
        f"{prefix}/artifact/artifact.json": (
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode(),
        f"{prefix}/artifact/{payload_path}": payload,
    }


def _wheel(
    directory: Path,
    *,
    platform_tag: str = "manylinux_2_28_x86_64",
    rust_target: str = "x86_64-unknown-linux-gnu",
    version: str = "0.1.0",
    candidate: bool = False,
    extension: bytes = b"synthetic abi3 extension",
    reverse: bool = False,
    bad_record: bool = False,
    requirement: str | None = None,
    requirements: list[str] | None = None,
    system_libraries: list[str] | None = None,
    sdk_version: str | None = None,
    notice: bytes | None = b"third-party notices\n",
    include_lock: bool = True,
    lock_bytes: bytes = _LOCK_BYTES,
    license_files: tuple[str, ...] | None = None,
    omitted_member: str | None = None,
    extra_files: dict[str, bytes] | None = None,
) -> Path:
    if requirement is not None and requirements is not None:
        raise ValueError("use requirement or requirements, not both")
    runtime_requirements = (
        [requirement]
        if requirement is not None
        else list(_DEFAULT_REQUIREMENTS if requirements is None else requirements)
    )
    declared_legal_files = _LEGAL_FILES if license_files is None else license_files
    dist_info = f"pyamplicol-{version}.dist-info"
    sdk_archive = b"synthetic static archive"
    sdk_sbom = _cyclonedx_sbom(
        "rusticol-capi",
        rust_target,
        version,
        candidate=candidate,
        include_target=True,
    )
    distribution_sbom = _distribution_cyclonedx_sbom(
        rust_target,
        version,
        candidate=candidate,
        lock_bytes=lock_bytes,
    )
    files = {
        name: b"synthetic canonical wheel member\n"
        for name in _canonical_wheel_package_members()
    }
    files.update(
        {
            "pyamplicol/__init__.py": b"",
            "pyamplicol/_examples/README.md": b"load ../models/example.json\n",
            "pyamplicol/_rusticol.abi3.so": extension,
            "pyamplicol/_sdk/config.py": b"",
            "pyamplicol/_sdk/include/rusticol.h": b"void rusticol(void);\n",
            "pyamplicol/_sdk/include/rusticol.hpp": b"#pragma once\n",
            "pyamplicol/_sdk/fortran/rusticol.f90": b"module rusticol\nend module\n",
            "pyamplicol/_sdk/lib/librusticol_capi.a": sdk_archive,
            "pyamplicol/_sdk/metadata.json": json.dumps(
                {
                    "schema_version": 1,
                    "abi_version": 1,
                    "version": sdk_version or version,
                    "target": rust_target,
                    "archive": "lib/librusticol_capi.a",
                    "archive_sha256": hashlib.sha256(sdk_archive).hexdigest(),
                    "sbom": "sboms/rusticol-capi.cyclonedx.json",
                    "sbom_sha256": hashlib.sha256(sdk_sbom).hexdigest(),
                }
            ).encode(),
            "pyamplicol/_sdk/link.json": json.dumps(
                {
                    "schema_version": 1,
                    "target": rust_target,
                    "system_libraries": system_libraries or ["c", "m"],
                    "frameworks": [],
                }
            ).encode(),
            "pyamplicol/_sdk/sboms/rusticol-capi.cyclonedx.json": sdk_sbom,
            f"{dist_info}/METADATA": (
                "Metadata-Version: 2.4\n"
                "Name: pyamplicol\n"
                f"Version: {version}\n"
                "Requires-Python: >=3.11\n"
                "License-Expression: 0BSD\n"
                + "".join(f"License-File: {item}\n" for item in declared_legal_files)
                + "".join(f"Requires-Dist: {item}\n" for item in runtime_requirements)
                + "\n"
            ).encode(),
            f"{dist_info}/WHEEL": (
                "Wheel-Version: 1.0\n"
                "Generator: synthetic\n"
                "Root-Is-Purelib: false\n"
                f"Tag: cp311-abi3-{platform_tag}\n\n"
            ).encode(),
            f"{dist_info}/entry_points.txt": (
                b"[console_scripts]\n"
                b"pyamplicol = pyamplicol.cli:main\n"
                b"rusticol-config = pyamplicol._sdk.config:main\n"
            ),
            f"{dist_info}/sboms/rusticol-python.cyclonedx.json": distribution_sbom,
        }
    )
    files.update(_selftest_files(rust_target, version))
    if include_lock:
        files["pyamplicol/assets/release/release-lock.toml"] = lock_bytes
        files["pyamplicol/assets/release/python-runtime-lock.toml"] = _PYTHON_LOCK_BYTES
    else:
        files.pop("pyamplicol/assets/release/release-lock.toml", None)
    for relative in _LEGAL_FILES:
        if relative == "THIRD_PARTY_NOTICES.md" and notice is None:
            continue
        data = (
            notice
            if relative == "THIRD_PARTY_NOTICES.md"
            else (ROOT / relative).read_bytes()
        )
        assert data is not None
        files[f"{dist_info}/licenses/{relative}"] = data
    for name in (
        "README.md",
        "artifact-manifest-v3.schema.json",
        "runtime-physics-v1.schema.json",
    ):
        files[f"pyamplicol/assets/schemas/{name}"] = (
            ROOT / "schemas" / name
        ).read_bytes()
    if candidate:
        files["pyamplicol/_build_info.json"] = json.dumps(
            {"schema_version": 1, "publishable": False, "version": version}
        ).encode()
    if extra_files is not None:
        files.update(extra_files)
    if omitted_member is not None:
        files.pop(omitted_member)
    record_name = f"{dist_info}/RECORD"
    rows = [
        [name, _record_hash(data), str(len(data))]
        for name, data in sorted(files.items())
    ]
    rows.append([record_name, "", ""])
    if bad_record:
        rows[0][1] = "sha256=wrong"
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(rows)
    files[record_name] = record.getvalue().encode()

    filename = f"pyamplicol-{version}-cp311-abi3-{platform_tag}.whl"
    path = directory / filename
    items = list(files.items())
    if reverse:
        items.reverse()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in items:
            info = zipfile.ZipInfo(
                name, date_time=(2024 if reverse else 2025, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return path


def _rewrite_wheel(path: Path, transform: Callable[[dict[str, bytes]], None]) -> None:
    with zipfile.ZipFile(path) as archive:
        files = {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/")
        }
    transform(files)
    sdk_sbom = "pyamplicol/_sdk/sboms/rusticol-capi.cyclonedx.json"
    sdk_metadata = "pyamplicol/_sdk/metadata.json"
    if sdk_sbom in files and sdk_metadata in files:
        metadata = json.loads(files[sdk_metadata])
        metadata["sbom_sha256"] = hashlib.sha256(files[sdk_sbom]).hexdigest()
        files[sdk_metadata] = json.dumps(metadata, sort_keys=True).encode()
    record_name = next(name for name in files if name.endswith(".dist-info/RECORD"))
    rows = [
        [name, _record_hash(data), str(len(data))]
        for name, data in sorted(files.items())
        if name != record_name
    ]
    rows.append([record_name, "", ""])
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(rows)
    files[record_name] = record.getvalue().encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)


def _mutate_sbom(
    path: Path,
    mutate: Callable[[dict[str, object]], None],
    *,
    sdk: bool = True,
) -> None:
    def transform(files: dict[str, bytes]) -> None:
        name = (
            "pyamplicol/_sdk/sboms/rusticol-capi.cyclonedx.json"
            if sdk
            else next(
                item
                for item in files
                if ".dist-info/sboms/" in item and item.endswith(".cyclonedx.json")
            )
        )
        document = json.loads(files[name])
        mutate(document)
        files[name] = (json.dumps(document, sort_keys=True) + "\n").encode()

    _rewrite_wheel(path, transform)


def _sdist(
    directory: Path,
    *,
    version: str = "0.1.0",
    candidate: bool = False,
    manifest_path: str | None = None,
    project_requirement: str | None = None,
    omitted_member: str | None = None,
    extra_files: dict[str, bytes] | None = None,
) -> Path:
    python_version = version.replace("-dev.0", ".dev0")
    root = f"pyamplicol-{python_version}"
    files = {
        name: b"synthetic canonical sdist member\n"
        for name in _canonical_sdist_members()
    }
    files.update(
        {
            "Cargo.lock": b"version = 4\n",
            "Cargo.toml": f'[workspace.package]\nversion = "{version}"\n'.encode(),
            "LICENSE": b"0BSD\n",
            "README.md": b"pyamplicol\n",
            "THIRD_PARTY_NOTICES.md": b"notices\n",
            "build_backend/_pyamplicol_build.py": b"",
            "dependencies/python-runtime-lock.toml": _PYTHON_LOCK_BYTES,
            "dependencies/release-lock.toml": _LOCK_BYTES,
            "pyproject.toml": (
                '[project]\nname = "pyamplicol"\n'
                + (
                    f"dependencies = [{json.dumps(project_requirement)}]\n"
                    if project_requirement is not None
                    else ""
                )
            ).encode(),
            "tools/release/_artifacts.py": b"",
            "tools/release/_common.py": b"",
            "tools/release/build_from_sdist.py": b"",
            "tools/release/build_release_artifacts.py": b"",
            "tools/release/check_dependencies.py": b"",
            "tools/release/install_wheel.py": b"",
            "tools/release/publish_dry_run.py": b"",
            "tools/release/test_deployment.py": b"",
        }
    )
    if candidate:
        files["src/pyamplicol/_build_info.json"] = json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "version": python_version,
            }
        ).encode()
    if manifest_path is not None:
        files["docs/development/PORT_MANIFEST.toml"] = (
            f'repository = "{manifest_path}"\n'.encode()
        )
    if extra_files is not None:
        files.update(extra_files)
    if omitted_member is not None:
        files.pop(omitted_member)
    path = directory / f"pyamplicol-{python_version}.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def test_release_and_candidate_wheels_are_distinct_and_audited(
    tmp_path: Path,
) -> None:
    release = _wheel(tmp_path)
    report = audit_wheel(release, mode="release", native_scan=False)
    assert report.target == "manylinux_2_28_x86_64"
    assert report.abi_tag == "abi3"

    candidate_version = "0.1.0.dev0+candidate.0123456789ab"
    candidate = _wheel(
        tmp_path,
        version=candidate_version,
        candidate=True,
    )
    candidate_report = audit_wheel(candidate, mode="candidate", native_scan=False)
    assert candidate_report.version == candidate_version
    with pytest.raises(ArtifactError, match="release wheel"):
        audit_wheel(candidate, mode="release", native_scan=False)


def test_candidate_allows_only_rustup_standard_library_source_paths(
    tmp_path: Path,
) -> None:
    candidate_version = "0.1.0.dev0+candidate.0123456789ab"
    rustup_location = (
        b"/Users/developer/.rustup/toolchains/stable-aarch64-apple-darwin/"
        b"lib/rustlib/src/rust/library/core/src/slice/index.rs"
    )
    candidate = _wheel(
        tmp_path,
        version=candidate_version,
        candidate=True,
        extension=rustup_location,
    )
    assert audit_wheel(candidate, mode="candidate", native_scan=False).version == (
        candidate_version
    )

    release = _wheel(tmp_path, extension=rustup_location)
    with pytest.raises(ArtifactError, match="non-relocatable path marker"):
        audit_wheel(release, mode="release", native_scan=False)

    other_local_path = _wheel(
        tmp_path,
        version="0.1.0.dev0+candidate.abcdef012345",
        candidate=True,
        extension=b"/Users/developer/project/private.rs",
    )
    with pytest.raises(ArtifactError, match="non-relocatable path marker"):
        audit_wheel(other_local_path, mode="candidate", native_scan=False)


def test_release_allows_rust_standard_library_backtrace_source_paths(
    tmp_path: Path,
) -> None:
    standard_library_location = (
        b"/rustc/59807616e1fa2540724bfbac14d7976d7e4a3860/library/std/src/"
        b"../../backtrace/src/symbolize/gimli.rs"
    )
    wheel = _wheel(tmp_path, extension=standard_library_location)
    assert audit_wheel(wheel, mode="release", native_scan=False).version == "0.1.0"


def test_release_allows_remapped_rust_standard_library_backtrace_paths(
    tmp_path: Path,
) -> None:
    remapped_standard_library_location = (
        b"library/std/src/../../backtrace/src/symbolize/gimli.rs"
    )
    wheel = _wheel(tmp_path, extension=remapped_standard_library_location)
    assert audit_wheel(wheel, mode="release", native_scan=False).version == "0.1.0"


@pytest.mark.parametrize("marker", [b"__gmpz_init", b"mpfr_set", b"malachite-q"])
def test_native_scan_rejects_static_arithmetic_runtime_markers(
    tmp_path: Path,
    marker: bytes,
) -> None:
    binary = tmp_path / "native-artifact"
    binary.write_bytes(b"prefix\x00" + marker + b"\x00suffix")
    with pytest.raises(ArtifactError, match="forbidden static arithmetic runtime"):
        _scan_static_runtime_families(binary, "test artifact")


@pytest.mark.parametrize(
    "symbol",
    ["_PyErr_SetString", "PyLong_FromLong", "_Py_Dealloc"],
)
def test_capi_archive_scan_rejects_undefined_python_symbols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"synthetic archive")
    monkeypatch.delenv("LLVM_NM", raising=False)
    monkeypatch.setattr(artifacts.shutil, "which", lambda _name: "/usr/bin/nm")
    monkeypatch.setattr(
        artifacts.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["nm"], 0, stdout=f"                 U {symbol}\n", stderr=""
        ),
    )

    with pytest.raises(ArtifactError, match="undefined Python"):
        _scan_capi_archive(archive, allow_local_rustup=False)


def test_capi_archive_scan_defers_incompatible_llvm_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"synthetic archive")
    monkeypatch.delenv("LLVM_NM", raising=False)
    monkeypatch.setattr(artifacts.shutil, "which", lambda _name: "/usr/bin/nm")
    monkeypatch.setattr(
        artifacts.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["nm"],
            1,
            stdout="",
            stderr="Unknown attribute kind (91) from LLVM 20",
        ),
    )

    assert not _scan_capi_archive(archive, allow_local_rustup=False)


def test_record_direct_dependency_and_size_fail_closed(tmp_path: Path) -> None:
    bad_record = _wheel(tmp_path, bad_record=True)
    with pytest.raises(ArtifactError, match="RECORD hash"):
        audit_wheel(bad_record, mode="release", native_scan=False)

    bad_record.unlink()
    direct = _wheel(tmp_path, requirement="numpy @ file:///tmp/numpy.whl")
    with pytest.raises(ArtifactError, match="direct URL or path"):
        audit_wheel(direct, mode="release", native_scan=False)

    direct.unlink()
    parent_reference = _wheel(tmp_path, extension=b"built from ../parent/source")
    with pytest.raises(ArtifactError, match="non-relocatable"):
        audit_wheel(parent_reference, mode="release", native_scan=False)

    oversized = tmp_path / "pyamplicol-0.1.0-cp311-abi3-manylinux_2_28_x86_64.whl"
    with oversized.open("wb") as stream:
        stream.truncate(MAX_WHEEL_BYTES + 1)
    with pytest.raises(ArtifactError, match="limit"):
        audit_wheel(oversized, mode="release", native_scan=False)


def test_wheel_filename_version_must_match_metadata(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    renamed = wheel.with_name(wheel.name.replace("-0.1.0-", "-9.9.9-"))
    wheel.rename(renamed)
    with pytest.raises(ArtifactError, match="filename version"):
        audit_wheel(renamed, mode="release", native_scan=False)


def test_candidate_pypi_purl_percent_encodes_local_version() -> None:
    assert (
        artifacts._pypi_purl("pyamplicol", "0.1.0.dev0+candidate.123456789abc")
        == "pkg:pypi/pyamplicol@0.1.0.dev0%2Bcandidate.123456789abc"
    )


def test_sdk_link_metadata_is_target_allowlisted(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, system_libraries=["python3.11"])
    with pytest.raises(ArtifactError, match="target-allowlisted"):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_sdk_sbom_is_required_and_hash_bound(tmp_path: Path) -> None:
    sbom = "pyamplicol/_sdk/sboms/rusticol-capi.cyclonedx.json"
    wheel = _wheel(tmp_path, omitted_member=sbom)
    with pytest.raises(ArtifactError, match=re.escape(sbom)):
        audit_wheel(wheel, mode="release", native_scan=False)

    wheel.unlink()
    wheel = _wheel(tmp_path, extra_files={sbom: b"{}\n"})
    with pytest.raises(ArtifactError, match="SBOM SHA-256"):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_wheel_requires_exactly_one_distribution_and_one_sdk_sbom(
    tmp_path: Path,
) -> None:
    dist_sbom = "pyamplicol-0.1.0.dist-info/sboms/rusticol-python.cyclonedx.json"
    missing = _wheel(tmp_path, omitted_member=dist_sbom)
    with pytest.raises(ArtifactError, match="exactly one Maturin distribution SBOM"):
        audit_wheel(missing, mode="release", native_scan=False)

    missing.unlink()
    extra = _wheel(
        tmp_path,
        extra_files={"pyamplicol-0.1.0.dist-info/sboms/extra.cyclonedx.json": b"{}\n"},
    )
    with pytest.raises(ArtifactError, match="exactly one Maturin distribution SBOM"):
        audit_wheel(extra, mode="release", native_scan=False)


def test_sdk_sbom_rejects_empty_and_duplicate_graph_entries(tmp_path: Path) -> None:
    empty = _wheel(tmp_path)

    def empty_components(document: dict[str, object]) -> None:
        document["components"] = []

    _mutate_sbom(empty, empty_components)
    with pytest.raises(ArtifactError, match="nonempty component list"):
        audit_wheel(empty, mode="release", native_scan=False)

    empty.unlink()
    empty_dependencies = _wheel(tmp_path)

    def empty_dependency_graph(document: dict[str, object]) -> None:
        document["dependencies"] = []

    _mutate_sbom(empty_dependencies, empty_dependency_graph)
    with pytest.raises(ArtifactError, match="nonempty dependency graph"):
        audit_wheel(empty_dependencies, mode="release", native_scan=False)

    empty_dependencies.unlink()
    duplicate_component = _wheel(tmp_path)

    def duplicate_first_component(document: dict[str, object]) -> None:
        components = document["components"]
        assert isinstance(components, list)
        components.append(dict(components[0]))

    _mutate_sbom(duplicate_component, duplicate_first_component)
    with pytest.raises(ArtifactError, match="repeats component"):
        audit_wheel(duplicate_component, mode="release", native_scan=False)

    duplicate_component.unlink()
    duplicate_dependency = _wheel(tmp_path)

    def duplicate_first_dependency(document: dict[str, object]) -> None:
        dependencies = document["dependencies"]
        assert isinstance(dependencies, list)
        dependencies.append(dict(dependencies[0]))

    _mutate_sbom(duplicate_dependency, duplicate_first_dependency)
    with pytest.raises(ArtifactError, match="repeats dependency ref"):
        audit_wheel(duplicate_dependency, mode="release", native_scan=False)


def test_sdk_sbom_rejects_dangling_and_missing_components(tmp_path: Path) -> None:
    dangling = _wheel(tmp_path)

    def add_dangling_reference(document: dict[str, object]) -> None:
        dependencies = document["dependencies"]
        assert isinstance(dependencies, list)
        root_dependency = dependencies[0]
        assert isinstance(root_dependency, dict)
        children = root_dependency["dependsOn"]
        assert isinstance(children, list)
        children.append("pkg:cargo/not-in-components@1.0.0")

    _mutate_sbom(dangling, add_dangling_reference)
    with pytest.raises(ArtifactError, match="dangling references"):
        audit_wheel(dangling, mode="release", native_scan=False)

    dangling.unlink()
    missing = _wheel(tmp_path)

    def remove_component_cleanly(document: dict[str, object]) -> None:
        components = document["components"]
        dependencies = document["dependencies"]
        assert isinstance(components, list) and isinstance(dependencies, list)
        removed = components.pop()
        assert isinstance(removed, dict)
        reference = removed["bom-ref"]
        document["dependencies"] = [
            dependency
            for dependency in dependencies
            if isinstance(dependency, dict) and dependency.get("ref") != reference
        ]
        root_dependency = document["dependencies"][0]
        assert isinstance(root_dependency, dict)
        children = root_dependency["dependsOn"]
        assert isinstance(children, list)
        root_dependency["dependsOn"] = [
            child for child in children if child != reference
        ]

    _mutate_sbom(missing, remove_component_cleanly)
    with pytest.raises(ArtifactError, match="target Cargo closure"):
        audit_wheel(missing, mode="release", native_scan=False)


def test_sdk_sbom_rejects_cargo_legal_inventory_mismatch(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)

    def replace_with_undeclared_component(document: dict[str, object]) -> None:
        components = document["components"]
        dependencies = document["dependencies"]
        assert isinstance(components, list) and isinstance(dependencies, list)
        component = components[0]
        assert isinstance(component, dict)
        old_reference = component["bom-ref"]
        new_reference = "pkg:cargo/undeclared-secret@9.9.9"
        component.update(
            {
                "bom-ref": new_reference,
                "name": "undeclared-secret",
                "version": "9.9.9",
                "purl": new_reference,
            }
        )
        for dependency in dependencies:
            assert isinstance(dependency, dict)
            if dependency.get("ref") == old_reference:
                dependency["ref"] = new_reference
            children = dependency.get("dependsOn", [])
            assert isinstance(children, list)
            dependency["dependsOn"] = [
                new_reference if child == old_reference else child for child in children
            ]

    _mutate_sbom(wheel, replace_with_undeclared_component)
    with pytest.raises(ArtifactError, match="Cargo legal inventory"):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_distribution_sbom_root_identity_is_exact(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)

    def corrupt_root_purl(document: dict[str, object]) -> None:
        metadata = document["metadata"]
        assert isinstance(metadata, dict)
        component = metadata["component"]
        assert isinstance(component, dict)
        component["purl"] = "pkg:cargo/not-rusticol-python@0.1.0"

    _mutate_sbom(wheel, corrupt_root_purl, sdk=False)
    with pytest.raises(ArtifactError, match="root has invalid component identity"):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_distribution_sbom_python_components_are_lock_bound(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)

    def corrupt_colorama(document: dict[str, object]) -> None:
        components = document["components"]
        assert isinstance(components, list)
        component = next(
            item
            for item in components
            if isinstance(item, dict)
            and item.get("bom-ref") == "pkg:pypi/colorama@0.4.6"
        )
        component["licenses"] = [{"expression": "MIT"}]

    _mutate_sbom(wheel, corrupt_colorama, sdk=False)
    with pytest.raises(ArtifactError, match="colorama disagrees with the lock"):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_distribution_sbom_python_artifact_hashes_are_lock_bound(
    tmp_path: Path,
) -> None:
    wheel = _wheel(tmp_path)

    def corrupt_colorama_hash(document: dict[str, object]) -> None:
        components = document["components"]
        assert isinstance(components, list)
        component = next(
            item
            for item in components
            if isinstance(item, dict)
            and item.get("bom-ref") == "pkg:pypi/colorama@0.4.6"
        )
        properties = component["properties"]
        assert isinstance(properties, list)
        artifact = next(
            item
            for item in properties
            if isinstance(item, dict) and item.get("name") == "pyamplicol:pypi:artifact"
        )
        payload = json.loads(str(artifact["value"]))
        payload["sha256"] = "0" * 64
        artifact["value"] = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    _mutate_sbom(wheel, corrupt_colorama_hash, sdk=False)
    with pytest.raises(ArtifactError, match="artifact records for colorama"):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_distribution_sbom_python_edges_are_lock_bound(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)

    def drop_ufo_loader_dependency(document: dict[str, object]) -> None:
        dependencies = document["dependencies"]
        assert isinstance(dependencies, list)
        record = next(
            item
            for item in dependencies
            if isinstance(item, dict)
            and item.get("ref") == "pkg:pypi/ufo-model-loader@0.1.7"
        )
        record["dependsOn"] = []

    _mutate_sbom(wheel, drop_ufo_loader_dependency, sdk=False)
    with pytest.raises(ArtifactError, match="edges for ufo-model-loader"):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_distribution_sbom_lock_hash_metadata_is_required(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)

    def corrupt_runtime_lock_hash(document: dict[str, object]) -> None:
        metadata = document["metadata"]
        assert isinstance(metadata, dict)
        properties = metadata["properties"]
        assert isinstance(properties, list)
        record = next(
            item
            for item in properties
            if isinstance(item, dict)
            and item.get("name") == "pyamplicol:python-runtime-lock:sha256"
        )
        record["value"] = "0" * 64

    _mutate_sbom(wheel, corrupt_runtime_lock_hash, sdk=False)
    with pytest.raises(ArtifactError, match="runtime-lock:sha256 is inconsistent"):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_runtime_requirements_must_agree_with_packaged_lock(tmp_path: Path) -> None:
    missing = _wheel(
        tmp_path,
        requirements=[
            item for item in _DEFAULT_REQUIREMENTS if not item.startswith("colorama")
        ],
    )
    with pytest.raises(ArtifactError, match="inventory disagrees"):
        audit_wheel(missing, mode="release", native_scan=False)

    missing.unlink()
    incompatible = _wheel(
        tmp_path,
        requirements=[
            "numpy<2" if item.startswith("numpy") else item
            for item in _DEFAULT_REQUIREMENTS
        ],
    )
    with pytest.raises(ArtifactError, match="excludes locked version"):
        audit_wheel(incompatible, mode="release", native_scan=False)

    incompatible.unlink()
    ordinary_range = _wheel(
        tmp_path,
        requirements=[
            "numpy>=1.26" if item.startswith("numpy") else item
            for item in _DEFAULT_REQUIREMENTS
        ],
    )
    with pytest.raises(ArtifactError, match=r"pin numpy==2\.4\.2 exactly"):
        audit_wheel(ordinary_range, mode="release", native_scan=False)

    ordinary_range.unlink()
    non_exact = _wheel(
        tmp_path,
        requirements=[
            "symbolica>=2.1.0,<3" if item.startswith("symbolica") else item
            for item in _DEFAULT_REQUIREMENTS
        ],
    )
    with pytest.raises(ArtifactError, match=r"pin symbolica==2\.1\.0 exactly"):
        audit_wheel(non_exact, mode="release", native_scan=False)

    non_exact.unlink()
    marker_string = _wheel(
        tmp_path,
        requirements=[
            *_DEFAULT_REQUIREMENTS,
            "unexpected==1; python_version == 'extra'",
        ],
    )
    with pytest.raises(ArtifactError, match="may not use environment markers"):
        audit_wheel(marker_string, mode="release", native_scan=False)

    marker_string.unlink()
    optional = _wheel(
        tmp_path,
        requirements=[*_DEFAULT_REQUIREMENTS, "build>=1; extra == 'test'"],
    )
    assert audit_wheel(optional, mode="release", native_scan=False).version == "0.1.0"


def test_wheel_requires_packaged_lock_and_nonempty_legal_members(
    tmp_path: Path,
) -> None:
    no_lock = _wheel(tmp_path, include_lock=False)
    with pytest.raises(ArtifactError, match="missing packaged dependency lock"):
        audit_wheel(no_lock, mode="release", native_scan=False)

    no_lock.unlink()
    changed_lock = _wheel(tmp_path, lock_bytes=_LOCK_BYTES + b"\n# changed\n")
    with pytest.raises(ArtifactError, match="differs from canonical lock"):
        audit_wheel(changed_lock, mode="release", native_scan=False)

    changed_lock.unlink()
    no_notice = _wheel(tmp_path, notice=None)
    with pytest.raises(ArtifactError, match="missing nonempty legal member"):
        audit_wheel(no_notice, mode="release", native_scan=False)

    no_notice.unlink()
    empty_notice = _wheel(tmp_path, notice=b"\n")
    with pytest.raises(ArtifactError, match="missing nonempty legal member"):
        audit_wheel(empty_notice, mode="release", native_scan=False)


def test_wheel_rejects_resources_installed_at_site_packages_root(
    tmp_path: Path,
) -> None:
    wheel = _wheel(tmp_path)
    with zipfile.ZipFile(wheel) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    files["schemas/artifact-manifest-v3.schema.json"] = b"{}\n"
    record_name = next(name for name in files if name.endswith(".dist-info/RECORD"))
    rows = [
        [name, _record_hash(data), str(len(data))]
        for name, data in sorted(files.items())
        if name != record_name
    ]
    rows.append([record_name, "", ""])
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(rows)
    files[record_name] = record.getvalue().encode()
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)

    with pytest.raises(ArtifactError, match="outside pyamplicol"):
        audit_wheel(wheel, mode="release", native_scan=False)


@pytest.mark.parametrize(
    "resource",
    [
        "pyamplicol/assets/schemas/README.md",
        "pyamplicol/assets/schemas/artifact-manifest-v3.schema.json",
        "pyamplicol/assets/schemas/runtime-physics-v1.schema.json",
    ],
)
def test_wheel_requires_all_package_owned_schema_resources(
    tmp_path: Path, resource: str
) -> None:
    wheel = _wheel(tmp_path, omitted_member=resource)
    with pytest.raises(ArtifactError, match=re.escape(resource)):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_wheel_rejects_an_unknown_top_level_root(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, extra_files={"docs/leak.txt": b"not wheel data\n"})
    with pytest.raises(ArtifactError, match=r"outside pyamplicol.*docs/leak\.txt"):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_wheel_rejects_unmanifested_package_and_repair_root_files(
    tmp_path: Path,
) -> None:
    package_secret = _wheel(
        tmp_path,
        extra_files={"pyamplicol/secrets/token.txt": b"not release data\n"},
    )
    with pytest.raises(ArtifactError, match=r"canonical package manifest.*token"):
        audit_wheel(package_secret, mode="release", native_scan=False)

    package_secret.unlink()
    repair_secret = _wheel(
        tmp_path,
        extra_files={"pyamplicol.libs/token.txt": b"not a repaired library\n"},
    )
    with pytest.raises(ArtifactError, match="non-library member"):
        audit_wheel(repair_secret, mode="release", native_scan=False)


def test_wheel_requires_exact_legal_inventory(tmp_path: Path) -> None:
    supplementary = "licenses/SymJIT.txt"
    dist_info = "pyamplicol-0.1.0.dist-info"
    wheel = _wheel(
        tmp_path,
        omitted_member=f"{dist_info}/licenses/{supplementary}",
    )
    with pytest.raises(ArtifactError, match=re.escape(supplementary)):
        audit_wheel(wheel, mode="release", native_scan=False)

    wheel.unlink()
    wheel = _wheel(
        tmp_path,
        license_files=tuple(item for item in _LEGAL_FILES if item != supplementary),
    )
    with pytest.raises(ArtifactError, match=r"License-File inventory.*SymJIT"):
        audit_wheel(wheel, mode="release", native_scan=False)

    wheel.unlink()
    wheel = _wheel(
        tmp_path,
        license_files=(*_LEGAL_FILES, "licenses/UNDECLARED.txt"),
    )
    with pytest.raises(ArtifactError, match=r"License-File inventory.*UNDECLARED"):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_candidate_sdk_version_must_match_staged_package_exactly(
    tmp_path: Path,
) -> None:
    version = "0.1.0.dev0+candidate.0123456789ab"
    wheel = _wheel(
        tmp_path,
        version=version,
        candidate=True,
        sdk_version="0.1.0",
    )
    with pytest.raises(ArtifactError, match="SDK version"):
        audit_wheel(wheel, mode="candidate", native_scan=False)


def test_normalized_comparison_ignores_zip_order_and_timestamps(
    tmp_path: Path,
) -> None:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    changed_dir = tmp_path / "changed"
    left_dir.mkdir()
    right_dir.mkdir()
    changed_dir.mkdir()
    left = _wheel(left_dir)
    right = _wheel(right_dir, reverse=True)
    compare_wheels(left, right)

    changed = _wheel(changed_dir, extension=b"different native payload")
    with pytest.raises(ArtifactError, match="payload differs"):
        compare_wheels(left, changed)


def test_sdist_candidate_identity_and_manifest_path_scan(tmp_path: Path) -> None:
    release = _sdist(tmp_path)
    assert audit_sdist(release, mode="release").version == "0.1.0"

    candidate = _sdist(
        tmp_path,
        version="0.1.0-dev.0+candidate.0123456789ab",
        candidate=True,
    )
    assert audit_sdist(candidate, mode="candidate").version.endswith("0123456789ab")

    release.unlink()
    leaked = _sdist(tmp_path, manifest_path="/Users/build/parent-project")
    with pytest.raises(ArtifactError, match="non-relocatable"):
        audit_sdist(leaked, mode="release")

    leaked.unlink()
    direct = _sdist(
        tmp_path,
        project_requirement="numpy @ https://example.invalid/numpy.whl",
    )
    with pytest.raises(ArtifactError, match="direct URL or path"):
        audit_sdist(direct, mode="release")


def test_sdist_rejects_unmanifested_source_files(tmp_path: Path) -> None:
    sdist = _sdist(
        tmp_path,
        extra_files={"secrets/token.txt": b"not source distribution data\n"},
    )
    with pytest.raises(ArtifactError, match=r"canonical source manifest.*token"):
        audit_sdist(sdist, mode="release")


@pytest.mark.parametrize(
    "missing_member",
    [
        "build_backend/_pyamplicol_build.py",
        "build_backend/sdk.py",
        "dependencies/patches/symjit/0004-fix-aarch64-long-conditional-branches.patch",
        "dependencies/patches/symbolica/0004-export-symjit-application.patch",
        "docs/user/installation.md",
        "examples/python/typed_generation.py",
        "rust/crates/rusticol-capi/include/rusticol.h",
        "rust/crates/rusticol-python/stubs/pyamplicol/_rusticol.pyi",
        "schemas/README.md",
        "schemas/artifact-manifest-v3.schema.json",
        "src/pyamplicol/assets/selftest/portable-64le/expected.json",
        "src/pyamplicol/assets/selftest/portable-64le/artifact/artifact.json",
        "tests/integration/test_examples.py",
        "tools/release/test_deployment.py",
    ],
)
def test_sdist_requires_nested_release_content(
    tmp_path: Path, missing_member: str
) -> None:
    sdist = _sdist(tmp_path, omitted_member=missing_member)
    with pytest.raises(ArtifactError, match=re.escape(missing_member)):
        audit_sdist(sdist, mode="release")


def _complete_release_wheels(
    directory: Path,
) -> tuple[list[Path], list[WheelReport]]:
    wheels: list[Path] = []
    reports: list[WheelReport] = []
    for platform_tag, rust_target in RELEASE_TARGETS.items():
        wheel = _wheel(
            directory,
            platform_tag=platform_tag,
            rust_target=rust_target,
        )
        wheels.append(wheel)
        reports.append(
            replace(
                audit_wheel(wheel, mode="release", native_scan=False),
                native_scan=True,
            )
        )
    return wheels, reports


def test_manifest_hashes_are_exact(tmp_path: Path) -> None:
    wheels, wheel_reports = _complete_release_wheels(tmp_path)
    sdist = _sdist(tmp_path)
    sdist_report = audit_sdist(sdist, mode="release")
    write_manifest(
        tmp_path,
        mode="release",
        wheels=wheel_reports,
        sdists=[sdist_report],
        parity="verified",
        retained_sdist=sdist_report,
        source_commit="1" * 40,
        source_tag="v0.1.0",
    )
    payload = verify_manifest(tmp_path, require_release=True, require_all_targets=False)
    assert payload["source"] == {"commit": "1" * 40, "tag": "v0.1.0"}
    assert {entry["filename"] for entry in payload["artifacts"]} == {
        *(wheel.name for wheel in wheels),
        sdist.name,
    }
    wheels[0].write_bytes(wheels[0].read_bytes() + b"tamper")
    with pytest.raises(ArtifactError, match="hash/size"):
        verify_manifest(tmp_path, require_release=True, require_all_targets=False)


def test_manifest_source_identity_is_atomic_and_exact(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="supplied together"):
        write_manifest(
            tmp_path,
            mode="candidate",
            wheels=[],
            sdists=[],
            parity="candidate-not-release-validated",
            source_commit="1" * 40,
        )
    with pytest.raises(ArtifactError, match="full lowercase Git SHA"):
        write_manifest(
            tmp_path,
            mode="candidate",
            wheels=[],
            sdists=[],
            parity="candidate-not-release-validated",
            source_commit="not-a-commit",
            source_tag="v0.1.0",
        )
    with pytest.raises(ArtifactError, match=r"must be v0\.1\.0"):
        write_manifest(
            tmp_path,
            mode="candidate",
            wheels=[],
            sdists=[],
            parity="candidate-not-release-validated",
            source_commit="1" * 40,
            source_tag="v0.1.1",
        )


def test_partial_release_manifest_is_explicitly_not_publishable(
    tmp_path: Path,
) -> None:
    sdist = _sdist(tmp_path)
    sdist_report = audit_sdist(sdist, mode="release")
    manifest = write_manifest(
        tmp_path,
        mode="release",
        wheels=[],
        sdists=[sdist_report],
        parity="sdist-only",
    )
    assert json.loads(manifest.read_text(encoding="utf-8"))["publishable"] is False
    with pytest.raises(ArtifactError, match="complete publishable release"):
        verify_manifest(tmp_path, require_release=True, require_all_targets=False)


def test_release_manifest_requires_all_targets_and_source_identity(
    tmp_path: Path,
) -> None:
    sdist = _sdist(tmp_path)
    sdist_report = audit_sdist(sdist, mode="release")
    one_wheel = _wheel(tmp_path)
    one_report = replace(
        audit_wheel(one_wheel, mode="release", native_scan=False), native_scan=True
    )
    partial = write_manifest(
        tmp_path,
        mode="release",
        wheels=[one_report],
        sdists=[sdist_report],
        parity="verified",
        retained_sdist=sdist_report,
        source_commit="1" * 40,
        source_tag="v0.1.0",
    )
    assert json.loads(partial.read_text(encoding="utf-8"))["publishable"] is False
    with pytest.raises(ArtifactError, match="complete publishable release"):
        verify_manifest(tmp_path, require_release=True, require_all_targets=False)
    forged = json.loads(partial.read_text(encoding="utf-8"))
    forged["publishable"] = True
    partial.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(
        ArtifactError, match="exactly one wheel for every release target"
    ):
        verify_manifest(tmp_path, require_release=True, require_all_targets=False)

    one_wheel.unlink()
    (tmp_path / "release-manifest.json").unlink()
    (tmp_path / "SHA256SUMS").unlink()
    _, reports = _complete_release_wheels(tmp_path)
    source_less = write_manifest(
        tmp_path,
        mode="release",
        wheels=reports,
        sdists=[sdist_report],
        parity="verified",
        retained_sdist=sdist_report,
    )
    assert json.loads(source_less.read_text(encoding="utf-8"))["publishable"] is False
    with pytest.raises(ArtifactError, match="complete publishable release"):
        verify_manifest(tmp_path, require_release=True, require_all_targets=False)
    forged = json.loads(source_less.read_text(encoding="utf-8"))
    forged["publishable"] = True
    source_less.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ArtifactError, match="no source identity"):
        verify_manifest(tmp_path, require_release=True, require_all_targets=False)
