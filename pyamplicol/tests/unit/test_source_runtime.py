# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import zipfile
from pathlib import Path

import pytest

from pyamplicol._internal import versions

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "developer" / "prepare_source_runtime.py"


def _module():
    spec = importlib.util.spec_from_file_location("prepare_source_runtime", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_root(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    source = root / "rust/crates/example/src/lib.rs"
    source.parent.mkdir(parents=True)
    source.write_text("pub fn value() -> u32 { 1 }\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (root / "src/pyamplicol").mkdir(parents=True)
    return root


def _wheel(
    path: Path,
    *,
    native_build_inputs_sha256: str,
    target: str | None = None,
) -> None:
    if target is None:
        machine = platform.machine().lower()
        if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
            target = "aarch64-apple-darwin"
        elif sys.platform == "darwin" and machine in {"amd64", "x86_64"}:
            target = "x86_64-apple-darwin"
        else:
            target = "x86_64-unknown-linux-gnu"
    version = "0.1.0.dev0+candidate.testsource"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"pyamplicol-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: pyamplicol\nVersion: {version}\n\n",
        )
        archive.writestr("pyamplicol/_rusticol.abi3.so", b"extension")
        archive.writestr(
            "pyamplicol/_build_info.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "publishable": False,
                    "native_build_inputs_sha256": native_build_inputs_sha256,
                    "source_checkout": "/candidate/source",
                    "version": version,
                }
            ),
        )
        archive.writestr(
            "pyamplicol/_sdk/metadata.json",
            json.dumps({"target": target}),
        )
        archive.writestr("pyamplicol/_sdk/link.json", "{}\n")
        archive.writestr("pyamplicol/_sdk/lib/librusticol_capi.a", b"archive")
        archive.writestr("pyamplicol/_sdk/include/rusticol.h", "/* header */\n")
        archive.writestr("pyamplicol/_sdk/fortran/rusticol.f90", "module r\n")
        archive.writestr(
            f"pyamplicol/assets/selftest/{target}/expected.json",
            "{}\n",
        )
        archive.writestr(
            f"pyamplicol/assets/selftest/{target}/artifact/artifact.json",
            "{}\n",
        )
        driver = zipfile.ZipInfo(
            f"pyamplicol/assets/selftest/{target}/artifact/API/python/"
            "check_standalone.py"
        )
        driver.external_attr = 0o100755 << 16
        archive.writestr(driver, "#!/usr/bin/env python3\n")
        archive.writestr("README.md", "must not be staged\n")
        archive.writestr("pyamplicol/_sdk/config.py", "must not overwrite source\n")


def test_source_runtime_stages_one_attested_extension(tmp_path: Path) -> None:
    module = _module()
    source_root = _source_root(tmp_path)
    native_digest = module._native_build_inputs_digest(source_root)
    wheel = tmp_path / "pyamplicol-test.whl"
    _wheel(wheel, native_build_inputs_sha256=native_digest)
    package = source_root / "src/pyamplicol"
    build_info = source_root / ".artifacts/source-runtime/_build_info.json"
    stale_extension = package / "_rusticol.cpython-312-darwin.so"
    stale_extension.write_bytes(b"stale")
    tracked_config = package / "_sdk/config.py"
    tracked_config.parent.mkdir()
    tracked_config.write_text("tracked\n", encoding="utf-8")

    report = module.stage_runtime(
        wheel,
        source_package=package,
        source_build_info=build_info,
        source_root=source_root,
        mode="candidate",
        audit=False,
    )

    assert report["version"] == "0.1.0.dev0+candidate.testsource"
    extension = package / "_rusticol.abi3.so"
    assert extension.read_bytes() == b"extension"
    assert not stale_extension.exists()
    assert tracked_config.read_text(encoding="utf-8") == "tracked\n"
    payload = json.loads(build_info.read_text(encoding="utf-8"))
    assert payload["source_runtime"] == {
        "extension_name": extension.name,
        "extension_sha256": hashlib.sha256(b"extension").hexdigest(),
        "native_build_inputs_sha256": native_digest,
    }
    assert not (build_info.parent / ".staging").exists()


def test_source_runtime_rejects_an_old_candidate_wheel(tmp_path: Path) -> None:
    module = _module()
    source_root = _source_root(tmp_path)
    wheel = tmp_path / "pyamplicol-test.whl"
    _wheel(
        wheel,
        native_build_inputs_sha256=module._native_build_inputs_digest(source_root),
    )
    (source_root / "rust/crates/example/src/lib.rs").write_text(
        "pub fn value() -> u32 { 2 }\n",
        encoding="utf-8",
    )

    with pytest.raises(module.ReleaseError, match="different native sources"):
        module.stage_runtime(
            wheel,
            source_package=source_root / "src/pyamplicol",
            source_build_info=source_root / ".artifacts/runtime/_build_info.json",
            source_root=source_root,
            mode="candidate",
            audit=False,
        )


def test_source_runtime_verification_rejects_replaced_binary_and_source(
    tmp_path: Path,
) -> None:
    source_root = _source_root(tmp_path)
    package = source_root / "src/pyamplicol"
    extension = package / "_rusticol.abi3.so"
    extension.write_bytes(b"current")
    payload = {
        "source_runtime": {
            "extension_name": extension.name,
            "extension_sha256": hashlib.sha256(b"current").hexdigest(),
            "native_build_inputs_sha256": versions._native_build_inputs_digest(
                source_root
            ),
        }
    }

    versions._verify_source_runtime(
        payload,
        package_root=package,
        source_root=source_root,
    )
    extension.write_bytes(b"stale")
    with pytest.raises(RuntimeError, match="stale or was replaced"):
        versions._verify_source_runtime(
            payload,
            package_root=package,
            source_root=source_root,
        )

    extension.write_bytes(b"current")
    (source_root / "rust/crates/example/src/lib.rs").write_text(
        "pub fn value() -> u32 { 2 }\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="native build inputs changed"):
        versions._verify_source_runtime(
            payload,
            package_root=package,
            source_root=source_root,
        )


def test_installed_candidate_is_bound_to_its_checkout(tmp_path: Path) -> None:
    source_root = _source_root(tmp_path)
    payload = {
        "publishable": False,
        "source_checkout": str(source_root),
        "native_build_inputs_sha256": versions._native_build_inputs_digest(source_root),
    }

    versions._verify_candidate_install(payload)
    (source_root / "rust/crates/example/src/lib.rs").write_text(
        "pub fn value() -> u32 { 2 }\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="candidate wheel is stale"):
        versions._verify_candidate_install(payload)


def test_native_module_build_id_must_match_candidate_metadata(monkeypatch) -> None:
    build_info = {
        "publishable": False,
        "native_build_inputs_sha256": "a" * 64,
    }
    monkeypatch.setattr(versions, "_active_build_info", lambda: build_info)
    native = type(
        "Native",
        (),
        {
            "__file__": "/candidate/pyamplicol/_rusticol.so",
            "package_version": staticmethod(lambda: "0.1.0-dev.0+candidate.same"),
            "native_build_inputs_sha256": staticmethod(lambda: "a" * 64),
        },
    )()
    versions.verify_native_module(
        native,
        expected_version="0.1.0.dev0+candidate.same",
    )

    build_info["native_build_inputs_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="different source inputs"):
        versions.verify_native_module(
            native,
            expected_version="0.1.0.dev0+candidate.same",
        )


def test_publishable_native_module_does_not_require_developer_build_id(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        versions,
        "_active_build_info",
        lambda: {"publishable": True, "version": "0.1.0"},
    )
    native = type(
        "Native",
        (),
        {
            "__file__": "/wheel/pyamplicol/_rusticol.so",
            "package_version": staticmethod(lambda: "0.1.0"),
        },
    )()

    versions.verify_native_module(native, expected_version="0.1.0")


def test_package_version_prefers_attested_source_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = _source_root(tmp_path)
    package = source_root / "src/pyamplicol"
    extension = package / "_rusticol.abi3.so"
    extension.write_bytes(b"extension")
    runtime = source_root / ".artifacts/source-runtime"
    runtime.mkdir(parents=True)
    build_info = runtime / "_build_info.json"
    build_info.write_text(
        json.dumps(
            {
                "publishable": False,
                "native_build_inputs_sha256": versions._native_build_inputs_digest(
                    source_root
                ),
                "source_runtime": {
                    "extension_name": extension.name,
                    "extension_sha256": hashlib.sha256(b"extension").hexdigest(),
                    "native_build_inputs_sha256": (
                        versions._native_build_inputs_digest(source_root)
                    ),
                },
                "version": "0.1.0.dev0+candidate.current",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(versions, "_SOURCE_ROOT", source_root)
    monkeypatch.setattr(versions, "_SOURCE_PACKAGE_ROOT", package)
    monkeypatch.setattr(versions, "_SOURCE_BUILD_INFO_PATH", build_info)
    monkeypatch.setattr(versions, "_SOURCE_RUNTIME_STAGING_PATH", runtime / ".staging")
    monkeypatch.setattr(versions, "_PACKAGE_BUILD_INFO_PATH", tmp_path / "missing")

    assert versions.package_version() == "0.1.0.dev0+candidate.current"


def test_package_version_fails_closed_during_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = _source_root(tmp_path)
    runtime = source_root / ".artifacts/source-runtime"
    runtime.mkdir(parents=True)
    marker = runtime / ".staging"
    marker.write_text("incomplete\n", encoding="utf-8")
    monkeypatch.setattr(versions, "_SOURCE_ROOT", source_root)
    monkeypatch.setattr(
        versions,
        "_SOURCE_PACKAGE_ROOT",
        source_root / "src/pyamplicol",
    )
    monkeypatch.setattr(versions, "_SOURCE_BUILD_INFO_PATH", runtime / "missing")
    monkeypatch.setattr(versions, "_SOURCE_RUNTIME_STAGING_PATH", marker)

    with pytest.raises(RuntimeError, match="staging is incomplete"):
        versions.package_version()


def test_source_runtime_rejects_dot_target(tmp_path: Path) -> None:
    module = _module()
    source_root = _source_root(tmp_path)
    wheel = tmp_path / "pyamplicol-test.whl"
    _wheel(
        wheel,
        native_build_inputs_sha256=module._native_build_inputs_digest(source_root),
        target="..",
    )

    with pytest.raises(module.ReleaseError, match="unsafe target"):
        module.stage_runtime(
            wheel,
            source_package=source_root / "src/pyamplicol",
            source_build_info=source_root / ".artifacts/runtime/_build_info.json",
            source_root=source_root,
            mode="candidate",
            audit=False,
        )


def test_source_runtime_rejects_a_wheel_for_another_host(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    source_root = _source_root(tmp_path)
    wheel = tmp_path / "pyamplicol-test.whl"
    _wheel(
        wheel,
        native_build_inputs_sha256=module._native_build_inputs_digest(source_root),
        target="x86_64-unknown-linux-gnu",
    )
    monkeypatch.setattr(module, "_host_target", lambda: "aarch64-apple-darwin")

    with pytest.raises(module.ReleaseError, match="this host requires"):
        module.stage_runtime(
            wheel,
            source_package=source_root / "src/pyamplicol",
            source_build_info=source_root / ".artifacts/runtime/_build_info.json",
            source_root=source_root,
            mode="candidate",
            audit=False,
        )


def test_source_runtime_stages_an_existing_local_wheel_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    wheel_directory = tmp_path / "wheelhouse"
    wheel_directory.mkdir()
    wheel = wheel_directory / "pyamplicol-test.whl"
    wheel.write_bytes(b"wheel")
    observed: dict[str, object] = {}

    def stage(path: Path, *, mode: str, audit: bool) -> dict[str, object]:
        observed.update(path=path, mode=mode, audit=audit)
        return {"wheel": path.name}

    monkeypatch.setattr(module, "stage_runtime", stage)
    report = module.stage_from_directory(wheel_directory, mode="candidate")

    assert report == {"wheel": wheel.name}
    assert observed == {"path": wheel, "mode": "candidate", "audit": False}


def test_native_provenance_digest_implementations_match(tmp_path: Path) -> None:
    module = _module()
    spec = importlib.util.spec_from_file_location(
        "source_runtime_build_backend",
        ROOT / "build_backend/_pyamplicol_build.py",
    )
    assert spec is not None and spec.loader is not None
    backend = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "build_backend"))
    try:
        sys.modules[spec.name] = backend
        spec.loader.exec_module(backend)
    finally:
        sys.path.pop(0)
    source_root = _source_root(tmp_path)

    expected = module._native_build_inputs_digest(source_root)
    assert backend._native_build_inputs_digest(source_root) == expected
    assert versions._native_build_inputs_digest(source_root) == expected
