# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

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


def _wheel(path: Path) -> None:
    version = "0.1.0.dev0+candidate.testsource"
    target = "aarch64-apple-darwin"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"pyamplicol-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: pyamplicol\nVersion: {version}\n\n",
        )
        archive.writestr("pyamplicol/_rusticol.abi3.so", b"extension")
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
        archive.writestr(
            "pyamplicol/_build_info.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "publishable": False,
                    "candidate_fingerprint": "testsource",
                    "version": version,
                }
            ),
        )
        archive.writestr("README.md", "must not be staged\n")
        archive.writestr("pyamplicol/_sdk/config.py", "must not overwrite source\n")


def test_source_runtime_stages_only_audited_generated_resources(
    tmp_path: Path,
) -> None:
    module = _module()
    wheel = tmp_path / "pyamplicol-test.whl"
    package = tmp_path / "src/pyamplicol"
    source_build_info = tmp_path / ".artifacts/source-runtime/_build_info.json"
    package.mkdir(parents=True)
    tracked_config = package / "_sdk/config.py"
    tracked_config.parent.mkdir()
    tracked_config.write_text("tracked\n", encoding="utf-8")
    _wheel(wheel)

    report = module.stage_runtime(
        wheel,
        source_package=package,
        source_build_info=source_build_info,
        mode="candidate",
        audit=False,
    )

    assert report["version"] == "0.1.0.dev0+candidate.testsource"
    assert (package / "_rusticol.abi3.so").read_bytes() == b"extension"
    assert (package / "_sdk/lib/librusticol_capi.a").read_bytes() == b"archive"
    assert (
        package / "assets/selftest/aarch64-apple-darwin/artifact/artifact.json"
    ).is_file()
    driver = (
        package
        / "assets/selftest/aarch64-apple-darwin/artifact/API/python/"
        / "check_standalone.py"
    )
    assert driver.stat().st_mode & 0o111 == 0o111
    assert tracked_config.read_text(encoding="utf-8") == "tracked\n"
    assert not (package / "README.md").exists()
    assert not (package / "_build_info.json").exists()
    assert json.loads(source_build_info.read_text(encoding="utf-8"))["version"] == (
        "0.1.0.dev0+candidate.testsource"
    )


def test_package_version_prefers_the_staged_source_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    build_info = tmp_path / "_build_info.json"
    build_info.write_text(
        json.dumps({"version": "0.1.0.dev0+candidate.current"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(versions, "_PACKAGE_BUILD_INFO_PATH", tmp_path / "missing")
    monkeypatch.setattr(versions, "_SOURCE_BUILD_INFO_PATH", build_info)

    assert versions.package_version() == "0.1.0.dev0+candidate.current"
