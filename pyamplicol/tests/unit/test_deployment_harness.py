# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "release"))

import test_deployment as deployment  # noqa: E402


def test_installed_smoke_uses_the_v1_builtin_model_name() -> None:
    assert "from pyamplicol.models import BuiltinSMModel" in deployment._INSTALLED_SMOKE
    assert "AmplicolSMLeadingColorModel" not in deployment._INSTALLED_SMOKE


def test_f64_deployment_smoke_hard_blocks_symbolica() -> None:
    smoke = deployment._SYMBOLICA_FREE_F64_SMOKE
    assert 'os.environ.pop("SYMBOLICA_LICENSE", None)' in smoke
    assert "builtins.__import__ = reject_symbolica" in smoke
    assert 'check.name == "physics-f64"' in smoke
    assert 'name.startswith("symbolica.")' in smoke
    assert "for name in sys.modules" in smoke


from _common import ReleaseError, clean_environment  # noqa: E402


def test_release_environment_discards_native_toolchain_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    injected = {
        "CPATH": "/opt/local/include",
        "DYLD_LIBRARY_PATH": "/opt/local/lib",
        "LDFLAGS": "-L/opt/local/lib",
        "LIBRARY_PATH": "/opt/local/lib",
        "RUSTFLAGS": "-L/opt/local/lib/libgcc -l dylib=gcc_s",
    }
    for name, value in injected.items():
        monkeypatch.setenv(name, value)

    environment = clean_environment(mode="candidate")

    assert not set(injected) & set(environment)
    assert environment["PYAMPLICOL_BUILD_MODE"] == "candidate"


def test_deployment_copies_the_complete_installed_examples_tree(tmp_path: Path) -> None:
    python = tmp_path / "venv/bin/python"
    destination = tmp_path / "examples"

    command = [
        os.fspath(item)
        for item in deployment._examples_copy_command(python, destination)
    ]

    assert command == [
        os.fspath(python),
        "-I",
        "-m",
        "pyamplicol",
        "examples",
        "copy",
        os.fspath(destination),
    ]


def test_release_native_sdk_smoke_requires_both_compilers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("FC", raising=False)
    monkeypatch.setattr(deployment.shutil, "which", lambda _name: None)

    with pytest.raises(ReleaseError, match=r"C\+\+17 and Fortran 2008"):
        deployment._native_sdk_smoke(
            Path(sys.executable),
            sandbox=tmp_path,
            mode="release",
            environment={},
        )


def test_candidate_native_sdk_smoke_reports_explicit_nonvalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("FC", raising=False)
    monkeypatch.setattr(deployment.shutil, "which", lambda _name: None)

    assert (
        deployment._native_sdk_smoke(
            Path(sys.executable),
            sandbox=tmp_path,
            mode="candidate",
            environment={},
        )
        is False
    )
    assert "did not run native SDK smoke tests" in capsys.readouterr().out


def test_native_sdk_smoke_compiles_and_runs_cpp_and_fortran(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    include = tmp_path / "sdk/include"
    include.mkdir(parents=True)
    library = tmp_path / "sdk/librusticol_capi.a"
    library.write_bytes(b"archive")
    fortran_module = tmp_path / "sdk/rusticol.f90"
    fortran_module.write_text(
        "module rusticol\nend module rusticol\n", encoding="utf-8"
    )
    sdk = {
        "cflags": [f"-I{include}"],
        "link_flags": [str(library)],
        "fortran_source": str(fortran_module),
    }
    artifact = tmp_path / "installed/artifact"
    cpp_driver = artifact / "API/cpp/check_standalone.cpp"
    fortran_driver = artifact / "API/fortran/check_standalone.f90"
    cpp_driver.parent.mkdir(parents=True)
    fortran_driver.parent.mkdir(parents=True)
    cpp_driver.write_text("int main() { return 0; }\n", encoding="utf-8")
    fortran_driver.write_text("program main\nend program main\n", encoding="utf-8")
    fixture = deployment.NativePhysicsFixture(
        artifact=artifact,
        process_id="smoke",
        shape=(1, 1, 1),
        totals=(2.5,),
    )
    commands: list[list[str]] = []

    monkeypatch.setattr(
        deployment,
        "_native_toolchain",
        lambda _mode: deployment.NativeToolchain(
            cxx=("/tool/c++",), fortran=("/tool/gfortran",)
        ),
    )
    monkeypatch.setattr(
        deployment,
        "_installed_selftest_fixture",
        lambda *_args, **_kwargs: fixture,
    )

    def fake_run(command, **_kwargs):
        rendered = [os.fspath(item) for item in command]
        commands.append(rendered)
        if "pyamplicol._sdk.config" in rendered:
            return subprocess.CompletedProcess(rendered, 0, json.dumps(sdk), "")
        for language in ("cpp", "fortran"):
            if rendered[0].endswith(f"check_standalone_{language}"):
                payload = {
                    "language": language,
                    "available": True,
                    "precision": 16,
                    "process_key": "smoke",
                    "shape": [1, 1, 1],
                    "values": [2.5],
                    "resolved_sum": [2.5],
                    "compatibility_total": [2.5],
                }
                return subprocess.CompletedProcess(rendered, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(rendered, 0, "", "")

    monkeypatch.setattr(deployment, "run", fake_run)
    assert (
        deployment._native_sdk_smoke(
            Path(sys.executable),
            sandbox=tmp_path,
            mode="release",
            environment={},
        )
        is True
    )

    assert any(command[0] == "/tool/c++" for command in commands)
    assert any(command[0] == "/tool/gfortran" for command in commands)
    assert any(command[0].endswith("check_standalone_cpp") for command in commands)
    assert any(command[0].endswith("check_standalone_fortran") for command in commands)
