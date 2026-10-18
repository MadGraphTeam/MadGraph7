# SPDX-License-Identifier: 0BSD
"""Existing full Community wheels use ordinary locked Cargo candidate builds."""

from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build_backend"))
import _pyamplicol_build as backend  # noqa: E402
from candidate_wheel import ENVIRONMENT, existing_community_wheel  # noqa: E402


def _module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _wheel(tmp_path, version="3.0.2"):
    path = tmp_path / f"symbolica-{version}-cp39-abi3-linux_x86_64.whl"
    # This is only a filename/configuration boundary test. Pip validates actual
    # wheels; the independent installed-package smoke uses the genuine build.
    path.write_bytes(b"unit-test input")
    return path


def test_existing_wheel_preserves_distribution_kernel_split_and_release_guard(tmp_path):
    wheel = _wheel(tmp_path)
    assert existing_community_wheel(ROOT, wheel) == wheel
    packages = tomllib.loads((ROOT / "Cargo.lock").read_text())["package"]
    assert next(p["version"] for p in packages if p["name"] == "symbolica") == "3.0.1"
    with pytest.raises(RuntimeError, match=r"symbolica==3\.0\.2"):
        existing_community_wheel(ROOT, _wheel(tmp_path, "3.0.1"))
    gate = _module("locked_candidate_gate", "tools/release/check_dependencies.py")
    assert gate.check(candidate=True, symbolica_wheel=wheel) == []
    assert any(
        issue.code == "candidate-wheel-mode"
        for issue in gate.check(candidate=False, symbolica_wheel=wheel)
    )


def test_existing_wheel_overlay_uses_canonical_cargo_and_generated_build_info(
    monkeypatch, tmp_path
):
    wheel = _wheel(tmp_path)
    monkeypatch.setenv(ENVIRONMENT, str(wheel))
    monkeypatch.setattr(backend, "_clean_source_revision", lambda **kw: "a" * 40)
    monkeypatch.setattr(
        backend,
        "_candidate_inputs",
        lambda: pytest.fail("ordinary locked build must not read managed source state"),
    )
    before = {
        p: (ROOT / p).read_bytes()
        for p in ("Cargo.toml", "Cargo.lock", "pyproject.toml")
    }
    identity = backend._native_build_inputs_digest(ROOT)
    wheel.write_bytes(b"different Python wheel bytes")
    assert backend._native_build_inputs_digest(ROOT) == identity
    with backend._overlay("candidate", temporary_directory=tmp_path) as (overlay, _):
        assert not (overlay / ".cargo/config.toml").exists()
        assert (overlay / "pyproject.toml").read_bytes() == before["pyproject.toml"]
        original = tomllib.loads(before["Cargo.toml"].decode())
        actual = tomllib.loads((overlay / "Cargo.toml").read_text())
        assert (
            actual["workspace"]["dependencies"] == original["workspace"]["dependencies"]
        )
        assert actual["patch"] == original["patch"]
        native = tomllib.loads((overlay / "Cargo.lock").read_text())["package"]
        assert next(p["version"] for p in native if p["name"] == "symbolica") == "3.0.1"
        build = json.loads((overlay / "src/pyamplicol/_build_info.json").read_text())
        assert build["publishable"] is False
        assert build["native_build_inputs_sha256"] == identity
        assert "+candidate." in build["version"]
    assert all((ROOT / p).read_bytes() == value for p, value in before.items())
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "release")
    with pytest.raises(RuntimeError, match="requires candidate mode"):
        backend._check_dependencies("release")


def test_installer_existing_wheel_never_rewrites_or_builds_community(
    monkeypatch, tmp_path
):
    module = _module(
        "locked_candidate_installer", "dependencies/install_dependencies.py"
    )
    wheel = _wheel(tmp_path)
    monkeypatch.setattr(module, "_ensure_venv", lambda *args, **kwargs: None)
    for name in (
        "_sources",
        "_configure_sources",
        "_write_cargo_config",
        "_write_candidate_lock",
        "_build_candidate_dependency_wheels",
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda *args, **kwargs: pytest.fail("historical source path called"),
        )
    installed, built = [], []
    monkeypatch.setattr(
        module,
        "_install_published_dependencies",
        lambda runner, **kw: installed.append(kw),
    )
    monkeypatch.setattr(
        module, "_build_project_wheel", lambda runner, **kw: built.append(kw)
    )
    assert (
        module.main(["--candidate", "--symbolica-wheel", str(wheel), "--dry-run"]) == 0
    )
    assert installed == [{"loader_wheel": None, "symbolica_wheel": wheel}]
    assert built == [{"mode": "candidate", "symbolica_wheel": wheel}]
