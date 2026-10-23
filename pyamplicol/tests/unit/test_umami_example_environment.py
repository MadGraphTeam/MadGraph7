# SPDX-License-Identifier: 0BSD
"""Exercise example preflights without native or optional integration packages."""

from __future__ import annotations

import builtins
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "umami_example"


@pytest.fixture
def example_modules(monkeypatch):
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, EXAMPLE / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    metadata = load("_metadata", "_metadata.py")
    runner = load("_umami_example_run", "run.py")
    return metadata, runner


def install_exporter(monkeypatch, path, *, legacy=False):
    """Model the imported package, not pytest's repository source path."""
    package = ModuleType("pyamplicol")
    package.__path__ = [str(path.parents[1])]
    artifacts = ModuleType("pyamplicol.artifacts")
    artifacts.__path__ = [str(path.parent)]
    exporter = ModuleType("pyamplicol.artifacts.umami")
    exporter.__file__ = str(path)

    def old_exporter(*, processes, compiled_model, grouping="exact"):
        raise AssertionError("preflight must not generate metadata")

    def current_exporter(*, processes, compiled_model):
        raise AssertionError("preflight must not generate metadata")

    exporter.build_umami_metadata = old_exporter if legacy else current_exporter
    package.artifacts = artifacts
    artifacts.umami = exporter
    for module in (package, artifacts, exporter):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return exporter


def metadata_document(*, physical=24, exported=4):
    entries = [
        {"members": [{} for _ in range(physical // exported)]} for _ in range(exported)
    ]
    return {
        "provider": {
            "id": "lc",
            "incoming_count": 2,
            "color_accuracy": "lc",
            "library": "libumami.so",
        },
        "channels": [
            {"processes": entries[:2]},
            {"processes": entries[2:3]},
            {"processes": entries[3:]},
        ],
        "grouping": {
            "physical_contributions": physical,
            "exported_contributions": exported,
        },
    }


def write_metadata(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def forbid_expensive_work(monkeypatch, runner):
    def forbidden(*args, **kwargs):
        pytest.fail("preflight must finish before dependencies, generation or build")

    monkeypatch.setattr(runner, "generate", forbidden)
    monkeypatch.setattr(runner, "execute", forbidden)
    monkeypatch.setattr(runner.shutil, "which", forbidden)
    original_import = builtins.__import__

    def import_without_optional(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"madspace", "madnis", "vegas", "torch"}:
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_optional)


def test_current_exporter_returns_the_imported_path(
    example_modules, monkeypatch, tmp_path
):
    metadata, _ = example_modules
    installed = tmp_path / "site-packages/pyamplicol/artifacts/umami.py"
    install_exporter(monkeypatch, installed)

    assert metadata.require_current_exporter() == installed.resolve()


def test_stale_exporter_stops_runner_before_generation(
    example_modules, monkeypatch, tmp_path
):
    _, runner = example_modules
    installed = tmp_path / "site-packages/pyamplicol/artifacts/umami.py"
    install_exporter(monkeypatch, installed, legacy=True)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["run.py", "--output", str(output)])
    forbid_expensive_work(monkeypatch, runner)

    with pytest.raises(RuntimeError) as error:
        runner.main()

    message = str(error.value)
    assert str(installed) in message
    assert sys.executable in message
    assert "reinstall" in message.lower()
    assert "--output" in message
    assert "--reuse" in message
    assert not output.exists()


@pytest.mark.parametrize("physical", [4, 24])
def test_current_metadata_accepts_uncompressed_and_grouped_contributions(
    example_modules, tmp_path, physical
):
    metadata, _ = example_modules
    data = metadata_document(physical=physical)
    path = write_metadata(tmp_path / "metadata.json", data)

    assert metadata.load_metadata(path) == data
    assert metadata.grouping_summary(data) == (
        "3 integration channels, 4 representative contributions, "
        f"{physical} physical contributions"
    )


def test_current_restricted_metadata_does_not_require_a_grouping_reduction(
    example_modules, tmp_path
):
    metadata, _ = example_modules
    data = metadata_document(physical=1, exported=1)
    data["channels"] = data["channels"][:1]
    path = write_metadata(tmp_path / "metadata.json", data)

    assert metadata.load_metadata(path) == data


@pytest.mark.parametrize(
    "grouping",
    [
        None,
        {},
        {"physical_contributions": 24},
        {"exported_contributions": 4},
        {"mode": "exact", "physical_contributions": 24, "exported_contributions": 24},
    ],
)
def test_legacy_metadata_is_rejected_with_recovery_instructions(
    example_modules, tmp_path, grouping
):
    metadata, _ = example_modules
    data = metadata_document()
    if grouping is None:
        del data["grouping"]
    else:
        data["grouping"] = grouping
    path = write_metadata(tmp_path / "metadata.json", data)

    with pytest.raises(ValueError) as error:
        metadata.load_metadata(path)

    message = str(error.value)
    assert str(path) in message
    assert "reinstall" in message.lower()
    assert "--output" in message
    assert "--reuse" in message


def test_reuse_rejects_legacy_metadata_before_optional_dependencies_or_build(
    example_modules, monkeypatch, tmp_path
):
    _, runner = example_modules
    install_exporter(monkeypatch, tmp_path / "installed/pyamplicol/artifacts/umami.py")
    data = metadata_document()
    data["grouping"]["mode"] = "exact"
    path = write_metadata(tmp_path / "output/lc/API/umami/metadata.json", data)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--output", str(tmp_path / "output"), "--reuse", "--accuracy", "lc"],
    )
    forbid_expensive_work(monkeypatch, runner)

    with pytest.raises(ValueError, match=r"metadata\.json"):
        runner.main()

    assert json.loads(path.read_text()) == data


def test_script_detects_the_installed_exporter_from_an_unrelated_working_directory(
    tmp_path,
):
    """The real script entry point must not depend on pytest's src path injection."""
    site = tmp_path / "site-packages"
    package = site / "pyamplicol"
    artifacts = package / "artifacts"
    artifacts.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (artifacts / "__init__.py").write_text("")
    exporter = artifacts / "umami.py"
    exporter.write_text(
        "def build_umami_metadata(*, processes, compiled_model, grouping='exact'):\n"
        "    raise AssertionError('generation must not start')\n"
    )
    output = tmp_path / "output"
    result = subprocess.run(
        [sys.executable, "-S", str(EXAMPLE / "run.py"), "--output", str(output)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(site)},
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert str(exporter) in result.stderr
    assert sys.executable in result.stderr
    assert "reinstall" in result.stderr.lower()
    assert "--reuse" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert not output.exists()
