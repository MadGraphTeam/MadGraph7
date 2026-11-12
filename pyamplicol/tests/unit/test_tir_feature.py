# SPDX-License-Identifier: 0BSD
"""TIRust behind the cargo feature ``tir`` (m-lit PLAN §2.c.2 item 3).

``tir`` is a DEFAULT feature of every crate that links TIRust (USER
2026-11-06 ~13:35): developer builds, tests, staged natives, CI, wheels and
releases have the backend, and ``--no-default-features`` builds without it.
That needs every crate depending on a ``tir`` crate to take it with
``default-features = false`` and to forward ``tir``; otherwise a dependent's
default would switch it back on. The native reports what it was built with.
"""

from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TIRUST = "https://github.com/huashengshao/TIRust.git"


def _manifests() -> dict[str, tuple[Path, dict]]:
    found = {}
    for path in sorted((ROOT / "rust").rglob("Cargo.toml")):
        if "target" in path.relative_to(ROOT).parts:
            continue
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        if "package" in data:
            found[data["package"]["name"]] = (path, data)
    return found


def test_tir_is_a_default_feature_every_dependent_can_turn_off() -> None:
    manifests = _manifests()
    with_tir = {
        name: data
        for name, (_, data) in manifests.items()
        if "tir" in data.get("features", {})
    }
    assert {
        "rusticol-loop-reduction",
        "rusticol-loop-python",
        "rusticol-loop-sdk",
        "rusticol-python",
        "rusticol-capi",
    } <= set(with_tir)
    for name, data in with_tir.items():
        assert "tir" in data["features"]["default"], name
    for name, (_, data) in manifests.items():
        for dependency, spec in data.get("dependencies", {}).items():
            if dependency not in with_tir or not isinstance(spec, dict):
                continue
            # A dependent of a tir crate takes it without its defaults and
            # forwards tir, so --no-default-features reaches TIRust.
            assert spec.get("default-features") is False, (name, dependency)
            assert f"{dependency}/tir" in data["features"]["tir"], (name, dependency)

    reduction = with_tir["rusticol-loop-reduction"]
    assert set(reduction["features"]["tir"]) == {
        "dep:iregi-rust",
        "dep:iregi-masters",
        "rusticol-core/tirust-capability",
    }
    # rusticol-core asks TIRust's capability query in the rank admission
    # (m-lit lit-eft) behind a feature only the loop crate's tir turns on:
    # not a default, so a --no-default-features build links no TIRust.
    core = manifests["rusticol-core"][1]
    assert core["features"]["tirust-capability"] == ["loop-v2", "dep:iregi-rust"]
    assert "tirust-capability" not in core["features"]["default"]
    for name, (_, data) in manifests.items():
        for feature, members in data.get("features", {}).items():
            if "rusticol-core/tirust-capability" in members:
                assert (name, feature) == ("rusticol-loop-reduction", "tir")
    # One TIRust pin across the workspace.
    pins = set()
    for name, (_, data) in manifests.items():
        for crate, spec in data.get("dependencies", {}).items():
            if isinstance(spec, dict) and spec.get("git") == TIRUST:
                assert spec["optional"] is True, (name, crate)
                pins.add(spec["rev"])
    for crate in ("iregi-rust", "iregi-masters"):
        assert reduction["dependencies"][crate]["git"] == TIRUST
    assert core["dependencies"]["iregi-rust"]["git"] == TIRUST
    assert len(pins) == 1 and len(next(iter(pins))) == 40


def test_build_loop_native_builds_with_and_without_tir() -> None:
    spec = importlib.util.spec_from_file_location(
        "_tir_build_loop_native", ROOT / "tools" / "developer" / "build_loop_native.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with_tir = module.cargo_features()
    without = module.cargo_features(tir=False)
    assert "--no-default-features" not in with_tir
    assert without == ["--no-default-features", *with_tir]


def _staged_tir() -> bool | None:
    """The ``tir`` feature of the staged source-runtime native, as its
    ``build-info.json`` records it (None: not recorded, e.g. a wheel)."""

    staged = ROOT / ".artifacts" / "source-runtime" / "_build_info.json"
    if not staged.is_file():
        return None
    native_build = json.loads(staged.read_text()).get("native_build") or {}
    info = Path(str(native_build.get("directory", ""))) / "build-info.json"
    if not info.is_file():
        return None
    features = json.loads(info.read_text()).get("cargo_features")
    return None if features is None else bool(features["tir"])


@pytest.mark.loop_fast
def test_the_native_extension_reports_whether_it_links_tirust() -> None:
    from pyamplicol.loops.native import load_loop_native

    available = load_loop_native().tirust_available()
    expected = _staged_tir()
    assert available is (True if expected is None else expected)
