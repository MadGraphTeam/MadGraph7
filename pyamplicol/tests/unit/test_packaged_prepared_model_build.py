# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build_backend"))

from prepared_models import stage_packaged_prepared_models  # noqa: E402


def _overlay(tmp_path: Path) -> Path:
    overlay = tmp_path / "overlay"
    shutil.copytree(
        ROOT / "src" / "pyamplicol",
        overlay / "src" / "pyamplicol",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    dependencies = overlay / "dependencies"
    dependencies.mkdir()
    shutil.copy2(
        ROOT / "dependencies" / "release-lock.toml",
        dependencies / "release-lock.toml",
    )
    shutil.copy2(ROOT / "Cargo.toml", overlay / "Cargo.toml")
    shutil.copy2(
        ROOT / "dependencies" / "candidate-Cargo.lock",
        overlay / "Cargo.lock",
    )
    metadata = json.loads(
        (
            ROOT
            / "src"
            / "pyamplicol"
            / "assets"
            / "prepared_models"
            / "built-in-sm-jit-o3-aarch64.metadata.json"
        ).read_text(encoding="utf-8")
    )
    build_contract = metadata["build_contract"]
    producer = metadata["producer"]
    assert isinstance(build_contract, dict)
    assert isinstance(producer, dict)
    (overlay / "src" / "pyamplicol" / "_build_info.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "candidate_fingerprint": build_contract["candidate_fingerprint"],
                "source_revision": None,
                "version": producer["package_version"],
            }
        ),
        encoding="utf-8",
    )
    return overlay


def test_candidate_wheel_staging_accepts_exact_packaged_model(
    tmp_path: Path,
) -> None:
    overlay = _overlay(tmp_path)
    bundle = (
        overlay
        / "src"
        / "pyamplicol"
        / "assets"
        / "prepared_models"
        / "built-in-sm-jit-o3-aarch64.pyamplicol-model"
    )
    before = bundle.read_bytes()
    stage_packaged_prepared_models(overlay, "candidate")
    assert bundle.read_bytes() == before


def test_wheel_staging_rejects_built_in_source_drift(tmp_path: Path) -> None:
    overlay = _overlay(tmp_path)
    source = overlay / "src" / "pyamplicol" / "models" / "builtin" / "model.py"
    source.write_text(
        source.read_text(encoding="utf-8") + "\n# drift\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="model_source_digest is stale"):
        stage_packaged_prepared_models(overlay, "candidate")


def test_wheel_staging_rejects_prepared_payload_compiler_drift(
    tmp_path: Path,
) -> None:
    overlay = _overlay(tmp_path)
    source = (
        overlay
        / "src"
        / "pyamplicol"
        / "evaluators"
        / "symbolica_compile.py"
    )
    source.write_text(
        source.read_text(encoding="utf-8") + "\n# drift\n",
        encoding="utf-8",
    )
    with pytest.raises(
        RuntimeError,
        match="prepared_pack_compiler_sha256 is stale",
    ):
        stage_packaged_prepared_models(overlay, "candidate")


def test_wheel_staging_rejects_bundle_hash_drift(tmp_path: Path) -> None:
    overlay = _overlay(tmp_path)
    bundle = (
        overlay
        / "src"
        / "pyamplicol"
        / "assets"
        / "prepared_models"
        / "built-in-sm-jit-o3-aarch64.pyamplicol-model"
    )
    bundle.write_bytes(bundle.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="size does not match"):
        stage_packaged_prepared_models(overlay, "candidate")


def test_wheel_staging_requires_both_architecture_assets(tmp_path: Path) -> None:
    overlay = _overlay(tmp_path)
    asset_root = overlay / "src" / "pyamplicol" / "assets" / "prepared_models"
    (asset_root / "built-in-sm-jit-o3-x86_64.pyamplicol-model").unlink()

    with pytest.raises(RuntimeError, match=r"missing:.*x86_64"):
        stage_packaged_prepared_models(overlay, "candidate")


def test_wheel_staging_rejects_architecture_target_drift(tmp_path: Path) -> None:
    overlay = _overlay(tmp_path)
    metadata_path = (
        overlay
        / "src"
        / "pyamplicol"
        / "assets"
        / "prepared_models"
        / "built-in-sm-jit-o3-x86_64.metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["target"]["target_triple"] = "symjit-storage-v3-aarch64"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="target does not match"):
        stage_packaged_prepared_models(overlay, "candidate")


def test_wheel_staging_rejects_unexpected_prepared_model_tree(
    tmp_path: Path,
) -> None:
    overlay = _overlay(tmp_path)
    unexpected = (
        overlay
        / "src"
        / "pyamplicol"
        / "assets"
        / "prepared_models"
        / "second-pack"
    )
    unexpected.mkdir()
    with pytest.raises(RuntimeError, match="unexpected: second-pack"):
        stage_packaged_prepared_models(overlay, "candidate")


def test_candidate_payload_fails_closed_in_release_mode(tmp_path: Path) -> None:
    overlay = _overlay(tmp_path)
    with pytest.raises(RuntimeError, match="not the active 'release'"):
        stage_packaged_prepared_models(overlay, "release")
