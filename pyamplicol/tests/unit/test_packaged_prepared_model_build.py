# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import shutil
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build_backend"))

import prepared_models as prepared_models_module  # noqa: E402
from prepared_models import (  # noqa: E402
    discard_release_packaged_prepared_model_store,
    project_release_packaged_prepared_model_store,
    stage_packaged_prepared_models,
    write_candidate_packaged_prepared_model_asset,
    write_release_packaged_prepared_model_asset,
)


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
    shutil.copy2(
        ROOT / "dependencies" / "contributor-lock.toml",
        dependencies / "contributor-lock.toml",
    )
    shutil.copy2(ROOT / "Cargo.toml", overlay / "Cargo.toml")
    shutil.copy2(
        ROOT / "dependencies" / "candidate-Cargo.lock",
        overlay / "Cargo.lock",
    )
    package_root = overlay / "src" / "pyamplicol"
    prepared_pack_compiler_sha256 = (
        prepared_models_module._prepared_pack_compiler_digest(package_root)
    )
    for metadata_path in (package_root / "assets" / "prepared_models").glob(
        "*.metadata.json"
    ):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["producer"]["prepared_pack_compiler_sha256"] = (
            prepared_pack_compiler_sha256
        )
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    metadata = json.loads(
        (
            package_root
            / "assets"
            / "prepared_models"
            / "built-in-sm-jit-o2-aarch64.metadata.json"
        ).read_text(encoding="utf-8")
    )
    producer = metadata["producer"]
    assert isinstance(producer, dict)
    (overlay / "src" / "pyamplicol" / "_build_info.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "source_revision": None,
                "version": producer["package_version"],
            }
        ),
        encoding="utf-8",
    )
    return overlay


def _release_overlay(tmp_path: Path) -> Path:
    overlay = tmp_path / "release-overlay"
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
    shutil.copy2(ROOT / "Cargo.lock", overlay / "Cargo.lock")
    return overlay


def _release_store(overlay: Path) -> Path:
    store = overlay / "release_assets" / "prepared_models"
    store.mkdir(parents=True)
    (store / "README.md").write_text("release source store\n", encoding="utf-8")
    source = overlay / "src/pyamplicol/assets/prepared_models"
    for path in source.iterdir():
        if path.name != "__init__.py":
            shutil.copy2(path, store / path.name)
    return store


def _release_bundle(overlay: Path, package_version: str = "0.1.0") -> object:
    package_root = overlay / "src" / "pyamplicol"
    compiled_schema = prepared_models_module._literal_assignment(
        package_root / "_internal" / "versions.py",
        "COMPILED_MODEL_SCHEMA_VERSION",
    )
    compiler_version = prepared_models_module._literal_assignment(
        package_root / "models" / "loading.py",
        "MODEL_COMPILER_VERSION",
    )
    compiler_digest = prepared_models_module._model_compiler_digest(package_root)
    source_digest = prepared_models_module._built_in_source_digest(package_root)
    symbolica_abi = prepared_models_module._literal_assignment(
        package_root / "_internal" / "versions.py",
        "SYMBOLICA_SERIALIZATION_ABI",
    )
    symjit_abi = prepared_models_module._literal_assignment(
        package_root / "_internal" / "versions.py",
        "SYMJIT_APPLICATION_ABI",
    )
    symjit_plane_abi = prepared_models_module._literal_assignment(
        package_root / "_internal" / "versions.py",
        "SYMJIT_PLANE_APPLICATION_ABI",
    )
    pack = SimpleNamespace(
        backend="jit",
        dependency_abis={
            "symbolica_serialization": symbolica_abi,
            "symbolica_version": "2.2.0",
            "symjit_application": symjit_abi,
            "symjit_plane_application": symjit_plane_abi,
        },
        kernels=(object(),),
        optimization_settings={"jit_optimization_level": 2},
        producer={"version": package_version},
        provenance={
            "compiled_model_digest": source_digest,
            "model_name": "built-in-sm",
            "model_source": {
                "digest": source_digest,
                "kind": "built-in-sm",
            },
        },
        target={
            "portable": True,
            "word_bits": 64,
            "endianness": "little",
            "target_triple": "symjit-storage-v3-portable",
            "cpu_features": [],
        },
    )
    return SimpleNamespace(
        backend="jit",
        compiled_model={
            "producer": {
                "compiled_model_schema_version": compiled_schema,
                "model_compiler_version": compiler_version,
                "model_compiler_sha256": compiler_digest,
                "pyamplicol": package_version,
            },
            "source": {
                "digest": source_digest,
                "kind": "built-in-sm",
            },
        },
        kernel_pack=pack,
    )


def test_source_ready_asset_metadata_is_derived_from_bundle_and_source(
    tmp_path: Path,
) -> None:
    asset_root = ROOT / "src/pyamplicol/assets/prepared_models"
    source_bundle = asset_root / "built-in-sm-jit-o2-aarch64.pyamplicol-model"
    expected_metadata = json.loads(
        (asset_root / "built-in-sm-jit-o2-aarch64.metadata.json").read_text(
            encoding="utf-8"
        )
    )

    metadata_path, bundle_path = write_candidate_packaged_prepared_model_asset(
        ROOT,
        source_bundle,
        tmp_path / "prepared",
        architecture="aarch64",
    )

    actual_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata["producer"]["prepared_pack_compiler_sha256"] = (
        prepared_models_module._prepared_pack_compiler_digest(
            ROOT / "src" / "pyamplicol"
        )
    )
    with (ROOT / "dependencies/contributor-lock.toml").open("rb") as stream:
        contributor = tomllib.load(stream)
    assert (
        actual_metadata["dependencies"]["symbolica_version"]
        == contributor["symbolica"]["candidate_version"]
    )
    assert actual_metadata["build_contract"] == {"mode": "candidate"}
    assert "native_build_inputs_sha256" not in actual_metadata["producer"]
    assert actual_metadata == expected_metadata
    assert bundle_path.read_bytes() == source_bundle.read_bytes()


def test_release_source_ready_asset_uses_only_release_lock_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay = _release_overlay(tmp_path)
    assert not (overlay / "dependencies/contributor-lock.toml").exists()
    source_bundle = tmp_path / "release.pyamplicol-model"
    source_bundle.write_bytes(b"release prepared bundle fixture")
    bundle = _release_bundle(overlay)
    monkeypatch.setattr(
        prepared_models_module,
        "_load_prepared_contract",
        lambda _path: SimpleNamespace(
            load_prepared_model_bundle=lambda _bundle_path: bundle
        ),
    )

    metadata_path, bundle_path = write_release_packaged_prepared_model_asset(
        overlay,
        source_bundle,
        tmp_path / "prepared",
        architecture="x86_64",
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["build_contract"] == {"mode": "release"}
    assert "native_build_inputs_sha256" not in metadata["producer"]
    assert metadata["producer"]["package_version"] == "0.1.0"
    assert metadata["dependencies"]["symbolica_version"] == "2.2.0"
    assert metadata["dependencies"]["symjit_version"] == "2.22.0"
    assert bundle_path.read_bytes() == source_bundle.read_bytes()


def test_release_source_ready_asset_rejects_candidate_producer_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay = _release_overlay(tmp_path)
    source_bundle = tmp_path / "candidate.pyamplicol-model"
    source_bundle.write_bytes(b"candidate prepared bundle fixture")
    bundle = _release_bundle(overlay, "0.1.0.dev0+candidate.123456789abc")
    monkeypatch.setattr(
        prepared_models_module,
        "_load_prepared_contract",
        lambda _path: SimpleNamespace(
            load_prepared_model_bundle=lambda _bundle_path: bundle
        ),
    )

    with pytest.raises(RuntimeError, match=r"producer version '0\.1\.0'"):
        write_release_packaged_prepared_model_asset(
            overlay,
            source_bundle,
            tmp_path / "prepared",
            architecture="aarch64",
        )


def test_release_source_store_projects_over_candidate_package_assets(
    tmp_path: Path,
) -> None:
    overlay = _release_overlay(tmp_path)
    package_assets = overlay / "src/pyamplicol/assets/prepared_models"
    package_init = (package_assets / "__init__.py").read_bytes()
    store = _release_store(overlay)
    expected: dict[str, bytes] = {}
    for path in store.iterdir():
        if path.name == "README.md":
            continue
        payload = f"release:{path.name}\n".encode()
        path.write_bytes(payload)
        expected[path.name] = payload

    assert project_release_packaged_prepared_model_store(
        overlay,
        require_store=True,
    )

    assert not (overlay / "release_assets").exists()
    assert (package_assets / "__init__.py").read_bytes() == package_init
    assert {
        path.name: path.read_bytes()
        for path in package_assets.iterdir()
        if path.name != "__init__.py"
    } == expected


def test_release_source_store_is_required_only_for_source_checkout(
    tmp_path: Path,
) -> None:
    overlay = _release_overlay(tmp_path)

    with pytest.raises(RuntimeError, match="source store is missing"):
        project_release_packaged_prepared_model_store(
            overlay,
            require_store=True,
        )

    assert (
        project_release_packaged_prepared_model_store(
            overlay,
            require_store=False,
        )
        is False
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "inventory is invalid"),
        ("unexpected", "inventory is invalid"),
        ("container", "container inventory is invalid"),
    ],
)
def test_release_source_store_rejects_incomplete_or_mixed_inventory(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    overlay = _release_overlay(tmp_path)
    store = _release_store(overlay)
    if mutation == "missing":
        next(store.glob("*.metadata.json")).unlink()
    elif mutation == "unexpected":
        (store / "candidate-only.json").write_text("{}\n", encoding="utf-8")
    else:
        (overlay / "release_assets" / "candidate-pack").mkdir()

    with pytest.raises(RuntimeError, match=match):
        project_release_packaged_prepared_model_store(
            overlay,
            require_store=True,
        )


def test_release_source_store_rejects_wrong_build_mode(
    tmp_path: Path,
) -> None:
    overlay = _release_overlay(tmp_path)
    store = _release_store(overlay)
    for metadata_path in store.glob("*.metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["build_contract"] = {"mode": "candidate"}
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    project_release_packaged_prepared_model_store(
        overlay,
        require_store=True,
    )

    with pytest.raises(RuntimeError, match="not the active 'release'"):
        stage_packaged_prepared_models(overlay, "release")


def test_store_discard_preserves_canonical_package_assets(
    tmp_path: Path,
) -> None:
    overlay = _release_overlay(tmp_path)
    package_assets = overlay / "src/pyamplicol/assets/prepared_models"
    before = {path.name: path.read_bytes() for path in package_assets.iterdir()}
    _release_store(overlay)

    discard_release_packaged_prepared_model_store(overlay)

    assert not (overlay / "release_assets").exists()
    assert {path.name: path.read_bytes() for path in package_assets.iterdir()} == before


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
        / "built-in-sm-jit-o2-aarch64.pyamplicol-model"
    )
    before = bundle.read_bytes()
    stage_packaged_prepared_models(overlay, "candidate")
    assert bundle.read_bytes() == before


def test_candidate_wheel_staging_ignores_legacy_build_info_fingerprint(
    tmp_path: Path,
) -> None:
    overlay = _overlay(tmp_path)
    build_info_path = overlay / "src" / "pyamplicol" / "_build_info.json"
    build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
    build_info["candidate_fingerprint"] = "0" * 12
    build_info_path.write_text(json.dumps(build_info), encoding="utf-8")

    stage_packaged_prepared_models(overlay, "candidate")


def test_wheel_staging_accepts_metadata_without_native_build_identity(
    tmp_path: Path,
) -> None:
    overlay = _overlay(tmp_path)
    asset_root = overlay / "src/pyamplicol/assets/prepared_models"
    for metadata_path in asset_root.glob("*.metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert "native_build_inputs_sha256" not in metadata["producer"]

    stage_packaged_prepared_models(overlay, "candidate")


def test_wheel_staging_rejects_built_in_source_edits(tmp_path: Path) -> None:
    overlay = _overlay(tmp_path)
    source = overlay / "src" / "pyamplicol" / "models" / "builtin" / "model.py"
    source.write_text(
        source.read_text(encoding="utf-8") + "\n# drift\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="model_source_digest is stale"):
        stage_packaged_prepared_models(overlay, "candidate")


def test_wheel_staging_rejects_prepared_payload_compiler_edits(
    tmp_path: Path,
) -> None:
    overlay = _overlay(tmp_path)
    source = overlay / "src" / "pyamplicol" / "evaluators" / "symbolica_compile.py"
    source.write_text(
        source.read_text(encoding="utf-8") + "\n# drift\n",
        encoding="utf-8",
    )
    with pytest.raises(
        RuntimeError,
        match="prepared_pack_compiler_sha256 is stale",
    ):
        stage_packaged_prepared_models(overlay, "candidate")


def test_wheel_staging_rejects_model_compiler_edits(tmp_path: Path) -> None:
    overlay = _overlay(tmp_path)
    source = overlay / "src/pyamplicol/models/loading.py"
    source.write_text(
        source.read_text(encoding="utf-8") + "\n# drift\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="model_compiler_sha256 is stale"):
        stage_packaged_prepared_models(overlay, "candidate")


def test_wheel_staging_rejects_bundle_hash_drift(tmp_path: Path) -> None:
    overlay = _overlay(tmp_path)
    bundle = (
        overlay
        / "src"
        / "pyamplicol"
        / "assets"
        / "prepared_models"
        / "built-in-sm-jit-o2-aarch64.pyamplicol-model"
    )
    bundle.write_bytes(bundle.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="size does not match"):
        stage_packaged_prepared_models(overlay, "candidate")


def test_wheel_staging_requires_both_architecture_assets(tmp_path: Path) -> None:
    overlay = _overlay(tmp_path)
    asset_root = overlay / "src" / "pyamplicol" / "assets" / "prepared_models"
    (asset_root / "built-in-sm-jit-o2-x86_64.pyamplicol-model").unlink()

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
        / "built-in-sm-jit-o2-x86_64.metadata.json"
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
        overlay / "src" / "pyamplicol" / "assets" / "prepared_models" / "second-pack"
    )
    unexpected.mkdir()
    with pytest.raises(RuntimeError, match="unexpected: second-pack"):
        stage_packaged_prepared_models(overlay, "candidate")


def test_candidate_payload_fails_closed_in_release_mode(tmp_path: Path) -> None:
    overlay = _overlay(tmp_path)
    with pytest.raises(RuntimeError, match="not the active 'release'"):
        stage_packaged_prepared_models(overlay, "release")


def test_release_payload_fails_closed_in_candidate_mode(tmp_path: Path) -> None:
    overlay = _overlay(tmp_path)
    asset_root = overlay / "src/pyamplicol/assets/prepared_models"
    for metadata_path in asset_root.glob("*.metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["build_contract"] = {"mode": "release"}
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="not the active 'candidate'"):
        stage_packaged_prepared_models(overlay, "candidate")
