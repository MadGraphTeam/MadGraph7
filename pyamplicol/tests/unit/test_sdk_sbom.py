# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build_backend"))

import sdk  # noqa: E402

TARGET = "x86_64-unknown-linux-gnu"
DEFAULT_MANIFEST = Path(
    "/Users/builder/checkouts/pyamplicol/rust/crates/rusticol-capi/Cargo.toml"
)
ROOT_ID = "path+file:///Users/builder/checkouts/pyamplicol/rusticol-capi#0.1.0"
NORMAL_ID = "registry+https://github.com/rust-lang/crates.io-index#normal-dep@1.2.3"
BUILD_ID = "registry+https://github.com/rust-lang/crates.io-index#build-dep@2.0.0"
TRANSITIVE_ID = (
    "registry+https://github.com/rust-lang/crates.io-index#transitive-dep@3.1.0"
)
DEV_ID = "registry+https://github.com/rust-lang/crates.io-index#dev-only@4.0.0"
LOCK_ONLY_ID = "registry+https://github.com/rust-lang/crates.io-index#lock-only@5.0.0"


def _package(
    package_id: str,
    name: str,
    version: str,
    *,
    manifest_path: Path,
    source: str | None,
) -> dict[str, object]:
    return {
        "id": package_id,
        "name": name,
        "version": version,
        "manifest_path": str(manifest_path),
        "source": source,
    }


def _dependency(
    name: str,
    package_id: str,
    *kinds: str | None,
) -> dict[str, object]:
    return {
        "name": name,
        "pkg": package_id,
        "dep_kinds": [{"kind": kind, "target": None} for kind in kinds],
    }


def _metadata(root_manifest: Path = DEFAULT_MANIFEST) -> dict[str, object]:
    registry = "registry+https://github.com/rust-lang/crates.io-index"
    packages = [
        _package(
            ROOT_ID,
            "rusticol-capi",
            "0.1.0",
            manifest_path=root_manifest,
            source=None,
        ),
        _package(
            NORMAL_ID,
            "normal-dep",
            "1.2.3",
            manifest_path=Path("/Users/builder/.cargo/normal-dep/Cargo.toml"),
            source=registry,
        ),
        _package(
            BUILD_ID,
            "build-dep",
            "2.0.0",
            manifest_path=Path("/Users/builder/.cargo/build-dep/Cargo.toml"),
            source=registry,
        ),
        _package(
            TRANSITIVE_ID,
            "transitive-dep",
            "3.1.0",
            manifest_path=Path("/Users/builder/.cargo/transitive-dep/Cargo.toml"),
            source=registry,
        ),
        _package(
            DEV_ID,
            "dev-only",
            "4.0.0",
            manifest_path=Path("/Users/builder/checkouts/dev-only/Cargo.toml"),
            source="git+file:///Users/builder/checkouts/dev-only",
        ),
        _package(
            LOCK_ONLY_ID,
            "lock-only",
            "5.0.0",
            manifest_path=Path("/Users/builder/.cargo/lock-only/Cargo.toml"),
            source=registry,
        ),
    ]
    nodes = [
        {
            "id": ROOT_ID,
            "deps": [
                _dependency("normal_dep", NORMAL_ID, "dev", None),
                _dependency("build_dep", BUILD_ID, "build"),
                _dependency("dev_only", DEV_ID, "dev"),
            ],
        },
        {
            "id": NORMAL_ID,
            "deps": [
                _dependency("transitive_dep", TRANSITIVE_ID, None),
                _dependency("dev_only", DEV_ID, "dev"),
            ],
        },
        {"id": BUILD_ID, "deps": []},
        {"id": TRANSITIVE_ID, "deps": []},
        {"id": DEV_ID, "deps": []},
        {"id": LOCK_ONLY_ID, "deps": []},
    ]
    return {
        "version": 1,
        "packages": packages,
        "workspace_members": [ROOT_ID],
        "workspace_root": "/Users/builder/checkouts/pyamplicol",
        "target_directory": "/Users/builder/checkouts/pyamplicol/target",
        "resolve": {"root": ROOT_ID, "nodes": nodes},
    }


def _sbom(metadata: object | None = None, manifest: Path = DEFAULT_MANIFEST) -> bytes:
    return sdk._cyclonedx_sbom(
        _metadata(manifest) if metadata is None else metadata,
        root_manifest=manifest,
        target=TARGET,
    )


def test_sbom_contains_only_normal_and_build_dependency_closure() -> None:
    document = json.loads(_sbom())
    root_ref = "pkg:cargo/rusticol-capi@0.1.0"
    normal_ref = "pkg:cargo/normal-dep@1.2.3"
    build_ref = "pkg:cargo/build-dep@2.0.0"
    transitive_ref = "pkg:cargo/transitive-dep@3.1.0"

    assert document["bomFormat"] == "CycloneDX"
    assert document["specVersion"] == "1.5"
    assert document["version"] == 1
    assert "serialNumber" not in document
    assert "timestamp" not in document["metadata"]
    assert document["metadata"]["component"]["bom-ref"] == root_ref
    assert {component["bom-ref"] for component in document["components"]} == {
        normal_ref,
        build_ref,
        transitive_ref,
    }
    dependencies = {
        dependency["ref"]: dependency["dependsOn"]
        for dependency in document["dependencies"]
    }
    assert dependencies == {
        build_ref: [],
        normal_ref: [transitive_ref],
        root_ref: [build_ref, normal_ref],
        transitive_ref: [],
    }


def test_sbom_is_deterministic_for_shuffled_cargo_metadata() -> None:
    original = _metadata()
    shuffled = copy.deepcopy(original)
    packages = shuffled["packages"]
    assert isinstance(packages, list)
    packages.reverse()
    resolve = shuffled["resolve"]
    assert isinstance(resolve, dict)
    nodes = resolve["nodes"]
    assert isinstance(nodes, list)
    nodes.reverse()
    for node in nodes:
        assert isinstance(node, dict)
        dependencies = node["deps"]
        assert isinstance(dependencies, list)
        dependencies.reverse()
        for dependency in dependencies:
            assert isinstance(dependency, dict)
            kinds = dependency["dep_kinds"]
            assert isinstance(kinds, list)
            kinds.reverse()

    assert _sbom(original) == _sbom(shuffled)


def test_sbom_strips_cargo_sources_and_checkout_paths() -> None:
    encoded = _sbom()
    assert b"file://" not in encoded
    assert b"path+file:" not in encoded
    assert b"/Users/" not in encoded
    assert b"checkouts" not in encoded
    document = json.loads(encoded)
    references = [document["metadata"]["component"]["bom-ref"]]
    references.extend(component["bom-ref"] for component in document["components"])
    assert all(reference.startswith("pkg:cargo/") for reference in references)


def test_sbom_rejects_a_dangling_resolve_edge() -> None:
    metadata = _metadata()
    resolve = metadata["resolve"]
    assert isinstance(resolve, dict)
    nodes = resolve["nodes"]
    assert isinstance(nodes, list)
    root = nodes[0]
    assert isinstance(root, dict)
    dependencies = root["deps"]
    assert isinstance(dependencies, list)
    dependency = dependencies[0]
    assert isinstance(dependency, dict)
    dependency["pkg"] = "registry+https://example.invalid#missing@1.0.0"

    with pytest.raises(RuntimeError, match="unknown package"):
        _sbom(metadata)


def test_sbom_rejects_ambiguous_canonical_package_references() -> None:
    metadata = _metadata()
    packages = metadata["packages"]
    assert isinstance(packages, list)
    build_package = packages[2]
    assert isinstance(build_package, dict)
    build_package["name"] = "normal-dep"
    build_package["version"] = "1.2.3"

    with pytest.raises(RuntimeError, match="ambiguous package reference"):
        _sbom(metadata)


def test_cargo_metadata_is_locked_offline_and_target_filtered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    manifest = root / sdk.CAPI_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text('[package]\nname = "rusticol-capi"\n', encoding="utf-8")
    target_dir = tmp_path / "cargo-target"
    observed: dict[str, Any] = {}

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        observed.update(
            command=command,
            cwd=cwd,
            env=env,
            check=check,
            capture_output=capture_output,
            text=text,
        )
        return subprocess.CompletedProcess(command, 0, stdout='{"version": 1}')

    monkeypatch.setattr(sdk.subprocess, "run", fake_run)
    monkeypatch.setenv("CARGO_HOME", str(tmp_path / "cargo-home"))

    assert sdk._cargo_metadata(root, target_dir, TARGET) == {"version": 1}
    assert observed["command"] == [
        "cargo",
        "metadata",
        "--format-version",
        "1",
        "--locked",
        "--offline",
        "--filter-platform",
        TARGET,
        "--manifest-path",
        str(manifest),
    ]
    assert observed["cwd"] == root
    environment = observed["env"]
    assert isinstance(environment, dict)
    assert environment["CARGO_HOME"] == os.environ["CARGO_HOME"]
    assert environment["CARGO_TARGET_DIR"] == str(target_dir)


def test_cargo_metadata_reports_captured_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    manifest = root / sdk.CAPI_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text('[package]\nname = "rusticol-capi"\n', encoding="utf-8")

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ["cargo", "metadata"],
            101,
            stdout="",
            stderr="error: candidate dependency is unavailable offline\n",
        )

    monkeypatch.setattr(sdk.subprocess, "run", fake_run)

    with pytest.raises(
        RuntimeError,
        match="candidate dependency is unavailable offline",
    ):
        sdk._cargo_metadata(root, tmp_path / "target", TARGET)


def test_build_sdk_stages_sbom_and_records_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    root_manifest = root / sdk.CAPI_MANIFEST
    root_manifest.parent.mkdir(parents=True)
    root_manifest.write_text('[package]\nname = "rusticol-capi"\n', encoding="utf-8")
    for source_name, _destination_name in sdk.SDK_SOURCES:
        source = root / source_name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(source.name, encoding="utf-8")
    archive = tmp_path / "built" / "librusticol_capi.a"
    archive.parent.mkdir()
    archive.write_bytes(b"static archive")
    cargo_metadata = _metadata(root_manifest)

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        message = {
            "reason": "compiler-artifact",
            "target": {"name": "rusticol_capi"},
            "filenames": [str(archive)],
        }
        native = "native-static-libs: -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(message),
            stderr=native,
        )

    monkeypatch.delenv("CARGO_BUILD_TARGET", raising=False)
    monkeypatch.delenv("MATURIN_BUILD_TARGET", raising=False)
    monkeypatch.setattr(sdk, "_host_target", lambda _root: TARGET)
    monkeypatch.setattr(sdk, "_cargo_metadata", lambda *_args: cargo_metadata)
    monkeypatch.setattr(sdk.subprocess, "run", fake_run)
    monkeypatch.setattr(sdk, "_package_version", lambda _root: "0.1.0")
    monkeypatch.setattr(sdk, "_scan_archive_symbols", lambda _path: None)
    monkeypatch.setattr(
        sdk, "_validate_archive_linkage", lambda *_args, **_kwargs: None
    )

    staging = sdk.build_sdk(root, tmp_path / "cargo-target")
    sbom_path = staging / sdk.SDK_SBOM
    metadata = json.loads((staging / "metadata.json").read_text(encoding="utf-8"))

    assert sbom_path.read_bytes() == _sbom(cargo_metadata, root_manifest)
    assert metadata["sbom"] == "sboms/rusticol-capi.cyclonedx.json"
    assert metadata["sbom_sha256"] == hashlib.sha256(sbom_path.read_bytes()).hexdigest()


def test_build_sdk_prefetches_before_offline_build_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    root_manifest = root / sdk.CAPI_MANIFEST
    root_manifest.parent.mkdir(parents=True)
    root_manifest.write_text('[package]\nname = "rusticol-capi"\n', encoding="utf-8")
    for source_name, _destination_name in sdk.SDK_SOURCES:
        source = root / source_name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(source.name, encoding="utf-8")
    archive = tmp_path / "built" / "librusticol_capi.a"
    archive.parent.mkdir()
    archive.write_bytes(b"static archive")
    cargo_metadata = _metadata(root_manifest)
    cargo_home = tmp_path / "cargo-home"
    cargo_home.mkdir()
    cache_marker = cargo_home / "fetch-populated-cache"
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command.copy())
        if command[1] == "fetch":
            cache_marker.write_text("ready\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[1] == "rustc":
            assert cache_marker.is_file(), "cargo rustc ran before cargo fetch"
            assert "--locked" in command
            assert "--offline" in command
            message = {
                "reason": "compiler-artifact",
                "target": {"name": "rusticol_capi"},
                "filenames": [str(archive)],
            }
            native = "native-static-libs: -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc"
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(message),
                stderr=native,
            )
        if command[1] == "metadata":
            assert cache_marker.is_file(), "metadata ran before cargo fetch"
            assert "--locked" in command
            assert "--offline" in command
            assert command[command.index("--filter-platform") + 1] == TARGET
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(cargo_metadata),
                stderr="",
            )
        raise AssertionError(f"unexpected Cargo command: {command}")

    monkeypatch.delenv("CARGO_BUILD_TARGET", raising=False)
    monkeypatch.delenv("MATURIN_BUILD_TARGET", raising=False)
    monkeypatch.setenv("CARGO_HOME", str(cargo_home))
    monkeypatch.setattr(sdk, "_host_target", lambda _root: TARGET)
    monkeypatch.setattr(sdk.subprocess, "run", fake_run)
    monkeypatch.setattr(sdk, "_package_version", lambda _root: "0.1.0")
    monkeypatch.setattr(sdk, "_scan_archive_symbols", lambda _path: None)
    monkeypatch.setattr(
        sdk, "_validate_archive_linkage", lambda *_args, **_kwargs: None
    )

    staging = sdk.build_sdk(root, tmp_path / "cargo-target")

    assert [command[1] for command in commands] == ["fetch", "rustc", "metadata"]
    assert (staging / sdk.SDK_SBOM).read_bytes() == _sbom(
        cargo_metadata,
        root_manifest,
    )


@pytest.mark.skipif(os.name == "nt", reason="the fake Cargo executable is POSIX-only")
def test_build_sdk_fetches_into_empty_cargo_home_before_offline_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    root_manifest = root / sdk.CAPI_MANIFEST
    root_manifest.parent.mkdir(parents=True)
    root_manifest.write_text('[package]\nname = "rusticol-capi"\n', encoding="utf-8")
    for source_name, _destination_name in sdk.SDK_SOURCES:
        source = root / source_name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(source.name, encoding="utf-8")

    cargo_home = tmp_path / "empty-cargo-home"
    cargo_home.mkdir()
    target_dir = tmp_path / "cargo-target"
    archive = tmp_path / "fake-build" / "librusticol_capi.a"
    metadata_path = tmp_path / "cargo-metadata.json"
    cargo_metadata = _metadata(root_manifest)
    metadata_path.write_text(json.dumps(cargo_metadata), encoding="utf-8")
    command_log = tmp_path / "cargo-commands.jsonl"
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_cargo = fake_bin / "cargo"
    fake_cargo.write_text(
        f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path

arguments = sys.argv[1:]
cargo_home = Path(os.environ["CARGO_HOME"])
marker = cargo_home / "fetch-populated-cache"
with Path(os.environ["FAKE_CARGO_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(arguments) + "\\n")

if arguments[0] == "fetch":
    if any(cargo_home.iterdir()):
        raise SystemExit("fake Cargo expected an initially empty CARGO_HOME")
    marker.write_text("ready\\n", encoding="utf-8")
elif arguments[0] == "rustc":
    if not marker.is_file():
        raise SystemExit("offline cargo rustc ran before cargo fetch")
    if "--locked" not in arguments or "--offline" not in arguments:
        raise SystemExit("cargo rustc must be locked and offline")
    archive = Path(os.environ["FAKE_CARGO_ARCHIVE"])
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"static archive")
    print(json.dumps({{
        "reason": "compiler-artifact",
        "target": {{"name": "rusticol_capi"}},
        "filenames": [str(archive)],
    }}))
    print(
        "native-static-libs: -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc",
        file=sys.stderr,
    )
elif arguments[0] == "metadata":
    if not marker.is_file():
        raise SystemExit("offline metadata ran before cargo fetch populated the cache")
    if "--locked" not in arguments or "--offline" not in arguments:
        raise SystemExit("metadata must be locked and offline")
    target_index = arguments.index("--filter-platform") + 1
    if arguments[target_index] != os.environ["FAKE_CARGO_TARGET"]:
        raise SystemExit("metadata used the wrong target filter")
    print(Path(os.environ["FAKE_CARGO_METADATA"]).read_text(encoding="utf-8"))
else:
    raise SystemExit(f"unexpected fake Cargo command: {{arguments}}")
""",
        encoding="utf-8",
    )
    fake_cargo.chmod(0o755)

    monkeypatch.delenv("CARGO_BUILD_TARGET", raising=False)
    monkeypatch.delenv("MATURIN_BUILD_TARGET", raising=False)
    monkeypatch.setenv("CARGO_HOME", str(cargo_home))
    monkeypatch.setenv("FAKE_CARGO_ARCHIVE", str(archive))
    monkeypatch.setenv("FAKE_CARGO_LOG", str(command_log))
    monkeypatch.setenv("FAKE_CARGO_METADATA", str(metadata_path))
    monkeypatch.setenv("FAKE_CARGO_TARGET", TARGET)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(sdk, "_host_target", lambda _root: TARGET)
    monkeypatch.setattr(sdk, "_package_version", lambda _root: "0.1.0")
    monkeypatch.setattr(sdk, "_scan_archive_symbols", lambda _path: None)
    monkeypatch.setattr(
        sdk, "_validate_archive_linkage", lambda *_args, **_kwargs: None
    )

    staging = sdk.build_sdk(root, target_dir)

    commands = [
        json.loads(line)
        for line in command_log.read_text(encoding="utf-8").splitlines()
    ]
    assert [command[0] for command in commands] == ["fetch", "rustc", "metadata"]
    assert cargo_home.joinpath("fetch-populated-cache").is_file()
    assert (staging / sdk.SDK_SBOM).read_bytes() == _sbom(
        cargo_metadata,
        root_manifest,
    )
