# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build_backend"))

import _pyamplicol_build as backend  # noqa: E402
import sdk  # noqa: E402


def _candidate_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    candidate_lock = tmp_path / "Cargo.lock"
    candidate_config = tmp_path / "config.toml"
    installer_state = tmp_path / "install-state.json"
    candidate_lock.write_bytes((ROOT / "Cargo.lock").read_bytes())
    candidate_config.write_text("[patch.crates-io]\n", encoding="utf-8")
    with (ROOT / "dependencies" / "contributor-lock.toml").open("rb") as stream:
        contributor = tomllib.load(stream)
    revisions = {
        "gammaloop": contributor["gammaloop_candidate"]["revision"],
        "symbolica": contributor["symbolica"]["candidate_revision"],
        "symbolica-community": contributor["symbolica"]["community_revision"],
        "symjit": contributor["symjit"]["candidate_revision"],
    }
    sources = {name: {"revision": revision} for name, revision in revisions.items()}
    installer_state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "sources": sources,
            }
        ),
        encoding="utf-8",
    )
    return candidate_lock, candidate_config, installer_state


def test_candidate_overlay_is_versioned_without_mutating_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_cargo = (ROOT / "Cargo.toml").read_bytes()
    source_lock = (ROOT / "Cargo.lock").read_bytes()
    source_core = (
        ROOT / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
    ).read_bytes()
    candidate_inputs = _candidate_inputs(tmp_path)
    monkeypatch.setattr(
        backend,
        "_candidate_inputs",
        lambda: candidate_inputs,
    )
    monkeypatch.setattr(backend, "_clean_source_revision", lambda: "a" * 40)

    with backend._overlay("candidate") as (overlay, target):
        assert target.parent == overlay.parent
        cargo = (overlay / "Cargo.toml").read_text(encoding="utf-8")
        lock = (overlay / "Cargo.lock").read_text(encoding="utf-8")
        assert (overlay / ".cargo" / "config.toml").read_bytes() == (
            candidate_inputs[1].read_bytes()
        )
        assert 'version = "0.1.0-dev.0+candidate.' in cargo
        assert lock.count('version = "0.1.0-dev.0+candidate.') == 3
        candidate_core = (
            overlay / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
        ).read_text(encoding="utf-8")
        with (ROOT / "dependencies" / "contributor-lock.toml").open("rb") as stream:
            candidate_version = tomllib.load(stream)["symjit"]["candidate_version"]
        assert f'symjit = {{ version = "={candidate_version}"' in candidate_core
        build_info = json.loads(
            (overlay / "src" / "pyamplicol" / "_build_info.json").read_text(
                encoding="utf-8"
            )
        )
        assert build_info["publishable"] is False
        assert len(build_info["candidate_fingerprint"]) == 12
        assert build_info["source_revision"] == "a" * 40
        target = "aarch64-apple-darwin"
        backend._stage_selftest_fixture(overlay, target)
        selftest_manifest = json.loads(
            (
                overlay
                / "src"
                / "pyamplicol"
                / "assets"
                / "selftest"
                / target
                / "artifact"
                / "artifact.json"
            ).read_text(encoding="utf-8")
        )
        assert selftest_manifest["producer"]["version"] == build_info["version"]
        assert selftest_manifest["runtime"]["engine_version"] == build_info["version"]
        compiled_payloads = [
            payload
            for payload in selftest_manifest["payloads"]
            if payload["role"] == "compiled-model"
        ]
        assert len(compiled_payloads) == 1
        compiled_payload = compiled_payloads[0]
        compiled_path = (
            overlay
            / "src"
            / "pyamplicol"
            / "assets"
            / "selftest"
            / target
            / "artifact"
            / compiled_payload["path"]
        )
        compiled_data = compiled_path.read_bytes()
        compiled_model = json.loads(compiled_data)
        assert compiled_model["producer"]["pyamplicol"] == build_info["version"]
        assert compiled_payload["sha256"] == hashlib.sha256(compiled_data).hexdigest()
        assert compiled_payload["size_bytes"] == len(compiled_data)
        assert selftest_manifest["producer"]["target"] == {
            "cpu_features": [],
            "triple": target,
        }
        target_payloads = [
            payload["target"]
            for payload in selftest_manifest["payloads"]
            if "target" in payload
        ]
        assert target_payloads
        assert target_payloads == [
            {"cpu_features": [], "triple": target} for _payload in target_payloads
        ]
        expected = json.loads(
            (
                overlay
                / "src"
                / "pyamplicol"
                / "assets"
                / "selftest"
                / target
                / "expected.json"
            ).read_text(encoding="utf-8")
        )
        assert expected["target"] == target
        assert expected["compatible_targets"] == sorted(
            backend._PORTABLE_SELFTEST_TARGETS
        )
        assert not (
            overlay
            / "src"
            / "pyamplicol"
            / "assets"
            / "selftest"
            / backend._PORTABLE_SELFTEST_TEMPLATE
        ).exists()
        content = dict(selftest_manifest)
        claimed_id = content.pop("artifact_id")
        canonical = (
            json.dumps(content, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        assert claimed_id == hashlib.sha256(canonical).hexdigest()

    assert (ROOT / "Cargo.toml").read_bytes() == source_cargo
    assert (ROOT / "Cargo.lock").read_bytes() == source_lock
    assert (
        ROOT / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
    ).read_bytes() == source_core


def test_release_overlay_uses_only_canonical_registry_lock() -> None:
    with backend._overlay("release") as (overlay, _target):
        assert (overlay / "Cargo.lock").read_bytes() == (
            ROOT / "Cargo.lock"
        ).read_bytes()
        assert not (overlay / ".cargo" / "config.toml").exists()
        assert not (overlay / "dependencies" / "candidate-Cargo.lock").exists()


def test_selftest_staging_rejects_an_unavailable_target() -> None:
    with (
        backend._overlay("release") as (overlay, _target),
        pytest.raises(RuntimeError, match="unsupported self-test target"),
    ):
        backend._stage_selftest_fixture(overlay, "x86_64-unknown-test")


@pytest.mark.parametrize("target", sorted(backend._PORTABLE_SELFTEST_TARGETS))
def test_portable_selftest_stages_for_every_release_target(target: str) -> None:
    with backend._overlay("release") as (overlay, _target):
        backend._stage_selftest_fixture(overlay, target)
        fixture_root = overlay / "src" / "pyamplicol" / "assets" / "selftest"
        assert tuple(path.name for path in fixture_root.iterdir()) == (target,)
        manifest = json.loads(
            (fixture_root / target / "artifact" / "artifact.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["producer"]["target"]["triple"] == target
        assert {
            payload["target"]["triple"]
            for payload in manifest["payloads"]
            if "target" in payload
        } == {target}


def test_cargo_overlay_rejects_unknown_resolution_mode(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unsupported Cargo input mode"):
        backend._stage_cargo_inputs(tmp_path, "mixed")


def test_candidate_inputs_fail_closed_when_installer_has_not_run() -> None:
    expected = (
        ROOT / "dependencies" / "candidate-Cargo.lock",
        ROOT / "dependencies" / "candidate-cargo-config.toml",
        ROOT / "dependencies" / "install-state.json",
    )
    if all(path.is_file() for path in expected):
        pytest.skip("candidate environment is installed")
    with pytest.raises(RuntimeError, match="dev-install"):
        backend._candidate_inputs()


def test_candidate_source_distributions_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "candidate")
    with pytest.raises(RuntimeError, match="wheel-only"):
        backend.build_sdist(str(tmp_path))


def test_overlay_excludes_managed_dependencies_and_includes_licenses() -> None:
    with backend._overlay("release") as (overlay, _target):
        assert (overlay / "licenses" / "MadGraph5_aMCatNLO.txt").is_file()
        assert (overlay / "rust-toolchain.toml").is_file()
        assert (overlay / "tools" / "typing" / "check_public_typing.py").is_file()
        assert (overlay / "dependencies" / "release-lock.toml").is_file()
        assert not (overlay / "dependencies" / "checkouts").exists()
        assert not (overlay / "dependencies" / "contributor-lock.toml").exists()
        assert not (overlay / "dependencies" / "install_dependencies.py").exists()
        assert not (overlay / "dependencies" / "install-state.json").exists()
        assert not (overlay / "dependencies" / "patches").exists()
        assert not (overlay / "dependencies" / "python-runtime-lock.toml").exists()
        assert not (overlay / "build_backend" / "python_lock.py").exists()


def test_wheel_examples_are_staged_from_the_single_canonical_tree() -> None:
    with backend._overlay("release") as (overlay, _target):
        packaged = overlay / "src" / "pyamplicol" / "_examples"
        assert not packaged.exists()
        backend._stage_packaged_examples(overlay)
        assert (packaged / "all_options.toml").is_file()
        assert (packaged / "native" / "runtime.cpp").is_file()
        assert (packaged / "all_options.toml").read_bytes() == (
            overlay / "examples" / "all_options.toml"
        ).read_bytes()
        with pytest.raises(RuntimeError, match="already contains"):
            backend._stage_packaged_examples(overlay)


def test_wheel_stages_the_single_maintained_rusticol_stub() -> None:
    with backend._overlay("release") as (overlay, _target):
        source = (
            overlay
            / "rust"
            / "crates"
            / "rusticol-python"
            / "stubs"
            / "pyamplicol"
            / "_rusticol.pyi"
        )
        target = overlay / "src" / "pyamplicol" / "_rusticol.pyi"
        assert source.is_file()
        assert not target.exists()
        backend._stage_python_stub(overlay)
        assert target.read_bytes() == source.read_bytes()
        with pytest.raises(RuntimeError, match="already contains"):
            backend._stage_python_stub(overlay)


def test_wheel_stages_runtime_resources_inside_package_namespace(
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "overlay"
    for relative in (
        Path("schemas/README.md"),
        Path("schemas/artifact-manifest-v3.schema.json"),
        Path("schemas/runtime-physics-v1.schema.json"),
    ):
        target = overlay / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())

    backend._stage_runtime_resources(overlay)
    package_assets = overlay / "src" / "pyamplicol" / "assets"
    assert not (package_assets / "release").exists()
    assert (
        package_assets / "schemas" / "artifact-manifest-v3.schema.json"
    ).read_bytes() == (
        overlay / "schemas" / "artifact-manifest-v3.schema.json"
    ).read_bytes()
    assert (package_assets / "schemas" / "runtime-physics-v1.schema.json").is_file()
    with pytest.raises(RuntimeError, match="already contains"):
        backend._stage_runtime_resources(overlay)


def test_overlay_rejects_symlinked_inputs(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("content", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(RuntimeError, match="symlinks"):
        backend._reject_symlinks(link)


def test_overlay_prunes_excluded_directories_before_symlink_walk(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    excluded = source / "checkouts"
    excluded.mkdir(parents=True)
    (excluded / "escape").symlink_to(tmp_path)
    backend._reject_symlinks(source)


def test_git_overlay_copies_only_tracked_files_and_prunes_ignored_symlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", source], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "Test"],
        check=True,
    )
    tracked = source / "build_backend" / "tracked.py"
    tracked.parent.mkdir()
    tracked.write_text("tracked = True\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(source), "add", "build_backend/tracked.py"],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "tracked"], check=True)
    (tracked.parent / "untracked.py").write_text("leak = True\n", encoding="utf-8")
    excluded = source / "dependencies" / "checkouts"
    excluded.mkdir(parents=True)
    (excluded / "escape").symlink_to(tmp_path)

    monkeypatch.setattr(backend, "ROOT", source)
    destination = tmp_path / "destination"
    backend._copy_allowlisted_source(destination)
    assert (destination / "build_backend" / "tracked.py").is_file()
    assert not (destination / "build_backend" / "untracked.py").exists()
    assert not (destination / "dependencies" / "checkouts").exists()


def test_archive_overlay_without_git_history_uses_pruned_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "archive"
    retained = {
        Path("build_backend/backend.py"): "archive = True\n",
        Path("docs/pyAmpliCol.tex"): "maintained TeX\n",
        Path("src/pyamplicol/_sdk/config.py"): "maintained SDK config\n",
        Path("tests/fixtures/candidate-Cargo.lock"): "fixture lock\n",
    }
    excluded = (
        Path("dependencies/candidate-Cargo.lock"),
        Path("dependencies/candidate-cargo-config.toml"),
        Path("dependencies/contributor-lock.toml"),
        Path("dependencies/install_dependencies.py"),
        Path("dependencies/install-state.json"),
        Path("dependencies/patches/dependency/fix.patch"),
        Path("dependencies/python-runtime-lock.toml"),
        Path("build_backend/python_lock.py"),
        Path("docs/.result_outputs/cache.json"),
        Path("docs/archive/retired.md"),
        Path("docs/pyAmpliCol.aux"),
        Path("docs/pyAmpliCol.synctex.gz"),
        Path("docs/pyAmpliCol.toc"),
        Path("src/pyamplicol.egg-info/PKG-INFO"),
        Path("src/pyamplicol/_rusticol.abi3.so"),
        Path("src/pyamplicol/_rusticol.dylib"),
        Path("src/pyamplicol/_rusticol.pyd"),
        Path("src/pyamplicol/_sdk/fortran/rusticol.f90"),
        Path("src/pyamplicol/_sdk/include/rusticol.h"),
        Path("src/pyamplicol/_sdk/lib/librusticol_capi.a"),
        Path("src/pyamplicol/_sdk/link.json"),
        Path("src/pyamplicol/_sdk/metadata.json"),
        Path("tests/.artifacts/build.json"),
        Path("tests/build/output.txt"),
        Path("tests/htmlcov/index.html"),
    )
    for relative, content in retained.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for relative in excluded:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated\n", encoding="utf-8")
    ignored = source / "build_backend" / "__pycache__"
    ignored.mkdir()
    (ignored / "escape").symlink_to(tmp_path)
    monkeypatch.setattr(backend, "ROOT", source)

    destination = tmp_path / "destination"
    backend._copy_allowlisted_source(destination)
    for relative, content in retained.items():
        assert (destination / relative).read_text(encoding="utf-8") == content
    for relative in excluded:
        assert not (destination / relative).exists()
    assert not (destination / "build_backend" / "__pycache__").exists()

    copy_ignore = backend._copy_ignore(source)
    assert copy_ignore(str(source / "docs"), ["pyAmpliCol.aux", "pyAmpliCol.tex"]) == {
        "pyAmpliCol.aux"
    }
    assert copy_ignore(
        str(source / "src" / "pyamplicol" / "_sdk"),
        ["config.py", "lib", "metadata.json"],
    ) == {"lib", "metadata.json"}


def test_candidate_digest_covers_only_contributor_dependency_inputs(
    tmp_path: Path,
) -> None:
    inputs = _candidate_inputs(tmp_path)
    first = backend._candidate_digest(*inputs)

    state = json.loads(inputs[2].read_text(encoding="utf-8"))
    state["sources"]["symbolica"]["worktree_sha256"] = "f" * 64
    inputs[2].write_text(json.dumps(state), encoding="utf-8")
    assert backend._candidate_digest(*inputs) == first

    state["sources"]["symbolica"]["revision"] = "0" * 40
    inputs[2].write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="contributor-lock revision"):
        backend._candidate_digest(*inputs)
    state["sources"]["symbolica"]["revision"] = tomllib.loads(
        (ROOT / "dependencies" / "contributor-lock.toml").read_text(encoding="utf-8")
    )["symbolica"]["candidate_revision"]
    inputs[2].write_text(json.dumps(state), encoding="utf-8")

    inputs[0].write_bytes(inputs[0].read_bytes() + b"\n# identity change\n")
    lock_changed = backend._candidate_digest(*inputs)
    assert lock_changed != first

    inputs[1].write_text("[patch.crates-io]\n# changed\n", encoding="utf-8")
    assert backend._candidate_digest(*inputs) not in {first, lock_changed}


@pytest.mark.parametrize(
    ("hook", "arguments", "expected", "with_sdk"),
    [
        ("build_wheel", ("wheel",), "delegated", True),
        ("build_sdist", ("sdist",), "delegated", False),
        ("get_requires_for_build_wheel", (), ["delegated"], False),
        ("get_requires_for_build_sdist", (), ["delegated"], False),
        ("prepare_metadata_for_build_wheel", ("metadata",), "delegated", False),
    ],
)
def test_retained_pep517_hooks_use_gate_overlay_and_clean_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hook: str,
    arguments: tuple[str, ...],
    expected: str | list[str],
    with_sdk: bool,
) -> None:
    # This test exercises the publishable hook contract independently of the
    # candidate mode used by the enclosing contributor source gate.
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "release")
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    target = tmp_path / "cargo-target"
    gates: list[str] = []
    sdk_stages: list[Path] = []
    selftest_stages: list[tuple[Path, str]] = []
    injected_bins = (tmp_path / "injected-bin", tmp_path / "injected-tools")
    for directory in injected_bins:
        directory.mkdir()
    injected = {
        "CARGO": "/attacker/cargo",
        "CARGO_BUILD_TARGET": "attacker-target",
        "CPATH": "/opt/local/include",
        "DYLD_LIBRARY_PATH": "/opt/local/lib",
        "GIT_INDEX_FILE": "/attacker/index",
        "LIBRARY_PATH": "/opt/local/lib",
        "MATURIN_PEP517_ARGS": "--target attacker-target",
        "PYO3_PYTHON": "/attacker/python",
        "PYTHONPATH": "/attacker/pythonpath",
        "RUSTC_WRAPPER": "/attacker/wrapper",
        "RUSTFLAGS": "-Clink-arg=/attacker/library",
    }
    for name, value in injected.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(str(path) for path in injected_bins)
        + os.pathsep
        + os.environ["PATH"],
    )

    @contextlib.contextmanager
    def fake_overlay(mode: str):
        assert mode == "release"
        yield overlay, target

    def fake_sdk(root: Path, target_dir: Path) -> Path:
        assert root == overlay and target_dir == target
        assert not set(injected) & set(os.environ)
        assert not {str(path) for path in injected_bins} & set(
            os.environ["PATH"].split(os.pathsep)
        )
        assert "CARGO_ENCODED_RUSTFLAGS" in os.environ
        staging = tmp_path / "sdk"
        staging.mkdir()
        (staging / "metadata.json").write_text(
            json.dumps({"target": "aarch64-apple-darwin"}), encoding="utf-8"
        )
        sdk_stages.append(staging)
        return staging

    def delegated(*_args, **_kwargs):
        assert Path.cwd() == overlay
        assert not set(injected) & set(os.environ)
        assert not {str(path) for path in injected_bins} & set(
            os.environ["PATH"].split(os.pathsep)
        )
        remaps = os.environ["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
        assert len(remaps) == 4
        assert all(flag.startswith("--remap-path-prefix=") for flag in remaps)
        assert os.environ["CARGO_HOME"] == str(tmp_path / "cargo-home")
        assert os.environ["CARGO_TARGET_DIR"] == str(target)
        assert os.environ["PYAMPLICOL_BUILD_OVERLAY"] == str(overlay)
        if sys.platform == "darwin":
            assert os.environ["MACOSX_DEPLOYMENT_TARGET"] == "11.0"
        if with_sdk:
            assert os.environ["PYAMPLICOL_SDK_STAGING"] == str(tmp_path / "sdk")
        return ["delegated"] if hook.startswith("get_requires") else "delegated"

    monkeypatch.setattr(backend, "_overlay", fake_overlay)
    monkeypatch.setattr(backend, "_check_dependencies", gates.append)
    monkeypatch.setattr(backend, "_stage_packaged_examples", lambda path: None)
    monkeypatch.setattr(backend, "_stage_python_stub", lambda path: None)
    monkeypatch.setattr(backend, "_stage_runtime_resources", lambda path: None)
    monkeypatch.setattr(
        backend,
        "_stage_selftest_fixture",
        lambda path, target_name: selftest_stages.append((path, target_name)),
    )
    monkeypatch.setattr(backend, "build_sdk", fake_sdk)
    monkeypatch.setattr(backend.maturin, hook, delegated)

    result = getattr(backend, hook)(*arguments)
    assert result == expected
    assert gates == ["release"]
    assert bool(sdk_stages) is with_sdk
    assert selftest_stages == ([(overlay, "aarch64-apple-darwin")] if with_sdk else [])
    for name, value in injected.items():
        assert os.environ[name] == value


def test_backend_does_not_advertise_pep660_hooks() -> None:
    editable_hooks = (
        "build_editable",
        "get_requires_for_build_editable",
        "prepare_metadata_for_build_editable",
    )
    assert not [name for name in editable_hooks if hasattr(backend, name)]


def test_sdk_build_stages_the_safe_rust_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"static archive")
    link = {
        "schema_version": 1,
        "target": "aarch64-apple-darwin",
        "system_libraries": ["System"],
        "frameworks": [],
    }

    monkeypatch.setattr(sdk, "_host_target", lambda _root: link["target"])
    monkeypatch.setattr(sdk, "_requested_target", lambda host: host)
    monkeypatch.setattr(sdk, "_cargo_fetch", lambda *_args: None)
    monkeypatch.setattr(
        sdk.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=(), returncode=0, stdout="", stderr=""
        ),
    )
    monkeypatch.setattr(sdk, "_cargo_messages", lambda _stdout: ())
    monkeypatch.setattr(sdk, "_static_library", lambda _messages: archive)
    monkeypatch.setattr(sdk, "_native_tokens", lambda _stderr: ())
    monkeypatch.setattr(sdk, "_typed_link_arguments", lambda _tokens, _target: link)
    monkeypatch.setattr(sdk, "_scan_archive", lambda _archive: None)
    monkeypatch.setattr(sdk, "_scan_archive_symbols", lambda _archive: None)
    monkeypatch.setattr(
        sdk, "_validate_archive_linkage", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(sdk, "_package_version", lambda _root: "0.1.0")

    staging = sdk.build_sdk(ROOT, tmp_path / "target")

    rust_source = staging / "rust" / "rusticol.rs"
    assert rust_source.read_bytes() == (ROOT / sdk.RUST_SDK_SOURCE).read_bytes()
    metadata = json.loads((staging / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["rust_source"] == "rust/rusticol.rs"


def test_dependency_gate_subprocess_ignores_inherited_build_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CARGO", "/attacker/cargo")
    monkeypatch.setenv("PYTHONPATH", "/attacker/pythonpath")
    monkeypatch.setenv("RUSTFLAGS", "-Clink-arg=/attacker/library")
    observed: dict[str, object] = {}

    def fake_run(command, *, cwd, env, check):
        observed.update(command=command, cwd=cwd, env=env, check=check)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(backend.subprocess, "run", fake_run)
    backend._check_dependencies("candidate")
    command = observed["command"]
    environment = observed["env"]
    assert isinstance(command, list) and command[1] == "-I"
    assert command[-1] == "--candidate"
    assert isinstance(environment, dict)
    assert not {"CARGO", "PYTHONPATH", "RUSTFLAGS"} & set(environment)


def test_build_tool_path_retains_isolated_interpreter_bin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment_bin = tmp_path / "build-env" / "bin"
    environment_bin.mkdir(parents=True)
    interpreter = environment_bin / "python"
    interpreter.symlink_to(Path(sys.executable).resolve())
    tool_bin = tmp_path / "rust" / "bin"
    tool_bin.mkdir(parents=True)

    monkeypatch.setattr(backend.sys, "executable", str(interpreter))
    monkeypatch.setattr(
        backend.shutil,
        "which",
        lambda executable, *, path: str(tool_bin / executable),
    )

    result = backend._build_tool_path("").split(os.pathsep)

    assert str(environment_bin) in result


def test_build_tool_path_does_not_require_git_for_unpacked_sdist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool_bin = tmp_path / "tools"
    tool_bin.mkdir()

    def locate(executable: str, *, path: str) -> str | None:
        del path
        if executable == "git":
            return None
        if executable in {"cargo", "rustc"}:
            return str(tool_bin / executable)
        return None

    monkeypatch.setattr(backend.shutil, "which", locate)

    result = backend._build_tool_path("").split(os.pathsep)

    assert str(tool_bin) in result


def test_build_tool_path_does_not_expose_base_python_package_manager_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment_bin = tmp_path / "build-env" / "bin"
    environment_bin.mkdir(parents=True)
    interpreter = environment_bin / "python"
    interpreter.touch()
    rust_bin = tmp_path / "rustup" / "bin"
    rust_bin.mkdir(parents=True)

    def locate(executable: str, *, path: str) -> str | None:
        if executable in {"cargo", "rustc"}:
            return str(rust_bin / executable)
        assert path.startswith("/usr/bin" + os.pathsep)
        return f"/usr/bin/{executable}"

    monkeypatch.setattr(backend.sys, "executable", str(interpreter))
    monkeypatch.setattr(backend.shutil, "which", locate)

    result = backend._build_tool_path(
        "/opt/local/bin:/opt/homebrew/bin:/usr/local/bin"
    ).split(os.pathsep)

    assert str(environment_bin) in result
    assert str(rust_bin) in result
    assert "/usr/bin" in result
    assert "/opt/local/bin" not in result
    assert "/opt/homebrew/bin" not in result
    assert "/usr/local/bin" not in result


def test_pep517_backend_rejects_recursive_delegation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "release")
    overlay = tmp_path / "overlay"
    overlay.mkdir()

    @contextlib.contextmanager
    def fake_overlay(_mode: str):
        yield overlay, tmp_path / "target"

    monkeypatch.setattr(backend, "_overlay", fake_overlay)
    monkeypatch.setattr(backend, "_check_dependencies", lambda _mode: None)
    monkeypatch.setattr(
        backend.maturin,
        "build_sdist",
        lambda *_args: backend.get_requires_for_build_sdist(),
    )
    with pytest.raises(RuntimeError, match="recursive PEP 517"):
        backend.build_sdist(str(tmp_path / "dist"))


def test_native_link_arguments_are_typed_and_allowlisted() -> None:
    macos = sdk._typed_link_arguments(
        ["-lgcc_s", "-lSystem", "-lc", "-framework", "Security"],
        "aarch64-apple-darwin",
    )
    assert "gcc_s" in macos["system_libraries"]
    assert macos["system_libraries"] == ["gcc_s", "System", "c"]
    assert macos["frameworks"] == ["Security"]

    linux = sdk._typed_link_arguments(
        ["-lgcc_s", "-lutil", "-lrt", "-lpthread", "-lm", "-ldl", "-lc"],
        "x86_64-unknown-linux-gnu",
    )
    assert linux["frameworks"] == []
    assert "gcc_s" in linux["system_libraries"]


def test_static_archive_byte_scan_rejects_python_markers(tmp_path: Path) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"archive rusticol_runtime_load native-static-libs")
    sdk._scan_archive(archive)

    archive.write_bytes(b"archive rusticol_runtime_load PyGILState_Ensure")
    with pytest.raises(RuntimeError, match="PyGILState"):
        sdk._scan_archive(archive)


@pytest.mark.parametrize(
    "symbol",
    ["_PyErr_SetString", "PyLong_FromLong", "_Py_Dealloc"],
)
def test_static_archive_symbol_scan_rejects_python_symbols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(sdk.shutil, "which", lambda _name: "/usr/bin/nm")
    monkeypatch.setattr(
        sdk.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["nm"], 0, stdout=f"                 U {symbol}\n", stderr=""
        ),
    )

    with pytest.raises(RuntimeError, match="undefined Python"):
        sdk._scan_archive_symbols(archive)


def test_static_archive_symbol_scan_accepts_non_python_symbols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(sdk.shutil, "which", lambda _name: "/usr/bin/nm")
    monkeypatch.setattr(
        sdk.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["nm"], 0, stdout="                 U _malloc\n", stderr=""
        ),
    )

    assert sdk._scan_archive_symbols(archive)


def test_static_archive_symbol_scan_defers_incompatible_llvm_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"archive")
    monkeypatch.delenv("LLVM_NM", raising=False)
    monkeypatch.setattr(sdk.shutil, "which", lambda _name: "/usr/bin/nm")
    monkeypatch.setattr(
        sdk.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["nm"],
            1,
            stdout="",
            stderr=(
                "Unknown attribute kind (91) "
                "(Producer: 'LLVM20.1.7' Reader: 'LLVM APPLE_1_1600')"
            ),
        ),
    )

    assert not sdk._scan_archive_symbols(archive)


@pytest.mark.skipif(os.name == "nt", reason="native SDK targets macOS and Linux")
def test_static_archive_link_probe_rejects_missing_public_export(
    tmp_path: Path,
) -> None:
    target = (
        "aarch64-apple-darwin"
        if sys.platform == "darwin"
        else "x86_64-unknown-linux-gnu"
    )
    try:
        compiler = sdk._native_c_compiler(target)
    except RuntimeError as error:
        pytest.skip(str(error))
    archiver = shutil.which("ar")
    if archiver is None:
        pytest.skip("ar is unavailable")

    header = tmp_path / "rusticol.h"
    header.write_text(
        """\
#include <stdint.h>
#define RUSTICOL_ABI_VERSION 1u
uint32_t rusticol_abi_version(void);
int rusticol_required_export(void);
""",
        encoding="utf-8",
    )
    implementation = tmp_path / "partial.c"
    implementation.write_text(
        "#include <stdint.h>\nuint32_t rusticol_abi_version(void) { return 1u; }\n",
        encoding="utf-8",
    )
    object_file = tmp_path / "partial.o"
    archive = tmp_path / "librusticol_capi.a"
    subprocess.run(
        [*compiler, "-c", str(implementation), "-o", str(object_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [archiver, "rcs", str(archive), str(object_file)],
        check=True,
        capture_output=True,
        text=True,
    )

    with pytest.raises(RuntimeError, match="complete C ABI link probe"):
        sdk._validate_archive_linkage(
            archive,
            header=header,
            link={"target": target, "system_libraries": [], "frameworks": []},
            target=target,
        )


@pytest.mark.parametrize(
    "token",
    [
        "-L/opt/local/lib",
        "-Wl,-rpath,/opt/local/lib",
        "/usr/local/lib/libfoo.a",
        "-lnot_allowlisted",
    ],
)
def test_native_link_arguments_reject_nonportable_tokens(token: str) -> None:
    with pytest.raises(RuntimeError):
        sdk._typed_link_arguments([token], "aarch64-apple-darwin")
