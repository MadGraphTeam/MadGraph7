# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "dependencies" / "install_dependencies.py"


def _module():
    spec = importlib.util.spec_from_file_location("dependency_installer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_inventory_is_exact_and_legacy_is_optional() -> None:
    module = _module()
    payload = module._lock()
    with_legacy = module._sources(payload, with_legacy=True)
    without_legacy = module._sources(payload, with_legacy=False)

    assert {item.key for item in without_legacy} == {
        "symjit",
        "symbolica",
        "symbolica-community",
        "gammaloop",
        "ratatui-ffi",
    }
    assert {item.key for item in with_legacy} == {
        *(item.key for item in without_legacy),
        "legacy-amplicol",
    }
    assert all(len(item.revision) == 40 for item in with_legacy)
    legacy = next(item for item in with_legacy if item.key == "legacy-amplicol")
    assert legacy.branch == payload["legacy_amplicol"]["branch"]
    assert legacy.revision == payload["legacy_amplicol"]["revision"]


def test_ratatui_distribution_and_ffi_source_are_exactly_pinned() -> None:
    module = _module()
    ratatui = module._lock()["ratatui"]

    assert ratatui["distribution"] == "ratatui"
    assert ratatui["version"] == "0.4.2"
    assert ratatui["sdist_sha256"] == (
        "2778b066378f8e1629b4b5e1076f1957c8c439b45cdeaf51970d1949730bb0d5"
    )
    assert ratatui["ffi_revision"] == ("7249c0bd1445c0c6ee76f3f24923eb35a1d931e0")
    ffi_source = next(
        source
        for source in module._sources(module._lock(), with_legacy=False)
        if source.key == "ratatui-ffi"
    )
    assert ffi_source.url == ratatui["ffi_repository"]
    assert ffi_source.revision == ratatui["ffi_revision"]


def test_ufo_loader_uses_the_verified_published_wheel_without_local_patch() -> None:
    module = _module()
    payload = module._lock()
    loader = payload["ufo_model_loader"]
    assert loader["required_version"] == "0.1.7"
    assert loader["latest_verified_published_version"] == "0.1.7"
    assert loader["published_revision"] == ("f3fda32c5e6a673075c345d74a11f12b83c00015")
    assert loader["wheel_sha256"] == (
        "803ae28141ec4be3189cc62469b88da17ca33907791fe99774c2fe756a45edf7"
    )
    assert loader["release_status"] == "verified"
    assert "patches" not in payload


def test_legacy_oracle_uses_the_pinned_remote_branch_without_local_patches() -> None:
    module = _module()
    payload = module._lock()
    assert "patches" not in payload
    assert not tuple(
        (module.DEPENDENCIES / "patches" / "legacy-amplicol").glob("*.patch")
    )


def test_venv_reset_bootstraps_with_the_unmoved_base_interpreter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    base_python = tmp_path / "base-python"
    active_venv_python = tmp_path / ".venv" / "bin" / "python"
    monkeypatch.setattr(module.sys, "_base_executable", str(base_python))
    monkeypatch.setattr(module.sys, "executable", str(active_venv_python))

    assert module._venv_bootstrap_python() == base_python


def test_official_symjit_git_revision_is_pinned_without_local_patches() -> None:
    module = _module()
    payload = module._lock()
    symjit = payload["symjit"]

    assert symjit == {
        "version": "2.22.0",
        "repository": "https://github.com/siravan/symjit-crate.git",
        "revision": "d8abfeeb4db98c13cdcf9dd39cf3e795fd5001a7",
    }
    source = next(
        item
        for item in module._sources(payload, with_legacy=False)
        if item.key == "symjit"
    )
    assert (source.url, source.revision, source.branch) == (
        symjit["repository"],
        symjit["revision"],
        None,
    )
    assert "patches" not in payload
    assert not tuple((module.DEPENDENCIES / "patches" / "symjit").rglob("*.patch"))


def test_symjit_checkout_uses_exact_git_revision_and_accepts_matching_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    checkouts = tmp_path / "workspace" / "dependencies" / "checkouts"
    monkeypatch.setattr(module, "CHECKOUTS", checkouts)
    source = module.Source(
        "symjit",
        "https://github.com/siravan/symjit-crate.git",
        "d8abfeeb4db98c13cdcf9dd39cf3e795fd5001a7",
    )
    calls: list[tuple[list[str], Path | None, bool]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, cwd=None, capture=False, **_kwargs):
            rendered = [str(item) for item in command]
            calls.append((rendered, cwd, capture))
            return subprocess.CompletedProcess(
                command,
                0,
                source.revision + "\n" if rendered[1:] == ["rev-parse", "HEAD"] else "",
                "",
            )

    module._checkout(FakeRunner(), source, update=False)
    destination = checkouts / "symjit"
    assert calls == [
        (
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                source.url,
                str(destination),
            ],
            None,
            False,
        ),
        (["git", "checkout", "--detach", source.revision], destination, False),
    ]

    destination.mkdir(parents=True)
    (destination / ".git").mkdir()
    calls.clear()
    module._checkout(FakeRunner(), source, update=False)
    assert calls == [(["git", "rev-parse", "HEAD"], destination, True)]


def test_compact_install_state_records_only_git_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    checkouts = tmp_path / "workspace" / "dependencies" / "checkouts"
    state_path = checkouts.parent / "install-state.json"
    sources = (
        module.Source("symjit", "https://example.invalid/symjit.git", "1" * 40),
        module.Source(
            "legacy-amplicol",
            "https://example.invalid/legacy.git",
            "2" * 40,
            "stable",
        ),
    )
    for source in sources:
        (checkouts / source.key).mkdir(parents=True)
    monkeypatch.setattr(module, "CHECKOUTS", checkouts)
    monkeypatch.setattr(module, "STATE", state_path)

    class FakeRunner:
        dry_run = False

        def run(self, command, *, cwd=None, capture=False, **_kwargs):
            assert command == ["git", "rev-parse", "HEAD"]
            assert cwd is not None and capture is True
            revision = next(item.revision for item in sources if item.key == cwd.name)
            return subprocess.CompletedProcess(command, 0, revision + "\n", "")

    module._write_state(FakeRunner(), sources)

    assert module.json.loads(state_path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "publishable": False,
        "sources": {
            "symjit": {
                "url": "https://example.invalid/symjit.git",
                "revision": "1" * 40,
            },
            "legacy-amplicol": {
                "url": "https://example.invalid/legacy.git",
                "revision": "2" * 40,
                "branch": "stable",
            },
        },
    }


@pytest.mark.parametrize(
    "checkout_name",
    ("symbolica", "symbolica-community", "gammaloop"),
)
def test_non_symjit_checkout_symlink_outside_workspace_is_rejected_before_write(
    checkout_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    checkouts = tmp_path / "workspace" / "dependencies" / "checkouts"
    checkouts.mkdir(parents=True)
    outside = tmp_path / f"outside-{checkout_name}"
    outside.mkdir()
    marker = outside / "Cargo.toml"
    marker.write_text("outside checkout must remain unchanged\n", encoding="utf-8")
    (checkouts / checkout_name).symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(module, "CHECKOUTS", checkouts)
    source = module.Source(
        checkout_name,
        "https://example.invalid/source.git",
        "1" * 40,
    )
    calls: list[list[str]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, **_kwargs):
            calls.append([str(item) for item in command])
            return subprocess.CompletedProcess(command, 0, "", "")

    runner = FakeRunner()
    before = marker.read_bytes()
    operations = (
        lambda: module._checkout(runner, source, update=False),
        lambda: module._configure_source_manifests(runner),
    )

    for operation in operations:
        with pytest.raises(module.SetupError, match="must not be a symbolic link"):
            operation()

    assert calls == []
    assert marker.read_bytes() == before
    assert tuple(outside.iterdir()) == (marker,)


def test_legacy_checkout_clones_the_named_branch_then_pins_its_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "CHECKOUTS", tmp_path / "checkouts")
    source = module.Source(
        "legacy-amplicol",
        "https://github.com/rikkert-frederix/AmpliCol.git",
        "60443f327c2203cf92625da2bf0969c27e68a4ac",
        "amplicol_with_patches",
    )
    calls: list[tuple[list[str], Path | None]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, cwd=None, **_kwargs):
            calls.append(([str(item) for item in command], cwd))
            return subprocess.CompletedProcess(command, 0, "", "")

    module._checkout(FakeRunner(), source, update=False)

    assert calls == [
        (
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--branch",
                "amplicol_with_patches",
                "--single-branch",
                "--no-checkout",
                "https://github.com/rikkert-frederix/AmpliCol.git",
                str(source.path),
            ],
            None,
        ),
        (
            [
                "git",
                "checkout",
                "--detach",
                "60443f327c2203cf92625da2bf0969c27e68a4ac",
            ],
            source.path,
        ),
    ]


def test_contributor_runtime_requirements_use_the_full_hash_locked_closure() -> None:
    module = _module()
    requirements = module._runtime_requirements_text()
    assert "symbolica==" not in requirements
    for requirement in (
        "colorama==0.4.6",
        "numpy==2.4.2",
        "prettytable==3.18.0",
        "progressbar2==4.5.0",
        "python-utils==4.0.0",
        "typing-extensions==4.16.0",
        "ufo-model-loader==0.1.7",
        "wcwidth==0.8.2",
    ):
        assert requirements.count(requirement) == 1
    assert requirements.count("--hash=sha256:") > 20


def test_contributor_tools_reuse_project_build_test_and_docs_requirements() -> None:
    module = _module()
    with module.PYPROJECT.open("rb") as stream:
        project_file = module.tomllib.load(stream)
    project = project_file["project"]
    optional = project["optional-dependencies"]
    expected = tuple(
        dict.fromkeys(
            (
                *project_file["build-system"]["requires"],
                *optional["test"],
                *optional["docs"],
            )
        )
    )

    requirements = module._contributor_python_requirements()

    assert requirements == expected
    assert requirements.count("maturin==1.14.1") == 1
    assert "pypdf>=5,<6" in requirements


def test_candidate_dependency_only_build_installs_and_verifies_symbolica(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    venv = tmp_path / ".venv"
    checkouts = tmp_path / "checkouts"
    wheelhouse = tmp_path / "wheelhouse"
    symbolica_wheels = wheelhouse / "symbolica"
    symbolica_wheels.mkdir(parents=True)
    wheel = symbolica_wheels / "symbolica-2.2.0-test.whl"
    wheel.touch()
    calls: list[tuple[list[str], Path | None, dict[str, str] | None]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, cwd=None, env=None, **_kwargs):
            rendered = [str(item) for item in command]
            calls.append((rendered, cwd, env))
            return subprocess.CompletedProcess(rendered, 0, "", "")

    monkeypatch.setattr(module, "VENV", venv)
    monkeypatch.setattr(module, "CHECKOUTS", checkouts)
    monkeypatch.setattr(module, "WHEELHOUSE", wheelhouse)
    ratatui_payloads: list[dict[str, object]] = []
    monkeypatch.setattr(
        module,
        "_build_ratatui_wheel",
        lambda _runner, payload: ratatui_payloads.append(payload),
    )

    payload: dict[str, object] = {"symbolica": {"candidate_version": "2.2.0"}}
    module._build_candidate_dependency_wheels(
        FakeRunner(),
        payload,
    )

    python = str(venv / "bin" / "python")
    assert calls[0][0][:4] == [python, "-m", "maturin", "build"]
    assert calls[0][1] == checkouts / "symbolica-community"
    assert calls[1][0] == [
        python,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        str(wheel),
    ]
    assert calls[2][0][:3] == [python, "-I", "-c"]
    assert calls[2][0][-1] == "2.2.0"
    assert "from symbolica import Expression" in calls[2][0][3]
    assert "from symbolica.community.idenso import simplify_color" in calls[2][0][3]
    assert "from symbolica.community.spenso import TensorNetwork" in calls[2][0][3]
    assert calls[2][2]["SYMBOLICA_HIDE_BANNER"] == "1"
    assert ratatui_payloads == [payload]


def test_ratatui_build_uses_verified_sdist_and_pinned_local_ffi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    venv = tmp_path / ".venv"
    checkouts = tmp_path / "checkouts"
    wheelhouse = tmp_path / "wheelhouse"
    ffi_source = checkouts / "ratatui-ffi"
    ffi_source.mkdir(parents=True)
    (ffi_source / "Cargo.toml").write_text(
        '[package]\nname = "ratatui-ffi"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    sdist = wheelhouse / "ratatui" / "ratatui-0.4.2.tar.gz"
    sdist.parent.mkdir(parents=True)
    sdist.touch()
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, env=None, **_kwargs):
            rendered = [str(item) for item in command]
            calls.append((rendered, env))
            if rendered[1:4] == ["-m", "pip", "wheel"]:
                (wheelhouse / "ratatui" / "ratatui-0.4.2-test.whl").touch()
            return subprocess.CompletedProcess(rendered, 0, "", "")

    payload = {
        "ratatui": {
            "version": "0.4.2",
            "ffi_revision": "7249c0bd1445c0c6ee76f3f24923eb35a1d931e0",
        }
    }
    monkeypatch.setattr(module, "VENV", venv)
    monkeypatch.setattr(module, "CHECKOUTS", checkouts)
    monkeypatch.setattr(module, "WHEELHOUSE", wheelhouse)
    monkeypatch.setattr(
        module,
        "_materialize_ratatui_sdist",
        lambda _runner, _payload: sdist,
    )
    monkeypatch.setattr(module, "_archive_candidate_wheels", lambda *_args: None)

    module._build_ratatui_wheel(FakeRunner(), payload)

    build, build_environment = calls[0]
    assert build == [
        str(venv / "bin" / "python"),
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        str(wheelhouse / "ratatui"),
        str(sdist),
    ]
    assert build_environment is not None
    assert Path(build_environment["RATATUI_FFI_SRC"]).name == "ratatui-ffi"
    assert Path(build_environment["RATATUI_FFI_SRC"]) != ffi_source.resolve()
    assert build_environment["RATATUI_FFI_TAG"] == payload["ratatui"]["ffi_revision"]
    assert calls[1][0][1:4] == ["-m", "pip", "install"]
    assert calls[2][0][:3] == [str(venv / "bin" / "python"), "-I", "-c"]
    assert "import ratatui" in calls[2][0][3]
    assert "import ratatui_py" in calls[2][0][3]


def test_ratatui_sdist_materialization_rejects_unpinned_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    wheelhouse = tmp_path / "wheelhouse"
    archive = b"not the pinned distribution"
    payload = {
        "ratatui": {
            "version": "0.4.2",
            "source_url": "https://example.invalid/ratatui-0.4.2.tar.gz",
            "sdist_sha256": hashlib.sha256(b"expected distribution").hexdigest(),
        }
    }
    monkeypatch.setattr(module, "WHEELHOUSE", wheelhouse)
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda _url: io.BytesIO(archive),
    )

    with pytest.raises(module.SetupError, match="sdist digest mismatch"):
        module._materialize_ratatui_sdist(module.Runner(dry_run=False), payload)

    assert not (wheelhouse / "ratatui" / "ratatui-0.4.2.tar.gz").exists()


@pytest.mark.parametrize("explicit_bootstrap", [None, "1"])
def test_candidate_project_wheel_does_not_force_asset_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    explicit_bootstrap: str | None,
) -> None:
    module = _module()
    venv = tmp_path / ".venv"
    artifacts = tmp_path / "artifacts"
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, env=None, **_kwargs):
            rendered = [str(item) for item in command]
            calls.append((rendered, env))
            if rendered[1:4] == ["-m", "build", "--wheel"]:
                artifacts.mkdir(parents=True, exist_ok=True)
                (artifacts / "pyamplicol-test.whl").touch()
            return subprocess.CompletedProcess(rendered, 0, "", "")

    if explicit_bootstrap is None:
        monkeypatch.delenv("PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP", raising=False)
    else:
        monkeypatch.setenv(
            "PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP",
            explicit_bootstrap,
        )
    monkeypatch.setattr(module, "VENV", venv)
    monkeypatch.setattr(module, "ARTIFACTS", artifacts)

    module._build_candidate_project_wheel(FakeRunner())

    assert len(calls) == 2
    build_command, build_environment = calls[0]
    assert build_command[1:4] == ["-m", "build", "--wheel"]
    assert build_environment is not None
    assert build_environment["PYAMPLICOL_BUILD_MODE"] == "candidate"
    install_command, install_environment = calls[1]
    assert install_command[1:4] == ["-m", "pip", "install"]
    assert install_environment is not None
    if explicit_bootstrap is None:
        assert "PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP" not in build_environment
        assert "PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP" not in install_environment
        assert "PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP" not in os.environ
    else:
        assert (
            build_environment["PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP"]
            == explicit_bootstrap
        )
        assert (
            install_environment["PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP"]
            == explicit_bootstrap
        )
        assert os.environ["PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP"] == explicit_bootstrap


def test_dependency_only_and_no_build_are_mutually_exclusive() -> None:
    module = _module()
    with pytest.raises(SystemExit):
        module._parser().parse_args(["--dependencies-only", "--no-build"])


def test_dependency_only_cli_skips_the_project_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    payload: dict[str, object] = {}
    monkeypatch.setattr(module, "_lock", lambda: payload)
    monkeypatch.setattr(module, "_sources", lambda *_args, **_kwargs: ())
    for name in (
        "_ensure_just",
        "_ensure_venv",
        "_configure_sources",
        "_write_cargo_config",
        "_write_candidate_lock",
        "_write_state",
    ):
        monkeypatch.setattr(module, name, lambda *_args, **_kwargs: None)
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "_build_candidate_dependency_wheels",
        lambda *_args: calls.append("dependencies"),
    )
    monkeypatch.setattr(
        module,
        "_build_candidate_wheels",
        lambda *_args: calls.append("dependencies-and-project"),
    )

    assert module.main(["--dry-run", "--dependencies-only"]) == 0
    assert calls == ["dependencies"]


def test_toml_section_replacement_is_idempotent() -> None:
    module = _module()
    original = '[package]\nname = "x"\n\n[dependencies]\na = "1"\n'
    once = module._replace_section(original, "dependencies", 'b = "2"')
    twice = module._replace_section(once, "dependencies", 'b = "2"')
    assert once == twice
    assert 'a = "1"' not in once
    assert once.count('b = "2"') == 1


def test_symjit_rlib_manifest_validation_does_not_rewrite_source(
    tmp_path: Path,
) -> None:
    module = _module()
    manifest = tmp_path / "Cargo.toml"
    pristine = (
        b'[package]\nname = "symjit"\nversion = "2.22.0"\n\n'
        b'[lib]\ncrate-type = ["rlib"]\n'
    )
    manifest.write_bytes(pristine)

    module._require_symjit_rlib_manifest(manifest, expected_version="2.22.0")

    assert manifest.read_bytes() == pristine

    manifest.write_text(
        '[package]\nname = "symjit"\nversion = "2.22.0"\n\n'
        '[lib]\ncrate-type = ["rlib", "cdylib"]\n',
        encoding="utf-8",
    )
    with pytest.raises(module.SetupError, match="rlib-only"):
        module._require_symjit_rlib_manifest(manifest)

    manifest.write_text(
        '[package]\nname = "symjit"\nversion = "2.21.0"\n\n'
        '[lib]\ncrate-type = ["rlib"]\n',
        encoding="utf-8",
    )
    with pytest.raises(module.SetupError, match="wrong package version"):
        module._require_symjit_rlib_manifest(
            manifest,
            expected_version="2.22.0",
        )


def test_candidate_community_lock_is_resolved_from_the_upstream_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    community = tmp_path / "checkouts" / "symbolica-community"
    community.mkdir(parents=True)
    lock = community / "Cargo.lock"
    lock.write_text("stale generated lock\n", encoding="utf-8")
    calls: list[list[str]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, cwd=None, capture=False, **_kwargs):
            rendered = [str(item) for item in command]
            calls.append(rendered)
            assert cwd == community
            assert capture is True
            if rendered[:2] == ["git", "show"]:
                return subprocess.CompletedProcess(command, 0, "upstream lock\n", "")
            if "--locked" not in rendered:
                assert lock.read_text(encoding="utf-8") == "upstream lock\n"
                lock.write_text("path-resolved lock\n", encoding="utf-8")
            else:
                assert lock.read_text(encoding="utf-8") == "path-resolved lock\n"
            return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(module, "CHECKOUTS", tmp_path / "checkouts")
    monkeypatch.setattr(module, "_configure_source_manifests", lambda _runner: None)

    module._configure_sources(FakeRunner())

    assert calls == [
        ["git", "show", "HEAD:Cargo.lock"],
        ["cargo", "metadata", "--format-version", "1"],
        ["cargo", "metadata", "--locked", "--format-version", "1"],
    ]
    assert lock.read_text(encoding="utf-8") == "path-resolved lock\n"


def test_canonical_release_lock_has_no_candidate_path_packages() -> None:
    module = _module()
    module._validate_release_cargo_lock(ROOT / "Cargo.lock")


def test_candidate_lock_is_seeded_without_mutating_canonical_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    project = tmp_path / "project"
    project.mkdir()
    (project / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
    release_lock = project / "Cargo.lock"
    release_lock.write_bytes(b"canonical release lock\n")
    (project / "rust").mkdir()
    cargo_config = project / ".cargo" / "config.toml"
    cargo_config.parent.mkdir()
    cargo_config.write_text("[patch.crates-io]\n", encoding="utf-8")
    candidate_lock = project / "dependencies" / "candidate-Cargo.lock"
    candidate_lock.parent.mkdir()
    calls: list[list[str]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, cwd=None, capture=False, **_kwargs):
            assert cwd is not None and capture is True
            calls.append(list(command))
            staged_lock = Path(cwd) / "Cargo.lock"
            if len(calls) == 1:
                assert staged_lock.read_bytes() == b"canonical release lock\n"
            elif len(calls) == 2:
                assert staged_lock.read_bytes() == b"canonical release lock\n"
                staged_lock.write_bytes(b"candidate path lock\n")
            else:
                assert staged_lock.read_bytes() == b"candidate path lock\n"
            return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(module, "ROOT", project)
    monkeypatch.setattr(module, "CARGO_CONFIG", cargo_config)
    monkeypatch.setattr(module, "CANDIDATE_LOCK", candidate_lock)
    monkeypatch.setattr(module, "_validate_release_cargo_lock", lambda _path: None)
    monkeypatch.setattr(module, "_validate_candidate_cargo_lock", lambda _path: None)
    projected: list[Path] = []
    monkeypatch.setattr(
        module,
        "_rewrite_candidate_requirements",
        lambda root: projected.append(root),
    )

    module._write_candidate_lock(FakeRunner())

    assert calls == [
        ["cargo", "metadata", "--locked", "--format-version", "1"],
        ["cargo", "metadata", "--format-version", "1"],
        ["cargo", "metadata", "--locked", "--format-version", "1"],
    ]
    assert release_lock.read_bytes() == b"canonical release lock\n"
    assert candidate_lock.read_bytes() == b"candidate path lock\n"
    assert len(projected) == 1
    assert projected[0].parent != project


def test_candidate_dependency_projection_rewrites_only_the_isolated_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    manifest = tmp_path / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        'symbolica = { version = "=2.1.0", default-features = false }\n'
        'symjit = { version = "=2.22.0", default-features = false }\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_lock",
        lambda: {
            "symbolica": {
                "rust_version": "2.1.0",
                "candidate_version": "2.2.0",
            },
        },
    )

    module._rewrite_candidate_requirements(tmp_path)

    projected = manifest.read_text(encoding="utf-8")
    assert 'symbolica = { version = "=2.2.0"' in projected
    assert 'symjit = { version = "=2.22.0"' in projected
