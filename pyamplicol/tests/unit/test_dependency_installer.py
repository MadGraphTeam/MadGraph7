# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import importlib.util
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
        "symbolica",
        "symbolica-community",
        "gammaloop",
    }
    assert {item.key for item in with_legacy} == {
        *(item.key for item in without_legacy),
        "legacy-amplicol",
    }
    assert all(len(item.revision) == 40 for item in with_legacy)
    legacy = next(item for item in with_legacy if item.key == "legacy-amplicol")
    assert legacy.branch == payload["legacy_amplicol"]["branch"]
    assert legacy.revision == payload["legacy_amplicol"]["revision"]


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
    assert all(patch["target"] != "ufo_model_loader" for patch in payload["patches"])


def test_legacy_oracle_uses_the_pinned_remote_branch_without_local_patches() -> None:
    module = _module()
    payload = module._lock()
    assert all(patch["target"] != "legacy_amplicol" for patch in payload["patches"])
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


def test_symjit_fork_revision_archive_tree_and_patch_are_pinned() -> None:
    module = _module()
    payload = module._lock()
    patches = module._contributor_patches(payload)

    assert len(patches) == 1
    patch = patches[0]
    assert patch.name == "symjit-allow-direct-complex-stack-spills"
    assert patch.target == "symjit"
    assert patch.relative_path == (
        "patches/symjit/0001-allow-direct-complex-stack-spills.patch"
    )
    assert patch.applies_to_revision == payload["symjit"]["candidate_revision"]
    assert len(module._patch_closure_sha256(patches)) == 64
    assert len(payload["symjit"]["candidate_revision"]) == 40
    assert len(payload["symjit"]["archive_sha256"]) == 64
    assert len(payload["symjit"]["source_tree_sha256"]) == 64
    assert len(payload["symjit"]["candidate_tree_sha256"]) == 64
    assert payload["symjit"]["source_tree_sha256"] != (
        payload["symjit"]["candidate_tree_sha256"]
    )
    assert payload["symjit"]["release_status"] == "fork-pr-candidate"


def test_tracked_symjit_direct_patch_lineage_replays_cleanly(tmp_path: Path) -> None:
    module = _module()
    target = tmp_path / "symjit"
    target.mkdir()
    environment = dict(os.environ)
    for name in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE"):
        environment.pop(name, None)
    environment["GIT_CEILING_DIRECTORIES"] = str(target.parent.resolve())

    applied: list[str] = []
    patch_paths = sorted(
        (module.DEPENDENCIES / "patches" / "symjit" / "upstream").glob(
            "000[2-4]-*.patch"
        )
    )
    for patch_path in patch_paths:
        if (
            b"diff --git a/rust/direct.rs b/rust/direct.rs\n"
            not in patch_path.read_bytes()
        ):
            continue
        command = [
            "git",
            "apply",
            "--whitespace=nowarn",
            "--include=rust/direct.rs",
            str(patch_path),
        ]
        check = subprocess.run(
            [*command[:2], "--check", *command[2:]],
            cwd=target,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert check.returncode == 0, (
            f"{patch_path.name} does not follow the tracked rust/direct.rs lineage:\n"
            f"{check.stderr}"
        )
        subprocess.run(
            command,
            cwd=target,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        applied.append(patch_path.name)

    assert applied == [
        "0003-Add-generic-direct-plane-applications.patch",
    ]
    direct = target / "rust" / "direct.rs"
    digest = subprocess.run(
        ["git", "hash-object", direct],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert digest == "e0575ed72913a692ba2c946fa6a5ec740da458a2"


def test_contributor_patch_application_is_exact_idempotent_and_fails_on_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    dependencies = tmp_path / "dependencies"
    checkouts = dependencies / "checkouts"
    target = checkouts / "symjit"
    target.mkdir(parents=True)
    source = target / "value.txt"
    source.write_text("before\n", encoding="utf-8")
    patch_path = dependencies / "patches" / "symjit" / "change.patch"
    patch_path.parent.mkdir(parents=True)
    patch_path.write_text(
        "diff --git a/value.txt b/value.txt\n"
        "--- a/value.txt\n"
        "+++ b/value.txt\n"
        "@@ -1 +1 @@\n"
        "-before\n"
        "+after\n",
        encoding="utf-8",
    )
    revision = "a" * 40
    source_tree_sha256 = module._source_tree_sha256(target)
    payload = {
        "symjit": {
            "candidate_revision": revision,
            "source_tree_sha256": source_tree_sha256,
            "candidate_tree_sha256": "f" * 64,
        },
        "patches": [
            {
                "name": "test-patch",
                "target": "symjit",
                "path": "patches/symjit/change.patch",
                "sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
                "applies_to_revision": revision,
            }
        ],
    }
    monkeypatch.setattr(module, "DEPENDENCIES", dependencies)
    monkeypatch.setattr(module, "CHECKOUTS", checkouts)
    runner = module.Runner(dry_run=False)

    module._apply_contributor_patches(runner, payload)
    assert source.read_text(encoding="utf-8") == "after\n"
    module._apply_contributor_patches(runner, payload)
    assert source.read_text(encoding="utf-8") == "after\n"

    source.write_text("ambient drift\n", encoding="utf-8")
    with pytest.raises(module.SetupError, match="neither cleanly applicable"):
        module._apply_contributor_patches(runner, payload)

    patch_path.write_bytes(patch_path.read_bytes() + b"# tampered\n")
    with pytest.raises(module.SetupError, match="digest mismatch"):
        module._contributor_patches(payload)


def test_overlapping_contributor_patch_series_is_idempotent_by_tree_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    dependencies = tmp_path / "dependencies"
    checkouts = dependencies / "checkouts"
    target = checkouts / "symjit"
    target.mkdir(parents=True)
    source = target / "value.txt"
    source.write_text("first\nsecond\n", encoding="utf-8")
    patch_directory = dependencies / "patches" / "symjit"
    patch_directory.mkdir(parents=True)
    first = patch_directory / "first.patch"
    first.write_text(
        "--- a/value.txt\n+++ b/value.txt\n@@ -1,2 +1,2 @@\n-first\n+FIRST\n second\n",
        encoding="utf-8",
    )
    second = patch_directory / "second.patch"
    second.write_text(
        "--- a/value.txt\n+++ b/value.txt\n@@ -1,2 +1,2 @@\n FIRST\n-second\n+SECOND\n",
        encoding="utf-8",
    )
    revision = "a" * 40

    def patch_entry(name: str, path: Path) -> dict[str, str]:
        return {
            "name": name,
            "target": "symjit",
            "path": f"patches/symjit/{path.name}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "applies_to_revision": revision,
        }

    payload = {
        "symjit": {
            "candidate_revision": revision,
            "source_tree_sha256": "0" * 64,
            "candidate_tree_sha256": "1" * 64,
        },
        "patches": [
            patch_entry("first", first),
            patch_entry("second", second),
        ],
    }
    monkeypatch.setattr(module, "DEPENDENCIES", dependencies)
    monkeypatch.setattr(module, "CHECKOUTS", checkouts)
    runner = module.Runner(dry_run=False)

    module._apply_contributor_patches(runner, payload)
    assert source.read_text(encoding="utf-8") == "FIRST\nSECOND\n"
    payload["symjit"]["candidate_tree_sha256"] = module._source_tree_sha256(target)
    module._apply_contributor_patches(runner, payload)
    assert source.read_text(encoding="utf-8") == "FIRST\nSECOND\n"


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

    module._build_candidate_dependency_wheels(
        FakeRunner(),
        {"symbolica": {"candidate_version": "2.2.0"}},
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
        "_materialize_symjit",
        "_apply_contributor_patches",
        "_configure_sources",
        "_verify_symjit_tree",
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


def test_archive_checkout_patch_does_not_discover_parent_repository(
    tmp_path: Path,
) -> None:
    module = _module()
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=repository,
        check=True,
    )
    parent_payload = repository / "payload.txt"
    parent_payload.write_text("old\n", encoding="utf-8")
    checkout = repository / "checkouts" / "dependency"
    checkout.mkdir(parents=True)
    checkout_payload = checkout / "payload.txt"
    checkout_payload.write_text("old\n", encoding="utf-8")
    patch = repository / "change.patch"
    patch.write_text(
        "--- a/payload.txt\n+++ b/payload.txt\n@@ -1 +1 @@\n-old\n+new\n",
        encoding="utf-8",
    )

    module._apply_patch(module.Runner(dry_run=False), checkout, patch)
    module._apply_patch(module.Runner(dry_run=False), checkout, patch)

    assert checkout_payload.read_text(encoding="utf-8") == "new\n"
    assert parent_payload.read_text(encoding="utf-8") == "old\n"


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


def test_installer_tree_fingerprint_matches_content_and_ignores_build_cache(
    tmp_path: Path,
) -> None:
    module = _module()
    source = tmp_path / "source"
    source.mkdir()
    payload = source / "input.txt"
    payload.write_text("first\n", encoding="utf-8")
    first = module._source_tree_sha256(source)
    payload.write_text("second\n", encoding="utf-8")
    second = module._source_tree_sha256(source)
    assert first != second
    target = source / "target"
    target.mkdir()
    (target / "output").write_text("build\n", encoding="utf-8")
    assert module._source_tree_sha256(source) == second


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
        'symjit = { version = "=2.21.1", default-features = false }\n',
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
            "symjit": {"candidate_version": "2.21.2"},
        },
    )
    monkeypatch.setattr(
        module,
        "_release_lock",
        lambda: {"symjit": {"version": "2.21.1"}},
    )

    module._rewrite_candidate_requirements(tmp_path)

    projected = manifest.read_text(encoding="utf-8")
    assert 'symbolica = { version = "=2.2.0"' in projected
    assert 'symjit = { version = "=2.21.2"' in projected
