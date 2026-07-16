# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib.util
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
        "symjit",
        "ufo-model-loader",
        "gammaloop",
    }
    assert {item.key for item in with_legacy} == {
        *(item.key for item in without_legacy),
        "legacy-amplicol",
    }
    assert all(len(item.revision) == 40 for item in with_legacy)


def test_ufo_loader_uses_public_base_and_checksummed_local_patch() -> None:
    module = _module()
    payload = module._lock()
    loader = payload["ufo_model_loader"]
    assert loader["candidate_revision"] == ("9cb4deeae40ddd64184049af07ac1d03ce5f6162")
    patches = [
        entry
        for entry in payload["patches"]
        if entry["dependency"] == "ufo-model-loader"
    ]
    assert patches == [
        {
            "dependency": "ufo-model-loader",
            "path": (
                "patches/ufo-model-loader/0001-fix-sparse-json-restrictions.patch"
            ),
            "sha256": (
                "34b70eb92402070e8aabaae8deb9a25ef09e6a6beb05537568d37aed6a972737"
            ),
        }
    ]


def test_contributor_runtime_requirements_use_the_full_hash_locked_closure() -> None:
    module = _module()
    requirements = module._runtime_requirements_text()
    assert "symbolica==" not in requirements
    assert "ufo-model-loader==" not in requirements
    for requirement in (
        "colorama==0.4.6",
        "numpy==2.4.2",
        "prettytable==3.18.0",
        "progressbar2==4.5.0",
        "python-utils==4.0.0",
        "typing-extensions==4.16.0",
        "wcwidth==0.8.2",
    ):
        assert requirements.count(requirement) == 1
    assert requirements.count("--hash=sha256:") > 20


def test_toml_section_replacement_is_idempotent() -> None:
    module = _module()
    original = '[package]\nname = "x"\n\n[dependencies]\na = "1"\n'
    once = module._replace_section(original, "dependencies", 'b = "2"')
    twice = module._replace_section(once, "dependencies", 'b = "2"')
    assert once == twice
    assert 'a = "1"' not in once
    assert once.count('b = "2"') == 1


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


def test_python_dependency_build_uses_an_isolated_clean_source_copy(
    tmp_path: Path,
) -> None:
    module = _module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    package = source / "src" / "package"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / ".git").mkdir()
    (source / ".git" / "index").write_bytes(b"git")
    egg_info = source / "src" / "package.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("generated\n", encoding="utf-8")
    (source / "build").mkdir()
    (source / "build" / "output").write_text("generated\n", encoding="utf-8")

    staged = tmp_path / "staged"
    module._stage_python_source(source, staged)

    assert (staged / "pyproject.toml").is_file()
    assert (staged / "src" / "package" / "__init__.py").is_file()
    assert not (staged / ".git").exists()
    assert not (staged / "src" / "package.egg-info").exists()
    assert not (staged / "build").exists()
    assert (source / "src" / "package.egg-info" / "PKG-INFO").is_file()


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
        "_rewrite_candidate_symjit_requirement",
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


def test_candidate_symjit_projection_rewrites_only_the_isolated_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    manifest = tmp_path / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        'symjit = { version = "=2.18.9", default-features = false }\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_lock",
        lambda: {
            "symbolica": {"published_symjit_version": "2.18.9"},
            "symjit": {"candidate_version": "2.19.3"},
        },
    )

    module._rewrite_candidate_symjit_requirement(tmp_path)

    assert 'version = "=2.19.3"' in manifest.read_text(encoding="utf-8")
