# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import json
import re
import shutil
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "release"))

from check_rust_licenses import (  # noqa: E402
    CargoArtifactSpec,
    CargoClosure,
    CargoClosurePackage,
    CargoPackageKey,
    _archive_legal_files,
    _rust_license_corpus_file_issues,
    _static_lgpl_graph,
    blocking_issues,
    check_repository,
    collect_cargo_closures,
    main,
    refresh_rust_license_corpus,
)


def _policy_checkout(tmp_path: Path) -> Path:
    for name in (
        "Cargo.lock",
        "Cargo.toml",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
    ):
        shutil.copy2(ROOT / name, tmp_path / name)
    shutil.copytree(ROOT / "licenses", tmp_path / "licenses")
    (tmp_path / "dependencies").mkdir()
    shutil.copy2(
        ROOT / "dependencies" / "release-lock.toml",
        tmp_path / "dependencies" / "release-lock.toml",
    )
    shutil.copytree(ROOT / "rust", tmp_path / "rust")
    return tmp_path


def _issue_codes(root: Path) -> set[str]:
    return {issue.code for issue in check_repository(root)}


def _corpus_files(root: Path) -> dict[str, bytes]:
    corpus_root = root / "licenses" / "rust"
    return {
        path.relative_to(corpus_root).as_posix(): path.read_bytes()
        for path in corpus_root.rglob("*")
        if path.is_file()
    }


def test_rust_inventory_covers_every_locked_third_party_package() -> None:
    issues = check_repository(ROOT)
    assert issues == []
    assert blocking_issues(issues, mode="candidate") == []
    assert blocking_issues(issues, mode="release") == []

    with (ROOT / "Cargo.lock").open("rb") as stream:
        lock = tomllib.load(stream)
    with (ROOT / "licenses" / "RUST_THIRD_PARTY.toml").open("rb") as stream:
        inventory = tomllib.load(stream)

    locked = [package for package in lock["package"] if "source" in package]
    assert len(inventory["package"]) == len(locked)
    assert {package["name"] for package in inventory["package"]} >= {
        "symbolica",
        "symjit",
    }
    assert all(package.get("checksum") for package in inventory["package"])

    graph = _static_lgpl_graph(
        collect_cargo_closures(
            ROOT,
            tuple(
                spec
                for spec in (
                    CargoArtifactSpec(
                        cargo_root="rusticol-capi",
                        artifact="librusticol_capi.a",
                        package="rusticol-capi",
                        manifest=Path("rust/crates/rusticol-capi/Cargo.toml"),
                    ),
                    CargoArtifactSpec(
                        cargo_root="rusticol-python",
                        artifact="pyamplicol._rusticol",
                        package="rusticol-python",
                        manifest=Path("rust/crates/rusticol-python/Cargo.toml"),
                        features=("extension-module",),
                    ),
                )
            ),
            ("aarch64-apple-darwin",),
        ),
        inventory,
    )
    assert graph == {
        "rusticol-capi": {},
        "rusticol-python": {},
    }


def test_retained_corpus_covers_release_closures_and_multiple_legal_files() -> None:
    with (ROOT / "licenses" / "RUST_THIRD_PARTY.toml").open("rb") as stream:
        inventory = tomllib.load(stream)

    corpus = inventory["rust_license_corpus"]
    packages = corpus["package"]
    keys = [
        (package["name"], package["version"], package["source"]) for package in packages
    ]
    assert keys == sorted(keys)
    assert len(packages) == 79
    assert sum(len(package["file"]) for package in packages) == 158
    assert set(_corpus_files(ROOT)) == {
        Path(file["retained_path"]).relative_to("licenses/rust").as_posix()
        for package in packages
        for file in package["file"]
    }

    by_name = {package["name"]: package for package in packages}
    assert {file["package_path"] for file in by_name["unicode-ident"]["file"]} == {
        "LICENSE-APACHE",
        "LICENSE-MIT",
        "LICENSE-UNICODE",
    }
    assert {file["package_path"] for file in by_name["pyo3"]["file"]} == {
        "LICENSE-APACHE",
        "LICENSE-MIT",
        "pyo3-runtime/LICENSE-APACHE",
        "pyo3-runtime/LICENSE-MIT",
    }
    assert {file["package_path"] for file in by_name["memchr"]["file"]} == {
        "COPYING",
        "LICENSE-MIT",
        "UNLICENSE",
    }

    fallbacks = {
        package["name"]: package for package in packages if package["fallback"]
    }
    assert set(fallbacks) == {
        "hugepage-rs",
        "spec_math",
        "wasmtime-jit-icache-coherence",
    }
    assert {
        file["source_kind"]
        for package in fallbacks.values()
        for file in package["file"]
    } == {"locked-registry-fallback"}


def test_archive_discovery_retains_all_conventional_and_manifest_legal_files(
    tmp_path: Path,
) -> None:
    key = CargoPackageKey(
        "fixture-crate",
        "1.2.3",
        "registry+https://github.com/rust-lang/crates.io-index",
    )
    archive = tmp_path / "fixture-crate-1.2.3.crate"
    files = {
        "Cargo.toml.orig": (
            b'[package]\nname = "fixture-crate"\nversion = "1.2.3"\n'
            b'license-file = "legal/custom.txt"\n'
        ),
        "LICENSE-APACHE": b"apache text\n",
        "LICENSE-MIT": b"mit text\n",
        "UNLICENSE": b"unlicense text\n",
        "third-party/NOTICE.txt": b"notice text\n",
        "legal/custom.txt": b"custom terms\n",
        "README.md": b"not a retained legal text\n",
    }
    with tarfile.open(archive, "w:gz") as package:
        for relative, data in files.items():
            member = tarfile.TarInfo(f"fixture-crate-1.2.3/{relative}")
            member.size = len(data)
            package.addfile(member, io.BytesIO(data))

    assert _archive_legal_files(archive, key) == {
        relative: files[relative]
        for relative in (
            "LICENSE-APACHE",
            "LICENSE-MIT",
            "UNLICENSE",
            "legal/custom.txt",
            "third-party/NOTICE.txt",
        )
    }


def test_corpus_refresh_is_byte_deterministic(tmp_path: Path) -> None:
    checkout = _policy_checkout(tmp_path)
    expected_inventory = (ROOT / "licenses" / "RUST_THIRD_PARTY.toml").read_bytes()
    expected_files = _corpus_files(ROOT)

    assert refresh_rust_license_corpus(checkout) == (79, 158)

    assert (checkout / "licenses" / "RUST_THIRD_PARTY.toml").read_bytes() == (
        expected_inventory
    )
    assert _corpus_files(checkout) == expected_files


def test_missing_cargo_archives_fail_closed_for_candidate_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with (ROOT / "licenses" / "RUST_THIRD_PARTY.toml").open("rb") as stream:
        inventory = tomllib.load(stream)
    monkeypatch.setenv("CARGO_HOME", str(tmp_path / "empty-cargo-home"))

    issues = _rust_license_corpus_file_issues(ROOT, inventory)

    source_issues = [
        issue for issue in issues if issue.code == "rust-license-corpus-source-archive"
    ]
    assert source_issues
    assert all(not issue.release_only for issue in source_issues)
    assert source_issues[0] in blocking_issues(issues, mode="candidate")


def test_checker_reports_candidate_and_release_readiness_separately(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--root", str(ROOT), "--mode", "candidate", "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["candidate_ready"] is True
    assert payload["release_ready"] is True
    assert payload["requested_mode_ready"] is True
    assert "ready" not in payload

    assert main(["--root", str(ROOT), "--mode", "release", "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["requested_mode_ready"] is True


def test_feature_graph_detects_malachite_family_in_resolved_closures() -> None:
    inventory = {
        "package": [
            {
                "name": "symbolica",
                "version": "2.1.0",
                "source": "registry+test",
                "license": "LicenseRef-Symbolica-Proprietary",
            },
            *(
                {
                    "name": name,
                    "version": "0.7.1",
                    "source": "registry+test",
                    "license": "LGPL-3.0-only",
                }
                for name in ("malachite-base", "malachite-nz", "malachite-q")
            ),
        ]
    }
    packages = tuple(
        CargoClosurePackage(
            name=name,
            version="2.1.0" if name == "symbolica" else "0.7.1",
            source="registry+test",
            package_id=f"registry+test#{name}",
        )
        for name in (
            "symbolica",
            "malachite-base",
            "malachite-nz",
            "malachite-q",
        )
    )
    closures = tuple(
        CargoClosure(
            spec=CargoArtifactSpec(
                cargo_root=root,
                artifact=artifact,
                package=root,
                manifest=Path(f"rust/crates/{root}/Cargo.toml"),
            ),
            target="aarch64-apple-darwin",
            command=("cargo", "metadata"),
            packages=packages,
            activated_features=(),
        )
        for root, artifact in (
            ("rusticol-capi", "librusticol_capi.a"),
            ("rusticol-python", "pyamplicol._rusticol"),
        )
    )
    assert _static_lgpl_graph(closures, inventory) == {
        "rusticol-capi": {
            "malachite": (
                "malachite-base@0.7.1",
                "malachite-nz@0.7.1",
                "malachite-q@0.7.1",
            )
        },
        "rusticol-python": {
            "malachite": (
                "malachite-base@0.7.1",
                "malachite-nz@0.7.1",
                "malachite-q@0.7.1",
            )
        },
    }


def test_clean_feature_closure_accepts_the_completed_separation_policy(
    tmp_path: Path,
) -> None:
    checkout = _policy_checkout(tmp_path)
    issues = check_repository(checkout)
    assert issues == []
    assert blocking_issues(issues, mode="candidate") == []


def test_checker_detects_lock_checksum_and_spdx_drift(tmp_path: Path) -> None:
    checkout = _policy_checkout(tmp_path)
    lock_path = checkout / "Cargo.lock"
    lock_text = lock_path.read_text(encoding="utf-8")
    changed_lock, replacements = re.subn(
        r'checksum = "[0-9a-f]{64}"',
        f'checksum = "{"0" * 64}"',
        lock_text,
        count=1,
    )
    assert replacements == 1
    lock_path.write_text(changed_lock, encoding="utf-8")
    assert "checksum-mismatch" in _issue_codes(checkout)

    shutil.copy2(ROOT / "Cargo.lock", lock_path)
    inventory_path = checkout / "licenses" / "RUST_THIRD_PARTY.toml"
    inventory_text = inventory_path.read_text(encoding="utf-8")
    changed_inventory = inventory_text.replace(
        'license = "MIT OR Apache-2.0"',
        'license = "MIT/Apache-2.0"',
        1,
    )
    assert changed_inventory != inventory_text
    inventory_path.write_text(changed_inventory, encoding="utf-8")
    assert "license-expression" in _issue_codes(checkout)


def test_checker_requires_legal_files_in_release_source(tmp_path: Path) -> None:
    checkout = _policy_checkout(tmp_path)
    pyproject_path = checkout / "pyproject.toml"
    pyproject = pyproject_path.read_text(encoding="utf-8")
    changed = pyproject.replace(
        '{ path = "licenses/**/*", format = ["sdist"] }',
        '{ path = "licenses/**/*", format = ["wheel"] }',
    )
    assert changed != pyproject
    pyproject_path.write_text(changed, encoding="utf-8")
    assert "sdist-license-file" in _issue_codes(checkout)

    (checkout / "licenses" / "SymJIT.txt").unlink()
    codes = _issue_codes(checkout)
    assert "legal-file-inventory" in codes
    assert "notice-missing" in codes


def test_retained_corpus_tampering_blocks_candidate_and_release(
    tmp_path: Path,
) -> None:
    checkout = _policy_checkout(tmp_path)
    retained = next((checkout / "licenses" / "rust").rglob("LICENSE-MIT"))
    retained.write_bytes(retained.read_bytes() + b"tampered\n")

    issues = check_repository(checkout)

    assert "rust-license-corpus-hash" in {issue.code for issue in issues}
    assert any(
        issue.code == "rust-license-corpus-hash"
        for issue in blocking_issues(issues, mode="candidate")
    )
    assert any(
        issue.code == "rust-license-corpus-hash"
        for issue in blocking_issues(issues, mode="release")
    )


def test_checker_rejects_missing_extra_and_stale_corpus_files(tmp_path: Path) -> None:
    checkout = _policy_checkout(tmp_path)
    retained = next((checkout / "licenses" / "rust").rglob("LICENSE-MIT"))
    original = retained.read_bytes()
    retained.unlink()
    missing_codes = _issue_codes(checkout)
    assert "rust-license-corpus-file" in missing_codes
    assert "rust-license-corpus-file-set" in missing_codes

    retained.write_bytes(original)
    extra = retained.parent / "UNRECORDED.txt"
    extra.write_text("unrecorded\n", encoding="utf-8")
    extra_codes = _issue_codes(checkout)
    assert "rust-license-corpus-file-set" in extra_codes
    assert "legal-file-inventory" in extra_codes
    extra.unlink()

    inventory_path = checkout / "licenses" / "RUST_THIRD_PARTY.toml"
    inventory_text = inventory_path.read_text(encoding="utf-8")
    before, marker, corpus = inventory_text.partition(
        "# BEGIN GENERATED RUST LICENSE CORPUS"
    )
    assert marker
    stale_corpus, replacements = re.subn(
        r'sha256 = "[0-9a-f]{64}"',
        f'sha256 = "{"0" * 64}"',
        corpus,
        count=1,
    )
    assert replacements == 1
    inventory_path.write_text(before + marker + stale_corpus, encoding="utf-8")
    stale_codes = _issue_codes(checkout)
    assert "rust-license-corpus-source-drift" in stale_codes
    assert "rust-license-corpus-hash" in stale_codes
