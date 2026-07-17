# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build_backend"))

import package_version  # noqa: E402


def _copy_version_contract(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    (root / "dependencies").mkdir(parents=True)
    for relative in (
        Path("Cargo.toml"),
        Path("pyproject.toml"),
        Path("dependencies/release-lock.toml"),
        Path("dependencies/contributor-lock.toml"),
    ):
        shutil.copy2(ROOT / relative, root / relative)
    return root


def test_cargo_workspace_is_the_canonical_package_version() -> None:
    assert package_version.canonical_package_version(ROOT) == "0.1.0"


def test_package_version_contract_rejects_release_lock_drift(tmp_path: Path) -> None:
    root = _copy_version_contract(tmp_path)
    lock = root / "dependencies" / "release-lock.toml"
    text = lock.read_text(encoding="utf-8")
    lock.write_text(
        text.replace('version = "0.1.0"', 'version = "0.1.1"', 1),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="package version mismatch"):
        package_version.canonical_package_version(root)


def test_package_version_contract_rejects_static_python_version(
    tmp_path: Path,
) -> None:
    root = _copy_version_contract(tmp_path)
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            'dynamic = ["version"]',
            'version = "0.1.0"',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=r"project\.version as dynamic"):
        package_version.canonical_package_version(root)


def test_contributor_and_release_locks_share_compatibility_abi(
    tmp_path: Path,
) -> None:
    root = _copy_version_contract(tmp_path)
    package_version.check_contributor_lock_consistency(root)
    lock = root / "dependencies" / "contributor-lock.toml"
    lock.write_text(
        lock.read_text(encoding="utf-8").replace(
            'symbolica_serialization = "symbolica-bincode2-v1"',
            'symbolica_serialization = "incompatible-test-abi"',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="dependency locks disagree"):
        package_version.check_contributor_lock_consistency(root)


def test_nix_shell_provides_python_build_frontend() -> None:
    flake = (ROOT / "flake.nix").read_text(encoding="utf-8")
    match = re.search(
        r"python = pkgs\.python311\.withPackages \(.*?\[(.*?)\]",
        flake,
        flags=re.DOTALL,
    )
    assert match is not None
    assert re.search(r"(?m)^\s+build\s*$", match.group(1)) is not None


@pytest.mark.parametrize(
    "recipe",
    (
        "dev-install",
        "dev-build",
        "dev-test",
        "legacy-physics",
        "legacy-physics-verify",
        "independent-physics-oracle",
        "test-deployment-candidate",
        "release-artifacts",
        "publish-dry-run",
    ),
)
def test_source_checkout_commands_fail_immediately_from_sdist(
    tmp_path: Path,
    recipe: str,
) -> None:
    just = shutil.which("just")
    if just is None:
        pytest.skip("just is not installed")
    source = tmp_path / "pyamplicol-0.1.0"
    source.mkdir()
    shutil.copy2(ROOT / "justfile", source / "justfile")
    marker = tmp_path / "python-was-invoked"
    fake_python = tmp_path / "python-must-not-run"
    fake_python.write_text(
        '#!/bin/sh\nprintf invoked > "$PYAMPLICOL_TEST_MARKER"\nexit 99\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment["PYTHON"] = str(fake_python)
    environment["PYAMPLICOL_TEST_MARKER"] = str(marker)

    completed = subprocess.run(
        [just, recipe],
        cwd=source,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "requires a full pyAmpliCol Git source checkout" in (
        completed.stdout + completed.stderr
    )
    assert not marker.exists()
