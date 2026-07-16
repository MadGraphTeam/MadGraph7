# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
JUSTFILE = ROOT / "justfile"
RUST_TOOLCHAIN = "1.89.0"
RUST_TOOLCHAIN_ACTION_SHA = "c1709d61444fb708e6ed87924f95626398d8d115"
RUSTUP_INIT_URL = (
    "https://static.rust-lang.org/rustup/archive/1.28.2/"
    "x86_64-unknown-linux-gnu/rustup-init"
)
RUSTUP_INIT_SHA256 = "20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c"
MANYLINUX_IMAGE = (
    "quay.io/pypa/manylinux_2_28_x86_64@"
    "sha256:b04887b645dde99b9e955aeae3ff4da414992d0bd88259f046295b56361c5614"
)
MEMORY_WATCHDOG = "tools/ci/memory_watchdog.py --limit-gib 30 --"


def _guarded_count(workflow: str, command: str) -> int:
    pattern = re.compile(
        re.escape(MEMORY_WATCHDOG) + r"(?:\s*\\)?\s+" + command,
        re.MULTILINE,
    )
    return len(pattern.findall(workflow))


def test_native_toolchains_and_manylinux_image_are_immutable() -> None:
    toolchain = tomllib.loads(
        (ROOT / "rust-toolchain.toml").read_text(encoding="utf-8")
    )["toolchain"]
    release_lock = tomllib.loads(
        (ROOT / "dependencies" / "release-lock.toml").read_text(encoding="utf-8")
    )["toolchain"]
    assert toolchain["channel"] == RUST_TOOLCHAIN
    assert release_lock["rust_toolchain"] == RUST_TOOLCHAIN
    assert release_lock["rust_toolchain_action_sha"] == RUST_TOOLCHAIN_ACTION_SHA
    assert release_lock["just"] == "1.46.0"
    assert (
        f"{release_lock['manylinux_image']}@"
        f"{release_lock['manylinux_image_digest']}" == MANYLINUX_IMAGE
    )

    workflows = "\n".join(
        (WORKFLOWS / name).read_text(encoding="utf-8")
        for name in ("candidate.yml", "release-artifacts.yml")
    )
    assert "rust-toolchain@stable" not in workflows
    assert "default-toolchain stable" not in workflows
    assert "manylinux_2_28_x86_64:latest" not in workflows
    assert workflows.count(f"rust-toolchain@{RUST_TOOLCHAIN_ACTION_SHA}") == 5
    assert workflows.count(f"default-toolchain {RUST_TOOLCHAIN}") == 2
    assert workflows.count(MANYLINUX_IMAGE) == 2
    assert "cargo install just --version 1.46.0 --locked" in workflows
    assert "pip install --upgrade pip" not in workflows


def test_external_actions_and_rustup_installer_are_immutable() -> None:
    workflows = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(WORKFLOWS.glob("*.yml"))
    }
    uses = []
    for name, workflow in workflows.items():
        matches = re.finditer(r"^\s*(?:-\s*)?uses:\s*([^#\s]+)", workflow, re.MULTILINE)
        for match in matches:
            action = match.group(1)
            assert "@" in action, f"{name}: action has no immutable revision: {action}"
            revision = action.rsplit("@", 1)[1]
            assert re.fullmatch(r"[0-9a-f]{40}", revision), (
                f"{name}: action is not pinned to a full commit: {action}"
            )
            uses.append(action)

    assert uses
    combined = "\n".join(workflows.values())
    assert "sh.rustup.rs" not in combined
    assert combined.count(RUSTUP_INIT_URL) == 2
    assert combined.count(RUSTUP_INIT_SHA256) == 2
    assert combined.count("sha256sum --check --strict") >= 2


def test_candidate_ci_is_read_only_and_covers_release_hosts() -> None:
    workflow = (WORKFLOWS / "candidate.yml").read_text(encoding="utf-8")
    assert "macos-15\n" in workflow
    assert "macos-15-intel" in workflow
    assert "manylinux_2_28_x86_64" in workflow
    assert "retention-days:" in workflow
    assert "id-token: write" not in workflow
    assert "contents: read" in workflow
    assert workflow.count("dependencies/install_dependencies.py") == 3
    assert workflow.count("--without-legacy-amplicol") == 3
    assert workflow.count("--no-build") == 3
    assert "Focused clean-checkout release tests" in workflow
    assert "PYAMPLICOL_BUILD_MODE: release" in workflow
    assert "Complete candidate source validation gate" in workflow
    assert workflow.count("needs: candidate-source-validation") == 2
    assert 'PYAMPLICOL_REQUIRE_NATIVE_TESTS: "1"' in workflow
    assert "just source-gate" in workflow
    assert "tools/release/test_deployment.py" in workflow
    assert "g++ gfortran make" in workflow
    assert "continue-on-error" not in workflow


def test_candidate_and_release_heavy_commands_use_memory_watchdog() -> None:
    candidate = (WORKFLOWS / "candidate.yml").read_text(encoding="utf-8")
    release = (WORKFLOWS / "release-artifacts.yml").read_text(encoding="utf-8")

    assert candidate.count(MEMORY_WATCHDOG) == 7
    assert (
        _guarded_count(
            candidate,
            r'(?:python|"\$PYTHON") dependencies/install_dependencies\.py',
        )
        == 3
    )
    assert (
        _guarded_count(
            candidate,
            r'env PYTHON="\$PWD/\.venv/bin/python" just source-gate',
        )
        == 1
    )
    assert (
        _guarded_count(
            candidate,
            r"\.venv/bin/python tools/release/test_deployment\.py",
        )
        == 1
    )
    assert (
        _guarded_count(
            candidate,
            r'(?:python|"\$PYTHON") tools/release/build_release_artifacts\.py',
        )
        == 2
    )

    assert "ulimit -v" not in release
    assert release.count(MEMORY_WATCHDOG) == 11
    assert (
        _guarded_count(
            release,
            r"cargo install just --version 1\.46\.0 --locked",
        )
        == 1
    )
    assert _guarded_count(release, r'python -m pip install "\.\[test\]"') == 1
    assert (
        _guarded_count(
            release,
            r'env PYTHON="\$\(command -v python\)" just source-gate',
        )
        == 1
    )
    assert (
        _guarded_count(
            release,
            r"python tools/developer/legacy_amplicol\.py",
        )
        == 1
    )
    assert (
        _guarded_count(
            release,
            r'(?:python|"\$PY311") tools/release/build_release_artifacts\.py',
        )
        == 3
    )
    assert (
        _guarded_count(
            release,
            r'(?:python|"\$PY311") tools/release/test_deployment\.py',
        )
        == 4
    )


def test_release_workflow_uses_one_retained_sdist_and_all_targets() -> None:
    workflow = (WORKFLOWS / "release-artifacts.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "signed_tag" not in workflow
    assert "Verify signed" not in workflow
    assert "verification.verified" not in workflow
    assert "ref: ${{ github.sha }}" in workflow
    assert "--sdist-only" in workflow
    assert workflow.count("--retained-sdist") >= 2
    assert "macos-15\n" in workflow
    assert "macos-15-intel" in workflow
    assert "manylinux_2_28_x86_64" in workflow
    assert 'python-version: "3.14"' in workflow
    assert "cp314-cp314" in workflow
    assert "--require-all-targets" in workflow
    assert "--output-dir .artifacts/validated" in workflow
    assert "--skip-clean-install" in workflow
    assert "--source-commit" not in workflow
    assert "--source-tag" not in workflow
    assert "release-manifest.json" not in workflow
    assert "SHA256SUMS" not in workflow
    assert "retention-days: 90" in workflow
    assert "id-token: write" not in workflow
    assert "Full source validation gate" in workflow
    assert "needs: [full-source-validation, independent-physics-oracle]" in workflow
    assert "Independent Fortran physics oracle" in workflow
    assert "Rebuild and verify pinned Fortran evidence" in workflow
    assert "ulimit -v" not in workflow
    assert "tests/fixtures/reference/physics-v2.json" in workflow
    assert "tests/fixtures/reference/legacy-fortran-v2.json" in workflow
    assert (
        "retained-sdist:\n    needs: [full-source-validation, "
        "independent-physics-oracle]" in workflow
    )
    assert 'PYAMPLICOL_REQUIRE_NATIVE_TESTS: "1"' in workflow
    assert "python tools/release/check_dependencies.py" in workflow
    assert "just source-gate" in workflow
    assert "g++ gfortran make" in workflow
    assert "brew install gcc" in workflow
    assert "gcc-c++ gcc-gfortran make" in workflow
    assert workflow.count("tools/release/test_deployment.py") == 4
    assert "Collect validated release artifacts" in workflow
    assert "python tools/release/publish_dry_run.py" in workflow
    assert "continue-on-error" not in workflow


def test_complete_source_gate_covers_every_required_suite_serially() -> None:
    justfile = JUSTFILE.read_text(encoding="utf-8")

    assert "source-gate:" in justfile
    required_targets = (
        "just dependency-gate",
        "just typing",
        "just python-unit",
        "just python-release",
        "just python-integration",
        "just python-physics",
        "just rust-check",
        "just rust-test",
        "just installed-smoke",
    )
    positions = [justfile.index(target) for target in required_targets]
    assert positions == sorted(positions)
    assert "PYAMPLICOL_REQUIRE_NATIVE_TESTS=1" in justfile
    assert "tests/integration/test_schema_v3_generation_runtime.py" in justfile
    assert "run_cargo.py --mode {{build_mode}} -- test --workspace" in justfile
    assert "{{python}} -m pyamplicol.selftest" in justfile
    assert "{{python}} -m pyamplicol examples list --format json" in justfile
    assert "examples run builtin_sm_lc" in justfile
    assert "generation.mode=replace" in justfile
    assert justfile.count("PYAMPLICOL_BUILD_MODE=release just source-gate") == 2
    assert "independent-physics-oracle:" in justfile
    assert justfile.count("just independent-physics-oracle") == 2
    assert "--prepare-checkout --fixture" in justfile
    assert "tests/fixtures/reference/physics-v2.json" in justfile
    assert "tests/fixtures/reference/legacy-fortran-v2.json" in justfile


def test_publisher_is_manual_oidc_only_and_has_no_build_checkout() -> None:
    workflow = (WORKFLOWS / "publish-pypi.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "environment:" in workflow
    assert workflow.count("id-token: write") == 1
    assert workflow.count("contents: read") == 1
    assert "run-id: ${{ inputs.artifact_run_id }}" in workflow
    assert 'workflow["path"] == ".github/workflows/release-artifacts.yml"' in workflow
    assert 'run["conclusion"] == "success"' in workflow
    assert 'run["event"] == "workflow_dispatch"' in workflow
    assert 'run["head_branch"] == run["repository"]["default_branch"]' in workflow
    for required_job in (
        "Full source validation gate",
        "Independent Fortran physics oracle",
        "Build retained source distribution",
        "macOS release wheel and native deployment (macos-arm64)",
        "macOS release wheel and native deployment (macos-x86_64)",
        "manylinux release wheel and native deployment",
        "Collect validated release artifacts",
    ):
        assert required_job in workflow
    assert "Run complete release source gate" in workflow
    assert "Rebuild and verify pinned Fortran evidence" in workflow
    assert "expected three wheels and one sdist" in workflow
    assert "candidate artifacts cannot be published" in workflow
    assert "release-manifest.json" not in workflow
    assert "SHA256SUMS" not in workflow
    assert "verification.verified" not in workflow
    assert "signed_tag" not in workflow
    assert "hashlib" not in workflow
    assert "actions/checkout" not in workflow
    assert "maturin" not in workflow
    assert "cargo" not in workflow
    assert "tools/release/build" not in workflow
    assert "gh-action-pypi-publish" in workflow


def test_publisher_requires_the_validated_default_branch_run() -> None:
    workflow = (WORKFLOWS / "publish-pypi.yml").read_text(encoding="utf-8")
    assert 'run["head_repository"]["full_name"] == repository' in workflow
    assert 'run["head_branch"] == run["repository"]["default_branch"]' in workflow
    assert 'workflow["path"] == ".github/workflows/release-artifacts.yml"' in workflow
    assert "git/ref/tags" not in workflow
    assert "git/tags" not in workflow


def test_release_pipeline_has_no_custom_supply_chain_bundle() -> None:
    retired = (
        ROOT / "build_backend" / "distribution_sbom.py",
        ROOT / "tools" / "release" / "check_legal_inventory.py",
        ROOT / "tools" / "release" / "check_rust_licenses.py",
    )
    assert not any(path.exists() for path in retired)

    release_tools = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "tools" / "release").glob("*.py"))
    )
    assert "CycloneDX" not in release_tools
    assert "release-manifest.json" not in release_tools
    assert "SHA256SUMS" not in release_tools
    assert "load_python_runtime_lock" not in release_tools
    assert "PythonRuntimeLock" not in release_tools
