# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import textwrap
import tomllib
import urllib.request
from pathlib import Path

import pytest

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
    assert workflows.count(f"rust-toolchain@{RUST_TOOLCHAIN_ACTION_SHA}") == 6
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
    assert combined.count("sha256sum --check --strict") >= 3


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


def test_release_workflow_uses_one_retained_sdist_and_all_targets() -> None:
    workflow = (WORKFLOWS / "release-artifacts.yml").read_text(encoding="utf-8")
    assert "git cat-file -t" in workflow
    assert "'.verification.verified'" in workflow
    assert "--sdist-only" in workflow
    assert workflow.count("--retained-sdist") >= 2
    assert "macos-15\n" in workflow
    assert "macos-15-intel" in workflow
    assert "manylinux_2_28_x86_64" in workflow
    assert 'python-version: "3.14"' in workflow
    assert "cp314-cp314" in workflow
    assert "--require-all-targets" in workflow
    assert (
        "SOURCE_COMMIT: ${{ needs.retained-sdist.outputs.source_commit }}" in workflow
    )
    assert "SOURCE_TAG: ${{ inputs.signed_tag }}" in workflow
    assert '--source-commit "$SOURCE_COMMIT"' in workflow
    assert '--source-tag "$SOURCE_TAG"' in workflow
    assert "retention-days: 90" in workflow
    assert "id-token: write" not in workflow
    assert "Full source validation gate" in workflow
    assert "needs: [verify-release-tag, full-source-validation]" in workflow
    assert "retained-sdist:\n    needs: legal-release-gate" in workflow
    assert "source_commit: ${{ steps.tag.outputs.source_commit }}" in workflow
    assert "workflow_commit: ${{ steps.tag.outputs.workflow_commit }}" in workflow
    assert "ref: ${{ needs.verify-release-tag.outputs.source_commit }}" in workflow
    assert 'PYAMPLICOL_REQUIRE_NATIVE_TESTS: "1"' in workflow
    assert "python tools/release/check_dependencies.py" in workflow
    assert "just source-gate" in workflow
    assert "g++ gfortran make" in workflow
    assert "brew install gcc" in workflow
    assert "gcc-c++ gcc-gfortran make" in workflow
    assert workflow.count("tools/release/test_deployment.py") == 4
    assert "continue-on-error" not in workflow


def test_legal_release_job_primes_an_isolated_cargo_home_then_runs_offline() -> None:
    workflow = (WORKFLOWS / "release-artifacts.yml").read_text(encoding="utf-8")
    legal_job = workflow.split("  legal-release-gate:\n", maxsplit=1)[1].split(
        "\n  retained-sdist:", maxsplit=1
    )[0]

    assert "CARGO_HOME: ${{ runner.temp }}/pyamplicol-legal-cargo" in legal_job
    assert f"dtolnay/rust-toolchain@{RUST_TOOLCHAIN_ACTION_SHA}" in legal_job
    assert "--fetch-locked-release-targets" in legal_job
    assert 'CARGO_NET_OFFLINE: "true"' in legal_job
    assert legal_job.index("--fetch-locked-release-targets") < legal_job.index(
        "Enforce native dependency legal gate"
    )
    assert "python tools/release/check_legal_inventory.py --mode release" in legal_job


def test_complete_source_gate_covers_every_required_suite_serially() -> None:
    justfile = JUSTFILE.read_text(encoding="utf-8")

    assert "source-gate:" in justfile
    required_targets = (
        "just legal-gate",
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


def test_publisher_is_manual_hash_checked_and_has_no_build_checkout() -> None:
    workflow = (WORKFLOWS / "publish-pypi.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "environment:" in workflow
    assert workflow.count("id-token: write") == 1
    assert workflow.count("contents: read") == 1
    assert "run-id: ${{ inputs.artifact_run_id }}" in workflow
    assert 'workflow["path"] == ".github/workflows/release-artifacts.yml"' in workflow
    assert 'run["conclusion"] == "success"' in workflow
    assert "VALIDATED_WORKFLOW_COMMIT" in workflow
    assert "VALIDATED_SOURCE_COMMIT" not in workflow
    assert 'manifest["source"]' in workflow
    assert 'signed_tag["verification"]["verified"] is True' in workflow
    assert 'target["sha"] == source_commit' in workflow
    for required_job in (
        "Verify signed release tag",
        "Native dependency legal gate",
        "Full source validation gate",
        "Build retained source distribution",
        "macOS release wheel and native deployment (macos-arm64)",
        "macOS release wheel and native deployment (macos-x86_64)",
        "manylinux release wheel and native deployment",
        "Assemble validated release bundle",
    ):
        assert required_job in workflow
    assert "Run complete release source gate" in workflow
    assert "sha256sum --check --strict SHA256SUMS" in workflow
    assert "actions/checkout" not in workflow
    assert "maturin" not in workflow
    assert "cargo" not in workflow
    assert "tools/release/build" not in workflow
    assert "gh-action-pypi-publish" in workflow


def test_publisher_does_not_conflate_dispatch_head_with_signed_tag_source() -> None:
    workflow = (WORKFLOWS / "publish-pypi.yml").read_text(encoding="utf-8")
    dispatch_head = "a" * 40
    signed_tag_source = "b" * 40

    assert dispatch_head != signed_tag_source
    assert "VALIDATED_WORKFLOW_COMMIT={run['head_sha']}" in workflow
    assert 'source_commit = source["commit"]' in workflow
    assert 'workflow_commit = os.environ["VALIDATED_WORKFLOW_COMMIT"]' in workflow
    assert 'target["sha"] == source_commit' in workflow
    assert "source_commit == workflow_commit" not in workflow
    assert '"tag": "v0.1.0"' not in workflow


def test_publisher_accepts_a_signed_tag_source_different_from_dispatch_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = (WORKFLOWS / "publish-pypi.yml").read_text(encoding="utf-8")
    verification = workflow.split("python - <<'PY'\n", maxsplit=2)[2]
    verification = textwrap.dedent(verification.split("\n          PY", maxsplit=1)[0])

    workflow_commit = "a" * 40
    source_commit = "b" * 40
    tag_object = "c" * 40
    artifact_specs = (
        (
            "pyamplicol-0.1.0-cp311-abi3-macosx_11_0_arm64.whl",
            "wheel",
            "macosx_11_0_arm64",
        ),
        (
            "pyamplicol-0.1.0-cp311-abi3-macosx_11_0_x86_64.whl",
            "wheel",
            "macosx_11_0_x86_64",
        ),
        (
            "pyamplicol-0.1.0-cp311-abi3-manylinux_2_28_x86_64.whl",
            "wheel",
            "manylinux_2_28_x86_64",
        ),
        ("pyamplicol-0.1.0.tar.gz", "sdist", None),
    )
    artifacts = []
    for index, (filename, kind, target) in enumerate(artifact_specs):
        payload = f"artifact-{index}".encode()
        (tmp_path / filename).write_bytes(payload)
        entry = {
            "filename": filename,
            "kind": kind,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }
        if target is not None:
            entry.update({"native_scan": True, "target": target})
        artifacts.append(entry)

    sdist = artifacts[-1]
    manifest = {
        "schema_version": 1,
        "distribution": "pyamplicol",
        "mode": "release",
        "publishable": True,
        "sdist_wheel_parity": "verified",
        "source": {"commit": source_commit, "tag": "v0.1.0"},
        "release_targets": [
            "macosx_11_0_arm64",
            "macosx_11_0_x86_64",
            "manylinux_2_28_x86_64",
        ],
        "artifacts": artifacts,
        "retained_sdist": {key: sdist[key] for key in ("filename", "sha256", "size")},
    }
    (tmp_path / "release-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    responses = {
        "/git/ref/tags/v0.1.0": {"object": {"type": "tag", "sha": tag_object}},
        f"/git/tags/{tag_object}": {
            "verification": {"verified": True},
            "object": {"type": "commit", "sha": source_commit},
        },
    }

    def fake_urlopen(request: urllib.request.Request, *, timeout: int) -> io.BytesIO:
        assert timeout == 30
        match = next(
            payload
            for suffix, payload in responses.items()
            if request.full_url.endswith(suffix)
        )
        return io.BytesIO(json.dumps(match).encode())

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GH_TOKEN", "fixture-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/pyamplicol")
    monkeypatch.setenv("VALIDATED_WORKFLOW_COMMIT", workflow_commit)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    exec(compile(verification, "publish-pypi.yml", "exec"), {})

    assert source_commit != os.environ["VALIDATED_WORKFLOW_COMMIT"]
