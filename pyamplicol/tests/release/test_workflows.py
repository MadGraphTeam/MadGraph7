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
    assert workflows.count(f"rust-toolchain@{RUST_TOOLCHAIN_ACTION_SHA}") == 3
    assert workflows.count(f"default-toolchain {RUST_TOOLCHAIN}") == 2
    assert workflows.count(MANYLINUX_IMAGE) == 2
    assert "cargo install just" not in workflows
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
    trigger = workflow.split("on:\n", maxsplit=1)[1].split(
        "\npermissions:\n", maxsplit=1
    )[0]
    assert trigger.strip() == "workflow_dispatch:"
    assert "macos-15\n" in workflow
    assert "macos-15-intel" in workflow
    assert "manylinux_2_28_x86_64" in workflow
    assert "retention-days:" in workflow
    assert "id-token: write" not in workflow
    assert "contents: read" in workflow
    assert workflow.count("dependencies/install_dependencies.py") == 2
    assert workflow.count("--without-legacy-amplicol") == 2
    assert workflow.count("--dependencies-only") == 2
    assert workflow.count("--no-build") == 2
    assert "Focused clean-checkout release tests" in workflow
    assert "PYAMPLICOL_BUILD_MODE: release" in workflow
    assert workflow.count("needs: release-tool-tests") == 2
    assert workflow.count("PYAMPLICOL_REQUIRE_NATIVE_TESTS") == 2
    assert "just source-gate" not in workflow
    assert workflow.count("tools/release/test_deployment.py") == 2
    assert "gcc-c++ gcc-gfortran make" in workflow
    assert "brew install gcc" in workflow
    macos_job = workflow.split("  macos-candidate:\n", maxsplit=1)[1].split(
        "\n  manylinux-candidate:\n",
        maxsplit=1,
    )[0]
    assert macos_job.index("Test the installed candidate wheel and native SDK") < (
        macos_job.index("actions/upload-artifact")
    )
    manylinux_job = workflow.split("  manylinux-candidate:\n", maxsplit=1)[1]
    assert manylinux_job.index("tools/release/test_deployment.py") < (
        manylinux_job.index("actions/upload-artifact")
    )
    assert "continue-on-error" not in workflow


def test_candidate_artifact_build_omits_duplicate_performance_ceremony() -> None:
    workflow = (WORKFLOWS / "candidate.yml").read_text(encoding="utf-8")
    for removed_job in (
        "candidate-source-validation",
        "x86-performance-runtime-bundle",
        "x86-performance-matrix-shard",
        "x86-qq-recurrence-capture",
        "x86-performance-matrix-aggregate",
        "x86-qq-recurrence-acceptance",
        "x86-portable-candidate-acceptance",
    ):
        assert f"  {removed_job}:\n" not in workflow

    assert "443f354a467cdda187996bef1a41fbd5a00ae28d" not in workflow
    assert "/private/tmp/pyamplicol-eager-compiled-arena-base-src" not in workflow
    assert "x86_performance_runtime_bundle.py" not in workflow
    assert "recurrence_z6g_benchmark.py" not in workflow
    assert "compiled_mode_matrix_x86.py" not in workflow
    assert "arena_native_x86_acceptance.py" not in workflow
    assert "compiled_all_jit_arena_gate.py" not in workflow
    assert "four_quark_compiled_gate.py" not in workflow
    assert "eager_benchmark_matrix.py" not in workflow


def test_automatic_tests_cover_generation_config_provenance() -> None:
    workflow = (WORKFLOWS / "tests.yml").read_text(encoding="utf-8")
    trigger = workflow.split("on:\n", maxsplit=1)[1].split(
        "\npermissions:\n", maxsplit=1
    )[0]
    assert "pull_request:" in trigger
    assert "push:" in trigger
    assert "branches: [main]" in trigger
    assert "workflow_dispatch:" in trigger
    assert (
        "group: tests-${{ github.workflow }}-${{ github.ref }}-${{ github.event_name }}"
    ) in workflow
    candidate_job = workflow.split(
        "  candidate-runtime:\n",
        maxsplit=1,
    )[1]
    assert "    if: github.event_name != 'push'\n" in candidate_job
    assert "    needs: [python-compatibility, source-contracts]\n" in candidate_job
    source_contract_job = workflow.split(
        "  source-contracts:\n",
        maxsplit=1,
    )[1].split("\n  candidate-runtime:\n", maxsplit=1)[0]
    assert 'python-version: "3.11"' in source_contract_job
    assert 'python -m pip install "pytest>=8.3,<9"' in source_contract_job
    assert "tests/unit/test_api_requests.py" in source_contract_job
    assert (
        "tests/unit/test_api_requests.py::"
        "test_public_compiled_model_uses_the_canonical_schema"
    ) in source_contract_job
    assert "tests/unit/test_repository_policy.py" in source_contract_job
    assert "tests/release" in source_contract_job
    assert "dependencies/install_dependencies.py" not in source_contract_job
    focused_unit_step = workflow.split(
        "      - name: Run focused Python unit checks (30 GiB RSS limit)\n",
        maxsplit=1,
    )[1].split("\n      - name:", maxsplit=1)[0]
    assert MEMORY_WATCHDOG in focused_unit_step
    assert "tests/unit/test_generation_config_provenance.py" in focused_unit_step


def test_candidate_eager_smoke_uses_a_bundled_prepared_pack() -> None:
    workflow = (WORKFLOWS / "tests.yml").read_text(encoding="utf-8")
    match = re.search(
        r"PYAMPLICOL_EAGER_BUILTIN_PACK: "
        r"\$\{\{ github\.workspace \}\}/(\S+\.pyamplicol-model)",
        workflow,
    )

    assert match is not None
    relative = Path(match.group(1))
    assert relative.parent == Path("src/pyamplicol/assets/prepared_models")
    assert (ROOT / relative).is_file()


def test_candidate_and_release_heavy_commands_use_memory_watchdog() -> None:
    candidate = (WORKFLOWS / "candidate.yml").read_text(encoding="utf-8")
    release = (WORKFLOWS / "release-artifacts.yml").read_text(encoding="utf-8")

    assert candidate.count(MEMORY_WATCHDOG) == 6
    assert (
        _guarded_count(
            candidate,
            r'(?:python|"\$PYTHON") dependencies/install_dependencies\.py',
        )
        == 2
    )
    assert (
        _guarded_count(
            candidate,
            r'(?:python|"\$PYTHON") tools/release/test_deployment\.py',
        )
        == 2
    )
    assert (
        _guarded_count(
            candidate,
            r'(?:python|"\$PYTHON") '
            r"tools/release/build_release_artifacts\.py",
        )
        == 2
    )

    assert "ulimit -v" not in release
    assert release.count(MEMORY_WATCHDOG) == 8
    assert "cargo install just" not in release
    assert 'python -m pip install ".[test]"' not in release
    assert "just source-gate" not in release
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
    assert "Focused release source preflight" in workflow
    assert "needs: [release-source-preflight, independent-physics-oracle]" in workflow
    assert "Independent Fortran physics oracle" in workflow
    assert "Run independent Fortran physics comparison" in workflow
    assert 'python -m pip install "jsonschema>=4.22,<5"' in workflow
    assert "ulimit -v" not in workflow
    assert "tests/fixtures/reference/physics-v2.json" in workflow
    assert "tests/fixtures/reference/legacy-fortran-v2.json" not in workflow
    assert "--check-output" not in workflow
    assert (
        "retained-sdist:\n    needs: [release-source-preflight, "
        "independent-physics-oracle]" in workflow
    )
    assert "python tools/release/check_dependencies.py" in workflow
    assert "tests/unit/test_dependency_gate.py" in workflow
    assert "just source-gate" not in workflow
    assert "brew install gcc" in workflow
    assert "gcc-c++ gcc-gfortran make" in workflow
    assert workflow.count("tools/release/test_deployment.py") == 4
    assert "Collect validated release artifacts" in workflow
    assert "python tools/release/publish_dry_run.py" in workflow
    assert "continue-on-error" not in workflow


def test_release_prepared_model_workflow_is_manual_and_non_publishable() -> None:
    workflow = (WORKFLOWS / "release-prepared-models.yml").read_text(encoding="utf-8")
    trigger = workflow.split("on:\n", maxsplit=1)[1].split(
        "\npermissions:\n", maxsplit=1
    )[0]

    assert trigger.strip() == "workflow_dispatch:"
    assert "contents: read" in workflow
    assert "persist-credentials: false" in workflow
    assert "id-token: write" not in workflow
    assert "secrets." not in workflow
    assert "publish-pypi" not in workflow
    assert "gh-action-pypi-publish" not in workflow
    assert "git push" not in workflow
    assert "PYAMPLICOL_BUILD_MODE: release" in workflow
    assert "PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP" not in workflow
    assert "prepare_release_prepared_models.py" in workflow
    assert "bootstrap-wheel" in workflow
    assert "eager_portability.py produce" in workflow
    assert "--asset-mode release" in workflow
    assert "Verify source-store output layout" in workflow
    assert "release_assets/prepared_models" in workflow
    assert "runner: ubuntu-24.04" in workflow
    assert "runner: macos-15" in workflow
    assert "architecture: x86_64" in workflow
    assert "architecture: aarch64" in workflow
    assert workflow.count(MEMORY_WATCHDOG) == 2
    assert "if-no-files-found: error" in workflow
    assert "retention-days: 14" in workflow


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
    independent_recipe = justfile.split(
        "independent-physics-oracle:", maxsplit=1
    )[1].split("\n\n", maxsplit=1)[0]
    assert "tests/fixtures/reference/legacy-fortran-v2.json" not in independent_recipe
    assert "--check-output" not in independent_recipe


def test_publisher_is_manual_oidc_only_and_has_no_build_checkout() -> None:
    workflow = (WORKFLOWS / "publish-pypi.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "environment:" in workflow
    assert workflow.count("id-token: write") == 1
    assert workflow.count("contents: read") == 1
    assert "run-id: ${{ inputs.artifact_run_id }}" in workflow
    assert "Verify the validated release workflow run" not in workflow
    assert "api.github.com" not in workflow
    assert "workflow_url" not in workflow
    assert "required_jobs" not in workflow
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


def test_publisher_does_not_reauthenticate_release_run_metadata() -> None:
    workflow = (WORKFLOWS / "publish-pypi.yml").read_text(encoding="utf-8")
    assert 'run["head_repository"]' not in workflow
    assert 'run["head_branch"]' not in workflow
    assert 'workflow["path"]' not in workflow
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
