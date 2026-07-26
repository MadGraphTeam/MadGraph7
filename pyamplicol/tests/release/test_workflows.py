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
ARENA_X86_ACCEPTANCE = "tools/developer/arena_native_x86_acceptance.py"


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
    assert workflow.count("dependencies/install_dependencies.py") == 5
    assert workflow.count("--without-legacy-amplicol") == 5
    assert workflow.count("--no-build") == 3
    assert "Focused clean-checkout release tests" in workflow
    assert "PYAMPLICOL_BUILD_MODE: release" in workflow
    assert "Complete candidate source validation gate" in workflow
    assert workflow.count("needs: candidate-source-validation") == 2
    assert workflow.count("PYAMPLICOL_REQUIRE_NATIVE_TESTS") == 3
    assert "just source-gate" in workflow
    assert workflow.count("tools/release/test_deployment.py") == 3
    assert "g++ gfortran make" in workflow
    assert "gcc-c++ gcc-gfortran make" in workflow
    assert "brew install gcc" in workflow
    source_job = workflow.split(
        "  candidate-source-validation:\n",
        maxsplit=1,
    )[1].split("\n  macos-candidate:\n", maxsplit=1)[0]
    assert source_job.index("Audit emitted Arena acceptance evidence") < (
        source_job.index("actions/upload-artifact")
    )
    macos_job = workflow.split("  macos-candidate:\n", maxsplit=1)[1].split(
        "\n  manylinux-candidate:\n",
        maxsplit=1,
    )[0]
    assert macos_job.index("Test the installed candidate wheel and native SDK") < (
        macos_job.index("actions/upload-artifact")
    )
    assert "continue-on-error" not in workflow


def test_candidate_native_x86_acceptance_is_exact_and_content_bound() -> None:
    workflow = (WORKFLOWS / "candidate.yml").read_text(encoding="utf-8")
    source_job = workflow.split(
        "  candidate-source-validation:\n",
        maxsplit=1,
    )[1].split("\n  macos-candidate:\n", maxsplit=1)[0]

    assert "runs-on: ubuntu-24.04" in source_job
    assert "timeout-minutes: 360" in source_job
    assert "ARENA_EVIDENCE_ROOT: /tmp/pyamplicol-arena-x86-${{ github.sha }}" in (
        source_job
    )
    assert "ref: ${{ github.sha }}" in source_job
    assert "persist-credentials: false" in source_job
    assert source_job.count(ARENA_X86_ACCEPTANCE) == 2
    assert "<<'PY'" not in source_job
    assert "runtime-identity-preflight.json" in source_job
    assert "compiled-all-jit-arena-gate.json" in source_job
    assert "four-quark-compiled-gate.json" in source_job
    assert "eager-compiled-color/result.json" in source_job
    assert "arena-native-x86-acceptance.json" in source_job
    assert '--expected-revision "${{ github.sha }}"' in source_job
    assert source_job.count('--expected-workspace "${{ github.workspace }}"') == 2
    assert source_job.count("--points 3") == 2
    assert "--point-count 3" in source_job
    assert "--generation-timeout 900" in source_job
    assert "--generation-timeout 2400" in source_job
    assert source_job.index(f"{ARENA_X86_ACCEPTANCE} \\\n            preflight") < (
        source_job.index("compiled_all_jit_arena_gate.py")
    )
    assert source_job.index("eager_benchmark_matrix.py") < source_job.index(
        f"{ARENA_X86_ACCEPTANCE}\n          audit"
    )
    assert source_job.index("arena-native-x86-acceptance.json") < source_job.index(
        "if-no-files-found:"
    )

    helper = (ROOT / ARENA_X86_ACCEPTANCE).read_text(encoding="utf-8")
    assert '"-I",' in helper
    assert '"-S",' in helper
    assert '"-B",' in helper
    assert "preimport_python_runtime_identity" in helper
    assert "source_only_bytecode_policy" in helper
    assert "loaded_pyamplicol_origin_policy" in helper
    assert "candidate_wheel_matches_loaded_runtime" in helper
    assert "all_evidence_files_content_bound" in helper


def test_candidate_x86_performance_pipeline_is_exact_and_fail_closed() -> None:
    workflow = (WORKFLOWS / "candidate.yml").read_text(encoding="utf-8")
    assert "/private/tmp/pyamplicol-arena-x86" not in workflow
    assert "continue-on-error" not in workflow
    for job in (
        "x86-performance-runtime-bundle",
        "x86-performance-matrix-shard",
        "x86-qq-recurrence-capture",
        "x86-performance-matrix-aggregate",
        "x86-qq-recurrence-acceptance",
        "x86-portable-candidate-acceptance",
    ):
        assert workflow.count(f"  {job}:\n") == 1

    runtime_job = workflow.split(
        "  x86-performance-runtime-bundle:\n",
        maxsplit=1,
    )[1].split("\n  x86-performance-matrix-shard:\n", maxsplit=1)[0]
    assert "runs-on: ubuntu-24.04" in runtime_job
    assert "timeout-minutes: 360" in runtime_job
    assert "needs: release-tool-tests" in runtime_job
    baseline_root = "/private/tmp/pyamplicol-eager-compiled-arena-base-src"
    assert f"BASELINE_SOURCE_ROOT: {baseline_root}" in runtime_job
    assert f"working-directory: {baseline_root}" in runtime_job
    assert (
        "working-directory: /tmp/pyamplicol-eager-compiled-arena-base-src"
        not in runtime_job
    )
    private_tmp_setup = "sudo install -d -m 1777 /private/tmp"
    assert private_tmp_setup in runtime_job
    assert 'test "$(stat -c \'%a\' /private/tmp)" = 1777' in runtime_job
    assert runtime_job.index(private_tmp_setup) < runtime_job.index(
        "git worktree add --detach"
    )
    assert "443f354a467cdda187996bef1a41fbd5a00ae28d" in runtime_job
    assert "freeze-baseline" in runtime_job
    assert "frozen-baseline-attestation.json" in runtime_job
    assert runtime_job.count("build_release_artifacts.py") == 2
    assert "bundle-dependencies" in runtime_job
    assert "prepare-ufo" in runtime_job
    assert "materialize" in runtime_job
    assert "create-manifest" in runtime_job
    assert "verify" in runtime_job
    assert "if-no-files-found: error" in runtime_job
    assert runtime_job.count("include-hidden-files: true") == 1

    shard_job = workflow.split(
        "  x86-performance-matrix-shard:\n",
        maxsplit=1,
    )[1].split("\n  x86-qq-recurrence-capture:\n", maxsplit=1)[0]
    assert "needs: x86-performance-runtime-bundle" in shard_job
    assert "timeout-minutes: 360" in shard_job
    assert "shard: [0, 1, 2, 3, 4, 5, 6, 7]" in shard_job
    assert "compiled_mode_matrix_x86.py shard" in shard_job
    assert "--shard-count 8" in shard_job
    assert "--samples 7" in shard_job
    assert "--target-runtime 5" in shard_job
    assert "--minimum-samples 7" in shard_job
    assert "--warmup-runs 2" in shard_job
    assert "--rerun-results" in shard_job
    assert "--regenerate-artifacts" in shard_job
    assert "runtime-bundle.json" in shard_job

    capture_job = workflow.split(
        "  x86-qq-recurrence-capture:\n",
        maxsplit=1,
    )[1].split("\n  x86-performance-matrix-aggregate:\n", maxsplit=1)[0]
    assert "timeout-minutes: 360" in capture_job
    for role in (
        "builtin-topology",
        "builtin-union",
        "ufo-topology",
        "ufo-union",
    ):
        assert capture_job.count(f"role: {role}") == 1
    assert "recurrence_z6g_benchmark.py" in capture_job
    assert capture_job.count("--mode compiled") == 1
    assert capture_job.count("--mode eager") == 1
    assert capture_job.count("--mode recurrence") == 1
    assert "--target-runtime 5" in capture_job
    assert "--minimum-samples 7" in capture_job
    assert "--subprocess-samples 7" in capture_job
    assert "--validation-samples 10" in capture_job
    assert "--generation-timeout 10800" in capture_job
    assert "--profile-timeout 1800" in capture_job
    assert "--specialize-flow-at-generation" not in capture_job
    assert "flow:2,4,5,6,7,8,9,1" in capture_job
    assert "h:-1,+1,-1,+1,-1,+1,-1,+1,-1" in capture_job
    assert (
        "path: ${{ env.QQ_UPLOAD_ROOT }}/${{ matrix.role }}.json"
        in capture_job
    )
    assert "path: ${{ env.QQ_CAPTURE_ROOT }}/" not in capture_job
    assert "path: ${{ env.QQ_WORK_ROOT }}" not in capture_job
    assert (
        '"$QQ_UPLOAD_ROOT/${{ matrix.role }}.json"'
        in capture_job
    )

    aggregate_job = workflow.split(
        "  x86-performance-matrix-aggregate:\n",
        maxsplit=1,
    )[1].split("\n  x86-qq-recurrence-acceptance:\n", maxsplit=1)[0]
    assert "compiled_mode_matrix_x86.py aggregate" in aggregate_job
    assert "--shard-count 8" in aggregate_job
    assert "merge-multiple: true" in aggregate_job
    assert "compiled-mode-matrix-x86-aggregate.json" in aggregate_job

    qq_job = workflow.split(
        "  x86-qq-recurrence-acceptance:\n",
        maxsplit=1,
    )[1].split("\n  x86-portable-candidate-acceptance:\n", maxsplit=1)[0]
    assert "x86_qq_recurrence_acceptance.py" in qq_job
    assert "merge-multiple: true" in qq_job
    assert "--builtin-topology" in qq_job
    assert "$QQ_CAPTURE_ROOT/builtin-topology.json" in qq_job
    assert "$QQ_CAPTURE_ROOT/builtin-topology/result.json" not in qq_job
    assert "--builtin-union" in qq_job
    assert "--ufo-topology" in qq_job
    assert "--ufo-union" in qq_job

    final_job = workflow.split(
        "  x86-portable-candidate-acceptance:\n",
        maxsplit=1,
    )[1].split("\n  macos-candidate:\n", maxsplit=1)[0]
    for need in (
        "candidate-source-validation",
        "x86-performance-matrix-aggregate",
        "x86-qq-recurrence-acceptance",
    ):
        assert f"- {need}" in final_job
    assert "x86_portable_performance_acceptance.py" in final_job
    assert "arena-native-x86-acceptance.json" in final_job
    assert "compiled-mode-matrix-x86-aggregate.json" in final_job
    assert "x86-qq-recurrence-acceptance.json" in final_job
    assert '--expected-revision "${{ github.sha }}"' in final_job
    assert "if-no-files-found: error" in final_job

    assert workflow.count(
        "actions/download-artifact@634f93cb2916e3fdff6788551b99b062d0335ce0"
    ) == 9


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


def test_candidate_and_release_heavy_commands_use_memory_watchdog() -> None:
    candidate = (WORKFLOWS / "candidate.yml").read_text(encoding="utf-8")
    release = (WORKFLOWS / "release-artifacts.yml").read_text(encoding="utf-8")

    assert candidate.count(MEMORY_WATCHDOG) == 19
    assert (
        _guarded_count(
            candidate,
            r'(?:python|"\$PYTHON") dependencies/install_dependencies\.py',
        )
        == 5
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
            r'(?:\.venv/bin/python|python|"\$PYTHON") '
            r"tools/release/test_deployment\.py",
        )
        == 3
    )
    assert (
        _guarded_count(
            candidate,
            r"\.venv/bin/python "
            r"tools/developer/compiled_all_jit_arena_gate\.py",
        )
        == 1
    )
    assert (
        _guarded_count(
            candidate,
            r"\.venv/bin/python tools/developer/four_quark_compiled_gate\.py",
        )
        == 1
    )
    assert (
        _guarded_count(
            candidate,
            r"\.venv/bin/python tools/developer/eager_benchmark_matrix\.py",
        )
        == 1
    )
    assert (
        _guarded_count(
            candidate,
            r'(?:\.venv/bin/python|python|"\$PYTHON") '
            r"tools/release/build_release_artifacts\.py",
        )
        == 3
    )
    assert (
        f"{MEMORY_WATCHDOG} \\\n"
        '            "$BASELINE_SOURCE_ROOT/.venv/bin/python" \\\n'
        '            "$BASELINE_SOURCE_ROOT/tools/release/'
        'build_release_artifacts.py"'
    ) in candidate

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
