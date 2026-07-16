#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Required source-distribution inventory for release audits."""

from __future__ import annotations

from collections.abc import Collection

REQUIRED_SDIST_MEMBERS = frozenset(
    {
        "Cargo.lock",
        "Cargo.toml",
        "rust-toolchain.toml",
        "LICENSE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "justfile",
        "build_backend/_pyamplicol_build.py",
        "build_backend/python_lock.py",
        "build_backend/sdk.py",
        "config/release-dependencies.toml",
        "dependencies/install_dependencies.py",
        "dependencies/patches/symbolica/0001-fix-complex-export-aarch64.patch",
        "dependencies/patches/symbolica/0002-release-gil-for-multiple-evaluator-build.patch",
        "dependencies/patches/symbolica/0003-native-aarch64-complex-jit-and-opt-level-forwarding.patch",
        "dependencies/patches/symbolica/0004-export-symjit-application.patch",
        "dependencies/patches/symjit/0001-fix-aarch64-o3-complex-register-allocation.patch",
        "dependencies/patches/symjit/0002-fix-aarch64-external-call-spill-offset.patch",
        "dependencies/patches/symjit/0003-fix-aarch64-large-stack-adjustments.patch",
        "dependencies/patches/symjit/0004-fix-aarch64-long-conditional-branches.patch",
        "dependencies/python-runtime-lock.toml",
        "dependencies/release-lock.toml",
        "docs/pyAmpliCol.tex",
        "docs/user/installation.md",
        "examples/builtin_sm_lc.toml",
        "examples/native/Makefile",
        "examples/python/typed_generation.py",
        "licenses/RUST_THIRD_PARTY.toml",
        "licenses/STATIC_LINK_COMPLIANCE.toml",
        "pyproject.toml",
        "rust/crates/rusticol-capi/fortran/rusticol.f90",
        "rust/crates/rusticol-capi/include/rusticol.h",
        "rust/crates/rusticol-capi/include/rusticol.hpp",
        "rust/crates/rusticol-python/stubs/pyamplicol/_rusticol.pyi",
        "schemas/README.md",
        "schemas/artifact-manifest-v3.schema.json",
        "schemas/runtime-physics-v1.schema.json",
        "src/pyamplicol/assets/selftest/portable-64le/expected.json",
        "src/pyamplicol/assets/selftest/portable-64le/artifact/artifact.json",
        "tests/integration/test_examples.py",
        "tests/release/test_artifacts.py",
        "tests/unit/test_cli_utilities.py",
        "tools/release/_artifacts.py",
        "tools/release/_common.py",
        "tools/release/audit_sdist.py",
        "tools/release/build_from_sdist.py",
        "tools/release/build_release_artifacts.py",
        "tools/release/check_dependencies.py",
        "tools/release/check_legal_inventory.py",
        "tools/release/check_rust_licenses.py",
        "tools/release/install_wheel.py",
        "tools/release/publish_dry_run.py",
        "tools/release/run_cargo.py",
        "tools/release/test_deployment.py",
        "tools/typing/check_public_typing.py",
    }
)


def missing_required_sdist_members(members: Collection[str]) -> tuple[str, ...]:
    """Return required release members absent from an extracted sdist."""

    return tuple(sorted(REQUIRED_SDIST_MEMBERS.difference(members)))


__all__ = ["REQUIRED_SDIST_MEMBERS", "missing_required_sdist_members"]
