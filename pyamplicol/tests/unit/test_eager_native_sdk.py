# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_native_sdk_wrappers_expose_the_runtime_execution_mode() -> None:
    header = _read("rust/crates/rusticol-capi/include/rusticol.h")
    cpp = _read("rust/crates/rusticol-capi/include/rusticol.hpp")
    fortran = _read("rust/crates/rusticol-capi/fortran/rusticol.f90")
    rust = _read("src/pyamplicol/_sdk/rust/rusticol.rs")

    assert "rusticol_runtime_execution_mode" in header
    assert "std::string execution_mode() const" in cpp
    assert "procedure, public :: execution_mode" in fortran
    assert 'bind(C, name="rusticol_runtime_execution_mode")' in fortran
    assert "pub fn execution_mode(&self) -> Result<String>" in rust
    assert "rusticol_runtime_execution_mode" in rust


def test_generated_native_drivers_share_total_and_resolved_entrypoints() -> None:
    templates = {
        "cpp": _read("src/pyamplicol/assets/api_templates/cpp/check_standalone.cpp"),
        "fortran": _read(
            "src/pyamplicol/assets/api_templates/fortran/check_standalone.f90"
        ),
        "rust": _read("src/pyamplicol/assets/api_templates/rust/check_standalone.rs"),
    }

    for language, source in templates.items():
        assert "evaluate" in source, language
        assert "evaluate_resolved" in source, language
        assert "compatibility_total" in source, language
        assert '"eager"' not in source, language
        assert "resolved_available" not in source, language


def test_compiled_driver_paths_still_check_resolved_sums() -> None:
    sources = (
        _read("src/pyamplicol/assets/api_templates/cpp/check_standalone.cpp"),
        _read("src/pyamplicol/assets/api_templates/fortran/check_standalone.f90"),
        _read("src/pyamplicol/assets/api_templates/rust/check_standalone.rs"),
    )

    for source in sources:
        assert "resolved components do not reproduce the compatibility total" in source
