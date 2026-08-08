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


def test_native_sdk_wrappers_expose_process_permutations_and_kinematics() -> None:
    header = _read("rust/crates/rusticol-capi/include/rusticol.h")
    cpp = _read("rust/crates/rusticol-capi/include/rusticol.hpp")
    fortran = _read("rust/crates/rusticol-capi/fortran/rusticol.f90")
    rust = _read("src/pyamplicol/_sdk/rust/rusticol.rs")

    for symbol in (
        "rusticol_runtime_representative_process_key",
        "rusticol_runtime_external_permutation",
        "rusticol_runtime_load_kinematics_json",
    ):
        assert symbol in header
        assert symbol in fortran
        assert symbol in rust
    assert "std::string representative_process_key() const" in cpp
    assert "std::vector<std::size_t> external_permutation() const" in cpp
    assert "std::vector<double> load_kinematics_json" in cpp
    assert "procedure, public :: representative_process_key" in fortran
    assert "procedure, public :: external_permutation" in fortran
    assert "procedure, public :: load_kinematics_json" in fortran
    assert "pub fn representative_process_key(&self) -> Result<String>" in rust
    assert "pub fn external_permutation(&self) -> Result<Vec<usize>>" in rust
    assert "pub fn load_kinematics_json" in rust


def test_native_sdk_wrappers_expose_per_point_runtime_selectors() -> None:
    header = _read("rust/crates/rusticol-capi/include/rusticol.h")
    cpp = _read("rust/crates/rusticol-capi/include/rusticol.hpp")
    fortran = _read("rust/crates/rusticol-capi/fortran/rusticol.f90")
    rust = _read("src/pyamplicol/_sdk/rust/rusticol.rs")

    assert "rusticol_runtime_evaluate_selected_f64" in header
    assert "evaluate_selected" in cpp
    assert "procedure, public :: evaluate_selected" in fortran
    assert "pub fn evaluate_selected_f64" in rust
    assert "helicity_by_point: Option<&[u32]>" in rust
    assert "color_flow_by_point: Option<&[u32]>" in rust


def test_native_sdk_wrappers_expose_structured_one_point_otf_warm_up() -> None:
    header = _read("rust/crates/rusticol-capi/include/rusticol.h")
    cpp = _read("rust/crates/rusticol-capi/include/rusticol.hpp")
    fortran = _read("rust/crates/rusticol-capi/fortran/rusticol.f90")
    rust = _read("src/pyamplicol/_sdk/rust/rusticol.rs")

    assert "rusticol_runtime_warm_up_f64" in header
    assert "RusticolWarmUpProgressCallback" in header
    assert "RusticolWarmUpProgressEvent" in header
    assert "RusticolWarmUpResult" in header
    assert "WarmUpResult warm_up(" in cpp
    assert "std::function<bool(const WarmUpProgress &)>" in cpp
    assert "procedure, public :: warm_up => rusticol_warm_up" in fortran
    assert 'bind(C, name="rusticol_runtime_warm_up_f64")' in fortran
    assert "type, bind(C), public :: rusticol_warm_up_progress_event" in fortran
    assert "pub fn warm_up(" in rust
    assert "pub fn warm_up_f64(" in rust
    assert "FnMut(&WarmUpProgress) -> bool" in rust


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
