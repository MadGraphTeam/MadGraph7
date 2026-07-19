# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from decimal import Decimal

import pytest

from pyamplicol.artifacts.api_bundle import (
    api_bundle_payloads,
    format_validation_points,
)


def test_api_bundle_has_one_complete_root_layout() -> None:
    payloads = api_bundle_payloads(
        {
            "ddbar_to_zg": (
                (500.0, 0.0, 0.0, 500.0),
                (500.0, 0.0, 0.0, -500.0),
                (Decimal("500.0"), "1.25", 2, -3),
                (500.0, -1.25, -2.0, 3.0),
            )
        }
    )
    assert {payload.path for payload in payloads} == {
        "API/validation_points.dat",
        "API/python/check_standalone.py",
        "API/c/check_standalone.c",
        "API/c/Makefile",
        "API/cpp/check_standalone.cpp",
        "API/cpp/Makefile",
        "API/fortran/check_standalone.f90",
        "API/fortran/Makefile",
        "API/rust/check_standalone.rs",
        "API/rust/Makefile",
    }
    python = next(payload for payload in payloads if payload.path.endswith(".py"))
    assert python.executable is True
    point = next(payload for payload in payloads if payload.path.endswith(".dat"))
    assert point.content.startswith(b"RUSTICOL_VALIDATION_POINTS_V1\n")
    assert b"ddbar_to_zg\t4\t" in point.content
    makefiles = {
        payload.path: payload.content.decode("utf-8")
        for payload in payloads
        if payload.path.endswith("Makefile")
    }
    assert all("/.pyamplicol-api-build/" in text for text in makefiles.values())
    assert all('cd "$(ARTIFACT_DIR)"' in text for text in makefiles.values())
    assert all("API/cpp/check_standalone" not in text for text in makefiles.values())
    assert "CC ?= cc" in makefiles["API/c/Makefile"]
    assert "CXX ?= c++" in makefiles["API/cpp/Makefile"]
    assert "FC = gfortran" in makefiles["API/fortran/Makefile"]
    rust_makefile = makefiles["API/rust/Makefile"]
    assert "RUSTC ?= rustc" in rust_makefile
    assert "$(RUSTICOL_CONFIG) --rust-source" in rust_makefile
    assert "$(RUSTICOL_CONFIG) --rustflags" in rust_makefile
    assert "$(RUSTICOL_CONFIG) --cargo-rustflags" in rust_makefile
    assert "RUSTICOL_RUST_SOURCE" in rust_makefile
    assert "CARGO_ENCODED_RUSTFLAGS" in rust_makefile
    assert "run-script:" in rust_makefile
    assert "TARGET := $(BUILD_DIR)/check_standalone" in rust_makefile
    assert "$(RUSTC) --edition=2021" in rust_makefile
    assert "\tcargo " not in rust_makefile.lower()
    assert "Cargo.toml" not in rust_makefile

    rust_source = next(
        payload.content.decode("utf-8")
        for payload in payloads
        if payload.path == "API/rust/check_standalone.rs"
    )
    assert "#!/usr/bin/env rust-script" in rust_source
    assert 'include!(env!("RUSTICOL_RUST_SOURCE"))' in rust_source
    assert "Runtime::load" in rust_source
    assert "Selectors::all()" in rust_source
    assert ".set_model_parameters(&options.overrides)" in rust_source
    assert ".evaluate_f64(&momenta, 1)" in rust_source
    assert ".evaluate_resolved_f64(&momenta, 1" in rust_source
    assert r"\"language\":\"rust\"" in rust_source
    assert "unsafe" not in rust_source
    assert 'extern "C"' not in rust_source
    assert "rusticol_runtime_" not in rust_source
    assert "extern crate" not in rust_source
    assert "serde" not in rust_source

    c_source = next(
        payload.content.decode("utf-8")
        for payload in payloads
        if payload.path == "API/c/check_standalone.c"
    )
    assert "#include <rusticol.h>" in c_source
    assert "rusticol_runtime_load" in c_source
    assert "rusticol_runtime_evaluate_f64" in c_source
    assert "rusticol_runtime_evaluate_resolved_f64" in c_source


def test_validation_points_are_sorted_and_require_four_vectors() -> None:
    output = format_validation_points(
        {
            "second": ((1, 2, 3, 4),),
            "first": ((5, 6, 7, 8),),
        }
    ).decode("ascii")
    assert output.splitlines()[1].startswith("first\t")
    assert output.splitlines()[2].startswith("second\t")
    with pytest.raises(ValueError, match="four-vectors"):
        format_validation_points({"broken": ((1, 2, 3),)})
    with pytest.raises(ValueError, match="non-empty tokens"):
        format_validation_points({"not valid": ((1, 2, 3, 4),)})
