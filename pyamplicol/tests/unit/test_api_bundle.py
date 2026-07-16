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
        "API/cpp/check_standalone.cpp",
        "API/cpp/Makefile",
        "API/fortran/check_standalone.f90",
        "API/fortran/Makefile",
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
    assert "CXX = c++" in makefiles["API/cpp/Makefile"]
    assert "FC = gfortran" in makefiles["API/fortran/Makefile"]


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
