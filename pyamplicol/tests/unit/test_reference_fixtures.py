# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import tomllib
from math import fsum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = ROOT / "tests" / "fixtures" / "reference"
FIXTURE = REFERENCE_ROOT / "physics-v1.json"
CAPTURE = REFERENCE_ROOT / "CAPTURE.toml"
LEGACY_FORTRAN = REFERENCE_ROOT / "legacy-fortran-v1.json"
BASELINE = ROOT / "docs" / "development" / "SOURCE_BASELINE.toml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reference_fixture_is_compact_and_self_consistent() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["fixture_schema_version"] == 1
    cases = payload["cases"]
    assert {
        "builtin_sm_ddbar_z_lc",
        "builtin_sm_ddbar_z_nlc",
        "builtin_sm_ddbar_z_full",
        "builtin_sm_ddbar_zg_lc",
        "builtin_sm_ddbar_zg_nlc",
        "builtin_sm_ddbar_zg_full",
        "scalars_2to2_lc",
        "scalar_gravity_2to2_lc",
    } == set(cases)

    for case in cases.values():
        resolved = case.get("resolved")
        if resolved is None:
            resolved = cases[case["resolved_from"]]["resolved"]
        resolved_total = fsum(
            value for colors in resolved.values() for value in colors.values()
        )
        assert resolved_total == case["total"] or abs(
            resolved_total - case["total"]
        ) <= 2.0e-15 * max(abs(case["total"]), 1.0)


def test_reference_capture_matches_the_pinned_clean_baseline() -> None:
    with CAPTURE.open("rb") as stream:
        capture = tomllib.load(stream)
    with BASELINE.open("rb") as stream:
        baseline = tomllib.load(stream)

    assert capture["schema_version"] == 1
    assert capture["source"]["revision"] == baseline["source"]["commit"]
    assert capture["source"]["relevant_paths_clean"] is True
    assert capture["generation"]["memory_watchdog_gb"] == 30
    assert {entry["case"] for entry in capture["generation"]["captures"]} == {
        "builtin_sm_ddbar_z_nlc",
        "builtin_sm_ddbar_z_full",
    }
    assert baseline["reference_fixtures"]["clean_fixture_sha256"] == _sha256(FIXTURE)
    assert baseline["reference_fixtures"]["capture_manifest_sha256"] == _sha256(CAPTURE)
    assert baseline["reference_fixtures"]["legacy_fortran_fixture_sha256"] == _sha256(
        LEGACY_FORTRAN
    )
