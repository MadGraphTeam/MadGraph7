# SPDX-License-Identifier: 0BSD
"""Environment-gated real numerical acceptance across all fifteen UFO-SM lanes."""

from __future__ import annotations

import importlib.util
import os

import pytest

from tools.developer.numerical_acceptance import (
    DEFAULT_FIXTURE,
    EXTRA_FULL_COLOUR_CASES,
    NumericalAcceptanceHarness,
    catalog_cases,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("PYAMPLICOL_RUN_NUMERICAL_ACCEPTANCE") != "1",
    reason=(
        "set PYAMPLICOL_RUN_NUMERICAL_ACCEPTANCE=1 to run the full numerical "
        "acceptance matrix"
    ),
)


def _unavailable(reason: str) -> None:
    if os.environ.get("PYAMPLICOL_REQUIRE_NATIVE_TESTS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def _require_native_dependencies() -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        _unavailable("the Rusticol extension has not been built")
    if importlib.util.find_spec("symbolica") is None:
        _unavailable("Symbolica is unavailable")


@pytest.fixture(scope="module")
def acceptance(
    tmp_path_factory: pytest.TempPathFactory,
) -> NumericalAcceptanceHarness:
    _require_native_dependencies()
    if not DEFAULT_FIXTURE.is_file():
        pytest.fail(
            f"the numerical acceptance fixture has not been captured: {DEFAULT_FIXTURE}"
        )
    return NumericalAcceptanceHarness.prepare(
        DEFAULT_FIXTURE,
        work_root=tmp_path_factory.mktemp("numerical-acceptance"),
    )


@pytest.mark.parametrize(
    "case_id",
    tuple(case.case_id for case in catalog_cases()),
)
def test_catalog_numerical_acceptance_case(
    acceptance: NumericalAcceptanceHarness,
    case_id: str,
) -> None:
    acceptance.assert_catalog_case(case_id)


@pytest.mark.parametrize(
    "case_id",
    tuple(case.case_id for case in EXTRA_FULL_COLOUR_CASES),
)
def test_non_catalog_full_colour_madgraph_benchmark(
    acceptance: NumericalAcceptanceHarness,
    case_id: str,
) -> None:
    acceptance.assert_full_extra(case_id)
