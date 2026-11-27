# SPDX-License-Identifier: 0BSD
"""Frozen legacy oracles (O1, O2) and three-provider slice references (WP0.1).

The checks only read committed fixtures: the legacy captures are frozen (the
legacy engine that wrote them was removed, user ruling 2026-10-28, M-QCD Q3).
"""

from __future__ import annotations

import math
from decimal import Decimal
from itertools import combinations

import pytest

from tools.loop_reference import legacy_capture

TOLERANCE = 1e-9
ALPHA_S = 0.118
LAURENT = ("finite", "single_pole", "double_pole")
SLICE_REFERENCES = ("uux_ddxg", "uux_uux", "uux_uuxg")
# legacy component -> key of the instrumented MadLoop component controls
COMPONENT_KEYS = {
    "virtual": "full",
    "bare": "bare",
    "uv": "uv",
    "r2": "r2",
    "uv_local": "uv_vertex",
    "uv_external_fields": "wf",
}
COMPONENT_CONTROLS = {
    "gg_gg": "gluon-scattering-component-control",
    "uudd": "uudd-component-control",
    "tt": "top-pair-component-control",
    "uux_ggg": "quark-three-gluon-component-control",
}
# A control without the UV split has no external-field counterterm (massless
# external quarks only), so its whole UV part is the vertex counterterm.
UV_SPLIT = ("uv_vertex", "wf")
# The slowest control (uux_ggg) regenerates in about 400 s on this host.


def relative(a, b, floor=0.0):
    """The required-references metric: |a-b| / max(|a|, |b|, floor)."""
    scale = max(abs(a), abs(b), floor)
    return 0.0 if scale == 0 else abs(a - b) / scale


def unit(born):
    """Floor for Laurent coefficients: one natural loop unit |Born| alpha_s/(2 pi)."""
    return abs(born) * ALPHA_S / (2 * math.pi)


def same_points(left, right):
    return [[[Decimal(str(x)) for x in leg] for leg in point] for point in left] == [
        [[Decimal(str(x)) for x in leg] for leg in point] for point in right
    ]


def legacy_rows(oracle, carrier):
    """Per point: Born and {component: {pole: value}} of the single order."""
    results = oracle["results"][carrier]
    assert len(oracle["orders"]["born"]) == len(oracle["orders"]["virtual"]) == 1
    rows = []
    for index in range(len(oracle["points"])):
        row = {"born": float(results["born"][index][0][0])}
        for name in legacy_capture.COMPONENTS:
            row[name] = {
                pole: float(value[0])
                for pole, value in zip(LAURENT, results[name][index][0], strict=True)
            }
        rows.append(row)
    return rows


def assert_agrees(expected, actual, context):
    """Born strictly relative; Laurent coefficients with the loop-unit floor."""
    floor = unit(max(abs(expected["born"]), abs(actual["born"])))
    error = relative(expected["born"], actual["born"])
    assert error <= TOLERANCE, f"{context} born: {error:.3g}"
    for pole in LAURENT:
        error = relative(expected[pole], actual[pole], floor)
        assert error <= TOLERANCE, f"{context} {pole}: {error:.3g}"


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", SLICE_REFERENCES)
def test_slice_providers_agree(loop_reference, name):
    reference = loop_reference(f"required/{name}")
    assert reference["validated_providers"] == ["madloop", "recola", "openloops"]
    for index, row in enumerate(reference["rows"]):
        providers = row["providers"]
        for a, b in combinations(reference["validated_providers"], 2):
            assert_agrees(providers[a], providers[b], f"{name} point {index} {a}-{b}")


@pytest.mark.loop_fast
@pytest.mark.parametrize("carrier", tuple(legacy_capture.CARRIERS))
@pytest.mark.parametrize("name", SLICE_REFERENCES)
def test_legacy_agrees_with_slice_providers(loop_reference, name, carrier):
    reference = loop_reference(f"required/{name}")
    oracle = loop_reference(f"legacy/{name}")
    assert same_points(oracle["points"], reference["points"])
    assert [Decimal(s) for s in oracle["scales_squared"]] == [
        Decimal(mu) ** 2 for mu in reference["mu_r"]
    ]
    for index, (legacy, row) in enumerate(
        zip(legacy_rows(oracle, carrier), reference["rows"], strict=True)
    ):
        actual = {"born": legacy["born"], **legacy["virtual"]}
        for provider in reference["validated_providers"]:
            assert_agrees(
                row["providers"][provider], actual, f"{name} point {index} {provider}"
            )


@pytest.mark.loop_fast
@pytest.mark.parametrize("carrier", tuple(legacy_capture.CARRIERS))
@pytest.mark.parametrize("name", tuple(COMPONENT_CONTROLS))
def test_legacy_components_agree_with_madloop_controls(loop_reference, name, carrier):
    reference = loop_reference(COMPONENT_CONTROLS[name])
    oracle = loop_reference(f"legacy/{name}")
    assert same_points(oracle["points"], reference["fixture"]["points"])
    assert set(oracle["scales_squared"]) == {"1000000"}
    for index, (legacy, row) in enumerate(
        zip(legacy_rows(oracle, carrier), reference["rows"], strict=True)
    ):
        if not any(key in row for key in UV_SPLIT):
            row = {**row, "uv_vertex": row["uv"], "wf": dict.fromkeys(LAURENT, 0.0)}
        for component, key in COMPONENT_KEYS.items():
            assert_agrees(
                {"born": row["born"], **row[key]},
                {"born": legacy["born"], **legacy[component]},
                f"{name} point {index} {component}",
            )


@pytest.mark.loop_fast
@pytest.mark.parametrize("carrier", tuple(legacy_capture.CARRIERS))
def test_legacy_drell_yan_agrees_with_madloop(loop_reference, carrier):
    reference = loop_reference("drell-yan-control")
    oracle = loop_reference("legacy/dy")
    assert same_points(oracle["points"], reference["points"])
    for index, (legacy, row) in enumerate(
        zip(legacy_rows(oracle, carrier), reference["results"], strict=True)
    ):
        assert_agrees(
            row, {"born": legacy["born"], **legacy["virtual"]}, f"dy point {index}"
        )
