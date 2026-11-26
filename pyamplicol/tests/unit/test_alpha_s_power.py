# SPDX-License-Identifier: 0BSD
# ruff: noqa: RUF001
"""The alpha_s homogeneity proof and its UMAMI metadata."""

from __future__ import annotations

import pytest

from pyamplicol.artifacts.umami import _alpha_s_power
from pyamplicol.artifacts.umami_c import _power
from pyamplicol.generation.alpha_s_power import _twice_degree, alpha_s_parameter

symbolica = pytest.importorskip("symbolica")


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        ("2𝑖*test::{}::aS^(1/2)*𝜋^(1/2)", 1),
        ("-4*test::{}::aS*𝜋", 2),
        ("test::{}::aS^(1/2)*test::{}::MT*(1+test::{}::aEWM1)^(-1)", 1),
        ("8𝑖*test::{}::aEWM1^(-1)*𝜋", 0),
        ("(4*test::{}::aS+3)^(1/2)", None),
        ("test::{}::aS^(1/3)", None),
        ("test::{}::aS^(1/2)+test::{}::aS", None),
    ),
)
def test_twice_degree_uses_euler_identity(expression, expected):
    assert _twice_degree(expression, symbolica.E("test::{}::aS")) == expected


def test_alpha_s_parameter_is_the_unique_sminputs_three():
    records = [
        {"name": "aS", "nature": "external", "lhablock": "SMINPUTS", "lhacode": [3]},
        {"name": "aEWM1", "nature": "external", "lhablock": "SMINPUTS", "lhacode": [1]},
    ]
    assert alpha_s_parameter(records) == "aS"
    assert alpha_s_parameter([*records, {**records[0], "name": "other"}]) is None
    assert alpha_s_parameter(records, "strong") == "strong"


def test_metadata_keeps_only_a_proof_of_its_alpha_s_parameter():
    physics = {"extensions": {"alpha_s_power": {"parameter": "aS", "power": 3}}}
    assert _alpha_s_power(physics, "aS") == 3
    assert _alpha_s_power(physics, "other") is None
    assert _alpha_s_power(physics, None) is None
    assert _alpha_s_power({"extensions": {}}, "aS") is None
    assert _power({"alpha_s_power": 3}) == "3"
    assert _power({"alpha_s_power": None}) == "-1"
    assert _power({}) == "-1"
