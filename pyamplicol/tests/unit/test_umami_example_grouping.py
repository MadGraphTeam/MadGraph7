# SPDX-License-Identifier: 0BSD
"""Check example bookkeeping without installing optional integration packages."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def helpers():
    directory = Path(__file__).resolve().parents[2] / "umami_example"
    namespace = {
        "__name__": __name__,
        "dataclass": dataclass,
        "field": field,
        "math": math,
        "Path": Path,
    }
    for filename, names in (
        ("test_integrate.py", {"Channel", "Config", "Cuts", "build_ms_cuts"}),
        ("test_grouping.py", {"physical_member_channels", "check_agreement"}),
    ):
        path = directory / filename
        definitions = [
            node
            for node in ast.parse(path.read_text(), filename=str(path)).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
        ]
        assert len(definitions) == len(names)
        exec(
            compile(
                ast.Module(body=definitions, type_ignores=[]),
                str(path),
                "exec",
                dont_inherit=True,
            ),
            namespace,
        )
    return namespace


def test_member_channels_compose_maps_and_keep_physical_weights(helpers):
    representative = [0, 1, 4, 2, 3]
    member_map = [0, 1, 3, 2, 4]
    pdgs = [1, -1, 21, 21, 21]
    member = {
        "runtime": {"permutation": member_map},
        "pdgs": pdgs,
        "factor": 0.5,
    }
    entry = {
        "runtime": {"permutation": representative},
        "matrix_elements": [{"factor": 17.0}],
        "members": [member, {**member, "factor": 1.0}],
    }
    data = {
        "provider": {"particle_count": 5, "masses": [0.0] * 5},
        "channels": [
            {"processes": []},
            {
                "phasespace_order": [0, 2, 1, 3, 4],
                "processes": [entry, {**entry, "members": [member]}],
            },
        ],
    }
    channels = helpers["physical_member_channels"](data)
    assert [ch.index for ch in channels] == [0, 1, 2]
    assert [ch.selector for ch in channels] == [(1, 0), (1, 0), (1, 1)]
    assert [ch.processes[0][0][0] for ch in channels] == [0.5, 1.0, 0.5]
    for channel in channels:
        permutation = channel.momentum_permutation
        assert permutation == [0, 1, 2, 4, 3]
        assert [permutation[i] for i in representative] == member_map
        assert permutation[:2] == [0, 1]
        assert channel.color_order == [0, 2, 1, 4, 3]
        assert channel.init_states == [(1, -1)]
        assert channel.pdg_final == pdgs[2:]


@pytest.mark.parametrize(
    ("value", "error", "failure"),
    [
        (10.02, 0.1, None),
        (12.0, 0.1, "differ"),
        (10.0, 0.6, "inconclusive"),
        (math.nan, 0.1, "positive finite"),
        (10.0, 0.0, "positive finite"),
    ],
)
def test_integral_agreement(helpers, value, error, failure):
    results = {
        "grouped": {"integral": value, "uncertainty": error},
        "physical_members": {"integral": 10.0, "uncertainty": 0.1},
    }
    if failure:
        with pytest.raises(AssertionError, match=failure):
            helpers["check_agreement"](results)
    else:
        comparison = helpers["check_agreement"](results)
        assert comparison["combined_error"] == pytest.approx(math.hypot(error, 0.1))
        assert comparison["standard_errors"] < 6.0


def test_species_selective_cuts_are_rejected(helpers):
    cfg = helpers["Config"](cuts=helpers["Cuts"](pdgs=[21]))
    with pytest.raises(ValueError, match="flavour-blind"):
        helpers["build_ms_cuts"](cfg, None)
