# SPDX-License-Identifier: 0BSD
"""Public virtual artifact transport, exact ingress and arithmetic selection."""

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.api.errors import CompatibilityError
from pyamplicol.runtime.virtual import _inputs, _parameters

pytestmark = pytest.mark.loop_fast


def test_exact_input_transport_and_rejection():
    precise = Decimal("0.118000000000000000000000000000000000001")
    assert _parameters({"aS": precise, "mass": ("1.2", "1e-80")}) == {
        "aS": [str(precise), "0"],
        "mass": ["1.2", "1E-80"],
    }
    assert _inputs([[[precise, "0", 0, 0.0]]], ["1e80"], 1) == (
        [[[str(precise), "0", "0", "0.0"]]],
        ["1E+80"],
    )
    assert _inputs([], "1", 1) == ([], [])
    assert _inputs(np.array([[[1, 0, 0, 1]]]), np.array(["1e5"]), 1) == (
        [[["1", "0", "0", "1"]]],
        ["1E+5"],
    )
    for bad in (True, complex(1, 0), "nan", "inf", "bad"):
        with pytest.raises((ValueError, TypeError)):
            _inputs([[[bad, 0, 0, 0]]], "1", 1)
    with pytest.raises(CompatibilityError, match="mu_r2"):
        _inputs([], None, 1)
    for bad in (0, -1, True, "NaN", "Infinity"):
        with pytest.raises((ValueError, TypeError)):
            _inputs([], bad, 1)
    with pytest.raises(ValueError, match="one entry"):
        _inputs([], ["1"], 1)
    with pytest.raises(ValueError, match="four-vectors"):
        _inputs([[[1, 2, 3]]], "1", 1)
    with pytest.raises(TypeError, match="mapping"):
        _inputs([[[1, 0, 0, 1]]], {1: "1000000"}, 1)


@pytest.fixture(scope="module")
def source_artifact(saved_loop_fixture, loop_native):
    path = str(saved_loop_fixture("source-public-v3/gg"))
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/gluon_scattering_qcd.json"
    )
    points = json.loads(fixture.read_text(), parse_int=str, parse_float=str)["points"]
    return Path(path), points
