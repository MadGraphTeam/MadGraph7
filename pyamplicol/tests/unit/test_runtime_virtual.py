# SPDX-License-Identifier: 0BSD
"""Public virtual artifact transport, exact ingress and arithmetic selection."""

from dataclasses import asdict, replace
from decimal import Decimal

import numpy as np
import pytest

from pyamplicol.api.errors import CompatibilityError
from pyamplicol.api.virtual_results import (
    VirtualFailureDiagnostic,
    VirtualLaneDiagnostic,
)
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


def test_failure_diagnostics_are_frozen_and_accuracy_status_is_invalid():
    def failure(point):
        return VirtualFailureDiagnostic(
            point,
            1,
            2,
            7,
            "reduction",
            "partial transport control",
            color_component_index=2,
        )

    values = [failure(1)]
    diagnostic = VirtualLaneDiagnostic(
        "accuracy-not-established", "double", 53, False, failures=values
    )
    values.clear()
    assert diagnostic.failures == (failure(1),) and not diagnostic.valid
    assert asdict(diagnostic)["failures"][0]["point_id"] == 1
    with pytest.raises(ValueError, match="unrecovered"):
        replace(diagnostic, status="success")
