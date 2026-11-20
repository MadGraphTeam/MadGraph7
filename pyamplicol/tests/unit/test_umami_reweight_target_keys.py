# SPDX-License-Identifier: 0BSD
"""Reweight target routes carry the process id of their flavour-union assignment."""

from __future__ import annotations

import pytest

from pyamplicol.artifacts.umami_reweight import _target_routes


def test_target_routes_need_the_assignment_key() -> None:
    member = {
        "process_id": "d_g_to_d_g",
        "pdgs": [1, 21, 1, 21],
        "runtime": {
            "process_index": 0,
            "permutation": [0, 1, 2, 3],
            "factor": 1.0,
            "assignment": 1,
            "assignment_key": "d_g_to_d_g",
        },
    }
    document = {
        "provider": {"artifact_id": "a"},
        "runtime_processes": [{"id": "u_g_to_u_g"}],
        "channels": [{"processes": [{"members": [member]}]}],
    }
    (route,) = _target_routes([document])
    assert route[5:] == (1, "d_g_to_d_g")
    del member["runtime"]["assignment_key"]
    with pytest.raises(ValueError, match="without its process id"):
        _target_routes([document])
