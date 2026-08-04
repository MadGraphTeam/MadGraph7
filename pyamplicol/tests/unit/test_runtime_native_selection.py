# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json

import pytest

from pyamplicol.api.errors import ArtifactError
from pyamplicol.runtime._native_selection import (
    native_physics_axes,
    native_process_selection,
    remap_reduction,
    representative_vector_to_public,
)


class _NativeRuntime:
    def __init__(
        self,
        payload: dict[str, object],
        *,
        physics: dict[str, object] | None = None,
    ) -> None:
        self._payload = payload
        self._physics = physics

    def _exact_runtime_state_json(self) -> str:
        return json.dumps(self._payload)

    def physics_json(self) -> str:
        return json.dumps(self._physics)


def _state(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_parameter_values": [],
        "normalization_factor": 1.0,
        "representative_process_id": "representative",
        "representative_process_key": "representative-key",
        "external_permutation": [1, 0, 3, 2],
    }
    payload.update(updates)
    return payload


_PROCESSES = (
    {
        "id": "representative",
        "external_pdgs": [2, -2, 23, 21],
        "aliases": [],
    },
)


def test_native_selection_is_authoritative_for_inferred_public_expression() -> None:
    selection = native_process_selection(_NativeRuntime(_state()), _PROCESSES)

    assert selection.process is _PROCESSES[0]
    assert selection.representative_process_id == "representative"
    assert selection.representative_process_key == "representative-key"
    assert selection.external_permutation == (1, 0, 3, 2)


@pytest.mark.parametrize(
    "updates",
    [
        {"representative_process_id": "missing"},
        {"representative_process_key": ""},
        {"external_permutation": [0, 1, 2]},
        {"external_permutation": [0, 1, 2, 2]},
        {"external_permutation": [0, 1, 2, True]},
    ],
)
def test_native_selection_rejects_untrusted_bridge_state(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ArtifactError):
        native_process_selection(_NativeRuntime(_state(**updates)), _PROCESSES)


def test_native_physics_axes_remap_representative_reductions() -> None:
    representative = {
        "color_accuracy": "lc",
        "helicities": [{"id": "h:-1,+1"}, {"id": "h:+1,-1"}],
        "color_components": [{"id": "flow:1,2"}, {"id": "flow:2,1"}],
    }
    public = {
        "color_accuracy": "lc",
        "helicities": [{"id": "h:+1,-1"}, {"id": "h:-1,+1"}],
        "color_components": [{"id": "flow:2,1"}, {"id": "flow:1,2"}],
    }
    axes = native_physics_axes(
        _NativeRuntime(_state(), physics=public), representative
    )

    assert axes.public_physics == public
    assert remap_reduction(
        {
            "groups": [
                {
                    "id": "reduction:0",
                    "representative_helicity_id": "h:-1,+1",
                    "representative_color_id": "flow:1,2",
                    "physical_helicity_ids": ["h:-1,+1", "h:+1,-1"],
                    "physical_color_ids": ["flow:1,2", "flow:2,1"],
                }
            ]
        },
        axes,
    ) == {
        "groups": [
            {
                "id": "reduction:0",
                "representative_helicity_id": "h:+1,-1",
                "representative_color_id": "flow:2,1",
                "physical_helicity_ids": ["h:+1,-1", "h:-1,+1"],
                "physical_color_ids": ["flow:2,1", "flow:1,2"],
            }
        ]
    }
    assert representative_vector_to_public((-1, 1), (1, 0)) == (1, -1)


def test_native_physics_axes_reject_missing_reduction_selector() -> None:
    representative = {
        "color_accuracy": "lc",
        "helicities": [{"id": "h:-1"}],
        "color_components": [{"id": "flow:1"}],
    }
    public = {
        "color_accuracy": "lc",
        "helicities": [{"id": "h:+1"}],
        "color_components": [{"id": "flow:1"}],
    }
    axes = native_physics_axes(
        _NativeRuntime(_state(), physics=public), representative
    )

    with pytest.raises(ArtifactError, match="absent from its axis"):
        remap_reduction(
            {
                "groups": [
                    {
                        "representative_helicity_id": "h:missing",
                        "representative_color_id": "flow:1",
                        "physical_helicity_ids": ["h:-1"],
                        "physical_color_ids": ["flow:1"],
                    }
                ]
            },
            axes,
        )
