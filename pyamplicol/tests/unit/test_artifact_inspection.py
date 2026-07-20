# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path

from pyamplicol.artifacts import inspect_artifact
from pyamplicol.artifacts.inspection import _computed

ROOT = Path(__file__).resolve().parents[2]
PORTABLE_ARTIFACT = ROOT / "src/pyamplicol/assets/selftest/portable-64le/artifact"


def test_artifact_inspection_lists_processes_without_loading_evaluators() -> None:
    inspection = inspect_artifact(PORTABLE_ARTIFACT)

    assert inspection.kind == "pyamplicol-artifact-inspection"
    assert inspection.integrity == "verified"
    assert inspection.model_name == "built-in-sm"
    assert inspection.target == "portable-64le"
    assert inspection.default_process_id == "d_dbar_to_z"
    assert inspection.payload_count > 0
    assert inspection.payload_size_bytes > 0

    assert len(inspection.processes) == 1
    process = inspection.processes[0]
    assert process.id == "d_dbar_to_z"
    assert process.expression == "d d~ > z"
    assert process.default
    assert process.physical_helicities == 12
    assert process.computed_helicities == 6
    assert process.physical_color_components == 1
    assert process.computed_color_components == 1
    assert process.helicity_coverage == "complete"
    assert process.color_coverage == "complete"
    assert process.execution_mode == "compiled"
    assert process.prepared_backend is None
    assert process.invocation_count is None
    assert process.native_profile_phases == (
        "source-fill",
        "momentum-setup",
        "stage-input-pack",
        "stage-evaluator-call",
        "output-assign",
        "amplitude-input-pack",
        "amplitude-evaluator-call",
        "reduction",
    )


def test_contracted_color_is_definitionally_computed() -> None:
    assert _computed(
        {
            "kind": "contracted-color",
            "id": "color:contracted",
            "index": 0,
            "description": "fully contracted color",
        },
        "color component",
    )
