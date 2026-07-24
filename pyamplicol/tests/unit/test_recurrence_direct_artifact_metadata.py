# SPDX-License-Identifier: 0BSD
"""Python metadata contracts for direct-arena recurrence artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol._internal import versions
from pyamplicol.api.errors import ArtifactError
from pyamplicol.artifacts import inspection
from pyamplicol.generation import artifact_writer, recurrence_physics


def _digest(character: str) -> str:
    return character * 64


def _recurrence_process() -> SimpleNamespace:
    return SimpleNamespace(
        expression="d d~ > z g",
        process_id="d_dbar_to_z_g",
        color_accuracy="lc",
        external_pdgs=(1, -1, 23, 21),
        recurrence_runtime_size_bytes=256,
        recurrence_runtime_sha256=_digest("a"),
        recurrence_runtime_member_count=1,
        recurrence_runtime_unpacked_size_bytes=192,
        recurrence_runtime_index_sha256=_digest("b"),
        builder_input_sha256=_digest("c"),
        prepared_kernel_pack_digest=_digest("d"),
        direct_template_catalog_digest=_digest("e"),
        inspection_summary={
            "execution_mode": "recurrence",
            "prepared_kernel_count": 2,
            "schedule": {
                "source_row_count": 4,
                "contribution_count": 34,
                "finalization_count": 22,
                "closure_term_count": 12,
            },
            "direct_arena": {
                "semantic_component_count": 48,
                "current_arena_components": 32,
                "arena_component_reuse_count": 16,
                "momentum_form_count": 7,
                "row_group_count": 11,
                "packed_input_bytes": 0,
                "packed_output_bytes": 0,
                "scatter_bytes": 0,
            },
        },
        runtime_metadata={},
        point_tile_size=1024,
        workspace_mib=256,
        recurrence_summary={"lc_flow_layout": "topology-replay"},
    )


def test_direct_recurrence_versions_replace_packet_abi() -> None:
    assert (
        versions.RECURRENCE_BUILDER_INPUT_ABI
        == "pyamplicol-recurrence-builder-input-v2"
    )
    assert versions.RECURRENCE_PLAN_ABI == "pyamplicol-recurrence-plan-v2"
    assert (
        versions.RECURRENCE_RUNTIME_LAYOUT_ABI
        == "pyamplicol-recurrence-runtime-layout-v2"
    )
    assert (
        versions.RECURRENCE_DIRECT_TEMPLATE_ABI
        == "pyamplicol-recurrence-direct-template-v1"
    )
    assert (
        versions.RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY
        == "rusticol.recurrence-direct-arena.complex-f64.v1"
    )
    assert not hasattr(versions, "RECURRENCE_RUNTIME_CAPABILITY")


def test_recurrence_execution_manifest_publishes_only_direct_arena_contract() -> None:
    manifest = artifact_writer._recurrence_execution_manifest(_recurrence_process())

    assert manifest["builder_input_abi"] == versions.RECURRENCE_BUILDER_INPUT_ABI
    assert manifest["recurrence_plan_abi"] == versions.RECURRENCE_PLAN_ABI
    assert manifest["runtime_layout_abi"] == versions.RECURRENCE_RUNTIME_LAYOUT_ABI
    assert manifest["direct_template_abi"] == versions.RECURRENCE_DIRECT_TEMPLATE_ABI
    assert manifest["direct_backend_abi"] == versions.RECURRENCE_DIRECT_BACKEND_ABI
    assert manifest["prepared_kernel_pack_digest"] == _digest("d")
    assert manifest["direct_template_catalog_digest"] == _digest("e")
    assert manifest["required_runtime_capabilities"] == [
        versions.RECURRENCE_COLOR_RUNTIME_CAPABILITY,
        versions.RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
    ]
    plan = manifest["plan"]
    assert isinstance(plan, dict)
    assert plan["builder_input_abi"] == versions.RECURRENCE_BUILDER_INPUT_ABI
    assert plan["runtime_layout_abi"] == versions.RECURRENCE_RUNTIME_LAYOUT_ABI
    assert plan["prepared_kernel_pack_digest"] == _digest("d")
    assert plan["direct_template_catalog_digest"] == _digest("e")
    container = plan["runtime_container"]
    assert isinstance(container, dict)
    assert container["plan_member_path"] == "plan/recurrence-direct-plan-v2.bin"
    encoded = json.dumps(manifest)
    assert "pyamplicol-recurrence-plan-v1" not in encoded
    assert "pyamplicol-recurrence-runtime-layout-v1" not in encoded
    assert "rusticol.recurrence-runtime.complex-f64.v1" not in encoded


def test_recurrence_physics_identifies_direct_plan_and_runtime_layout() -> None:
    exact_one = SimpleNamespace(
        real_numerator=1,
        real_denominator=1,
        imag_numerator=0,
        imag_denominator=1,
    )
    source_state = SimpleNamespace(state_index=0, public_helicity=-1)
    external_leg = SimpleNamespace(
        source_slot=0,
        public_label=1,
        source_states=(source_state,),
    )
    logical = SimpleNamespace(
        process_id="d_dbar_to_z_g",
        layout="topology-replay",
        selected_source_coverage=None,
        selected_public_flow_ids=None,
        external_legs=(external_leg,),
        public_flows=(
            SimpleNamespace(
                flow_id=0,
                public_id="flow:1",
                word_source_slots=(0,),
                reduction_weight=exact_one,
            ),
        ),
    )
    process = SimpleNamespace(
        key="d_dbar_to_z_g",
        process="d d~ > z g",
        legs=(
            SimpleNamespace(
                label=1,
                particle="d",
                pdg=1,
                is_initial=True,
            ),
        ),
    )
    physics = recurrence_physics.build_recurrence_physics(
        process,
        logical,
        SimpleNamespace(parameters=()),
        process_id="d_dbar_to_z_g",
        resolved_helicities=((-1,),),
        normalization={},
    )

    extensions = physics["extensions"]
    assert isinstance(extensions, dict)
    selectors = extensions["runtime_selectors"]
    assert isinstance(selectors, dict)
    assert selectors["provenance"] == versions.RECURRENCE_PLAN_ABI
    recurrence = extensions["recurrence"]
    assert isinstance(recurrence, dict)
    assert recurrence == {
        "builder_input_abi": versions.RECURRENCE_BUILDER_INPUT_ABI,
        "plan_abi": versions.RECURRENCE_PLAN_ABI,
        "runtime_layout_abi": versions.RECURRENCE_RUNTIME_LAYOUT_ABI,
        "direct_template_abi": versions.RECURRENCE_DIRECT_TEMPLATE_ABI,
        "lc_flow_layout": "topology-replay",
    }
    reduction = extensions["recurrence_runtime_reduction"]
    assert isinstance(reduction, dict)
    assert reduction["kind"] == "pyamplicol-recurrence-native-reduction-v2"
    assert reduction["plan_member_path"] == "plan/recurrence-direct-plan-v2.bin"


def test_inspection_rejects_retired_recurrence_before_strict_manifest_load(
    tmp_path: Path,
) -> None:
    (tmp_path / "artifact.json").write_text(
        json.dumps(
            {
                "runtime": {
                    "required_runtime_capabilities": [
                        "rusticol.recurrence-runtime.complex-f64.v1"
                    ]
                },
                "processes": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match=r"retired packet ABI.*regenerate"):
        inspection.inspect_artifact(tmp_path)


def test_direct_recurrence_inspection_exposes_arena_counters(tmp_path: Path) -> None:
    pack_path = tmp_path / "model" / "eager-kernel-pack.json"
    pack_path.parent.mkdir(parents=True)
    pack_path.write_text(
        json.dumps({"backend": "jit", "kernels": [{}, {}]}),
        encoding="utf-8",
    )
    execution = artifact_writer._recurrence_execution_manifest(_recurrence_process())
    result = inspection._recurrence_execution_inspection(
        SimpleNamespace(root=tmp_path),
        execution,
    )

    assert result.invocation_count == 34
    assert result.attachment_count == 0
    assert result.arena_semantic_component_count == 48
    assert result.arena_component_count == 32
    assert result.arena_component_reuse_count == 16
    assert result.momentum_form_count == 7
    assert result.direct_source_row_count == 4
    assert result.direct_contribution_row_count == 34
    assert result.direct_finalization_row_count == 22
    assert result.direct_closure_row_count == 12
    assert result.direct_row_group_count == 11
    assert result.packed_input_bytes == 0
    assert result.packed_output_bytes == 0
    assert result.scatter_bytes == 0


def test_inspection_requires_v2_metadata_without_packet_fallback(
    tmp_path: Path,
) -> None:
    pack_path = tmp_path / "model" / "eager-kernel-pack.json"
    pack_path.parent.mkdir(parents=True)
    pack_path.write_text(
        json.dumps({"backend": "jit", "kernels": [{}, {}]}),
        encoding="utf-8",
    )
    execution = artifact_writer._recurrence_execution_manifest(_recurrence_process())
    execution["recurrence_plan_abi"] = "pyamplicol-recurrence-plan-v1"

    with pytest.raises(ArtifactError, match="regenerate the recurrence artifact"):
        inspection._recurrence_execution_inspection(
            SimpleNamespace(root=tmp_path),
            execution,
        )


def test_v2_inspection_tolerates_pending_direct_counter_wiring(
    tmp_path: Path,
) -> None:
    pack_path = tmp_path / "model" / "eager-kernel-pack.json"
    pack_path.parent.mkdir(parents=True)
    pack_path.write_text(
        json.dumps({"backend": "jit", "kernels": [{}, {}]}),
        encoding="utf-8",
    )
    process = _recurrence_process()
    process.inspection_summary.pop("direct_arena")
    execution = artifact_writer._recurrence_execution_manifest(process)

    result = inspection._recurrence_execution_inspection(
        SimpleNamespace(root=tmp_path),
        execution,
    )

    assert result.arena_component_count is None
    assert result.momentum_form_count is None
    assert result.direct_row_group_count is None
    assert result.packed_input_bytes == 0
    assert result.packed_output_bytes == 0
    assert result.scatter_bytes == 0
