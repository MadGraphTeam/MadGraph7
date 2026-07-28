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
from pyamplicol.generation.recurrence_columnar import RecurrenceSemanticDigestV1


def _digest(character: str) -> str:
    return character * 64


def _recurrence_process() -> SimpleNamespace:
    return SimpleNamespace(
        expression="d d~ > z g",
        process_id="d_dbar_to_z_g",
        color_accuracy="lc",
        external_pdgs=(1, -1, 23, 21),
        recurrence_schedule_size_bytes=256,
        recurrence_schedule_sha256=_digest("a"),
        recurrence_schedule_member_count=1,
        recurrence_schedule_unpacked_size_bytes=192,
        recurrence_schedule_index_sha256=_digest("b"),
        builder_input_sha256=_digest("c"),
        prepared_kernel_pack_digest=_digest("d"),
        direct_template_catalog_digest=_digest("e"),
        inspection_summary={
            "execution_mode": "recurrence",
            "process_id": "d_dbar_to_z_g",
            "semantic_digest": _digest("c"),
            "schedule_digest": _digest("f"),
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


def _execution_manifest(process: SimpleNamespace | None = None) -> dict[str, object]:
    process = process or _recurrence_process()
    return artifact_writer._recurrence_execution_manifest(
        process,
        schedule_path=f"recurrence/schedules/{_digest('f')}/recurrence-runtime.pacbin",
        binding={
            "abi": "pyamplicol-recurrence-process-binding-v2",
            "process_id": process.process_id,
            "schedule_digest": _digest("f"),
            "process_semantic_digest": _digest("c"),
            "process_support_words": [1],
            "path": "recurrence-binding.bin",
            "size_bytes": 128,
            "sha256": _digest("9"),
        },
        color_contraction_record=None,
    )


def _recurrence_execution_summary_with_size(
    monkeypatch: pytest.MonkeyPatch,
    target_size: int,
) -> bytes:
    empty_summary_size = len(b'{"padding":""}\n')
    assert target_size >= empty_summary_size

    def padded_manifest(
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        return {"padding": "x" * (target_size - empty_summary_size)}

    monkeypatch.setattr(
        artifact_writer,
        "_recurrence_execution_manifest",
        padded_manifest,
    )
    return artifact_writer._bounded_recurrence_execution_summary(
        _recurrence_process(),
        schedule_path="recurrence-runtime.pacbin",
        binding={},
        color_contraction_record=None,
    )


def _write_prepared_pack(root: Path) -> None:
    pack_path = root / "model" / "eager-kernel-pack.json"
    pack_path.parent.mkdir(parents=True)
    pack_path.write_text(
        json.dumps(
            {
                "backend": "jit",
                "kernels": [{"kernel_id": 0}, {"kernel_id": 1}],
                "recurrence_direct_template": {
                    "templates": [
                        {"payload_binding": {"prepared_kernel_id": 0}},
                        {"payload_binding": {"prepared_kernel_id": 1}},
                    ]
                },
            }
        ),
        encoding="utf-8",
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
    manifest = _execution_manifest()

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
    container = plan["runtime_schedule"]
    assert isinstance(container, dict)
    assert container["plan_member_path"] == (
        "schedule/recurrence-direct-schedule-v2.bin"
    )
    encoded = json.dumps(manifest)
    assert "pyamplicol-recurrence-plan-v1" not in encoded
    assert "pyamplicol-recurrence-runtime-layout-v1" not in encoded
    assert "rusticol.recurrence-runtime.complex-f64.v1" not in encoded


def test_recurrence_execution_summary_accepts_observed_large_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_maximum_size = 4_449_912

    content = _recurrence_execution_summary_with_size(
        monkeypatch,
        observed_maximum_size,
    )

    assert len(content) == observed_maximum_size


def test_recurrence_execution_summary_rejects_manifest_above_16_mib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = artifact_writer._MAX_RECURRENCE_EXECUTION_SUMMARY_BYTES + 1

    with pytest.raises(
        ValueError,
        match=rf"smaller than 16 MiB; received {oversized} bytes",
    ):
        _recurrence_execution_summary_with_size(monkeypatch, oversized)


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
        color_accuracy="lc",
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
    assert reduction["plan_member_path"] == (
        "schedule/recurrence-direct-schedule-v2.bin"
    )


def test_recurrence_physics_expands_global_helicity_flip_aliases() -> None:
    exact_one = SimpleNamespace(
        real_numerator=1,
        real_denominator=1,
        imag_numerator=0,
        imag_denominator=1,
    )
    external_leg = SimpleNamespace(
        source_slot=0,
        public_label=1,
        source_states=(
            SimpleNamespace(state_index=0, public_helicity=-1),
            SimpleNamespace(state_index=1, public_helicity=1),
        ),
    )
    proof_digest = _digest("7")
    logical = SimpleNamespace(
        process_id="d_dbar_to_z_g",
        layout="topology-replay",
        semantic_digests=(
            RecurrenceSemanticDigestV1(
                "helicity-equivalence:global-flip-v1",
                proof_digest,
            ),
        ),
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
        color_accuracy="lc",
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

    assert physics["coverage"]["structural_zero_helicity_count"] == 0
    negative, positive = physics["helicities"]
    assert negative["computed"] is True
    assert negative["representative_id"] == "h:-1"
    assert positive["computed"] is False
    assert positive["structural_zero"] is False
    assert positive["representative_id"] == "h:-1"
    reduction = physics["extensions"]["global_helicity_flip_reduction"]
    assert reduction == {
        "kind": "pyamplicol-global-helicity-flip-reduction-v1",
        "proof_role": "helicity-equivalence:global-flip-v1",
        "proof_digest": proof_digest,
        "physical_nonzero_helicity_count": 2,
        "representative_helicity_count": 1,
        "aliases": [
            {
                "physical_id": "h:+1",
                "representative_id": "h:-1",
            }
        ],
    }


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
    _write_prepared_pack(tmp_path)
    execution = _execution_manifest()
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


def test_direct_recurrence_inspection_exposes_contracted_color_summary(
    tmp_path: Path,
) -> None:
    _write_prepared_pack(tmp_path)
    execution = _execution_manifest()
    execution["runtime_metadata"]["color_contraction"] = {
        "abi": "pyamplicol-recurrence-color-contraction-v3",
        "color_accuracy": "full",
        "storage": "repeated",
        "includes_color_factor": True,
        "group_count": 384,
        "sector_count": 36,
        "active_sector_count": 6,
        "component_count": 64,
        "destination_count": 384,
        "entry_count": 21,
        "logical_entry_count": 1344,
        "semantic_digest": _digest("7"),
        "factorization": {
            "kind": "klein-four-walsh",
            "rank": 2,
            "coset_count": 1,
        },
        "path": "recurrence-color.bin",
        "size_bytes": 4096,
        "sha256": _digest("8"),
    }

    result = inspection._recurrence_execution_inspection(
        SimpleNamespace(root=tmp_path),
        execution,
    )

    assert result.recurrence_color_accuracy == "full"
    assert result.recurrence_color_storage == "repeated"
    assert result.recurrence_color_sector_count == 36
    assert result.recurrence_color_active_sector_count == 6
    assert result.recurrence_color_component_count == 64
    assert result.recurrence_color_group_count == 384
    assert result.recurrence_color_destination_count == 384
    assert result.recurrence_color_entry_count == 21
    assert result.recurrence_color_logical_entry_count == 1344
    assert result.recurrence_color_factorization_kind == "klein-four-walsh"
    assert result.recurrence_color_factorization_rank == 2
    assert result.recurrence_color_factorization_coset_count == 1


def test_inspection_requires_v2_metadata_without_packet_fallback(
    tmp_path: Path,
) -> None:
    _write_prepared_pack(tmp_path)
    execution = _execution_manifest()
    execution["recurrence_plan_abi"] = "pyamplicol-recurrence-plan-v1"

    with pytest.raises(ArtifactError, match="regenerate the recurrence artifact"):
        inspection._recurrence_execution_inspection(
            SimpleNamespace(root=tmp_path),
            execution,
        )


def test_v2_inspection_tolerates_pending_direct_counter_wiring(
    tmp_path: Path,
) -> None:
    _write_prepared_pack(tmp_path)
    process = _recurrence_process()
    process.inspection_summary.pop("direct_arena")
    execution = _execution_manifest(process)

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
