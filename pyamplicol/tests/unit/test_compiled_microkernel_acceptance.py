# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy
import json
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest

from tools.developer import compiled_microkernel_acceptance as acceptance


def test_dependency_identity_is_read_from_the_release_lock() -> None:
    root = Path(__file__).resolve().parents[2]
    with (root / "dependencies" / "release-lock.toml").open("rb") as stream:
        symjit = tomllib.load(stream)["symjit"]

    assert symjit["repository"] == acceptance.DEPENDENCY_REPOSITORY
    assert symjit["version"] == acceptance.DEPENDENCY_VERSION
    assert symjit["revision"] == acceptance.DEPENDENCY_REVISION


def _digest(character: str) -> str:
    return character * 64


def _certificate(kind: str, rows: list[dict[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {"kind": kind, "rows": rows}
    return {**payload, "sha256": acceptance.canonical_sha256(payload)}


def _diagnostics(
    *,
    islands: int = 1,
    kernels: int = 1,
    invocations: int = 2,
    attachments: int = 2,
    table_code: int = 100,
    residual_code: int = 50,
    semantic_bytes: int = 32,
    arena_bytes: int = 64,
    allocations: int = 0,
) -> dict[str, int]:
    return {
        "island_count": islands,
        "kernel_count": kernels,
        "invocation_count": invocations,
        "attachment_count": attachments,
        "table_machine_code_bytes": table_code,
        "residual_machine_code_bytes": residual_code,
        "semantic_row_bytes": semantic_bytes,
        "arena_bytes": arena_bytes,
        "warmed_allocation_count": allocations,
    }


def _stage_plan() -> dict[str, object]:
    return {
        "kind": acceptance.STAGE_PLAN_KIND,
        "schema_version": acceptance.STAGE_PLAN_SCHEMA_VERSION,
        "stage_id": "stage-0",
        "direct_application_abi": acceptance.DIRECT_APPLICATION_ABI,
        "direct_table_binding_abi": acceptance.DIRECT_TABLE_BINDING_ABI,
        "direct_table_descriptor_abi": acceptance.DIRECT_TABLE_DESCRIPTOR_ABI,
        "output_current_ids": [10, 11],
        "table_current_ids": [10],
        "residual_current_ids": [11],
        "residual_leaves": [
            {
                "leaf_id": 0,
                "current_ids": [11],
                "source_sha256": _digest("1"),
                "source_abi": acceptance.DIRECT_APPLICATION_ABI,
                "source_bytes": 120,
                "machine_code_bytes": 50,
                "optimization_level": 3,
            }
        ],
        "kernels": [
            {
                "kernel_id": 0,
                "kernel_family": acceptance.VECTOR_WEYL_KERNEL_FAMILY,
                "motif_sha256": _digest("2"),
                "source_sha256": _digest("3"),
                "descriptor_sha256": _digest("4"),
                "source_abi": acceptance.DIRECT_APPLICATION_ABI,
                "binding_abi": acceptance.DIRECT_TABLE_BINDING_ABI,
                "descriptor_abi": acceptance.DIRECT_TABLE_DESCRIPTOR_ABI,
                "canonical_input_order": ["left-current", "momentum"],
                "input_permutation": [0, 1],
                "result_signature": "vector-weyl:2",
                "mutable_parameter_sha256": _digest("5"),
                "coupling_provenance_sha256": _digest("6"),
                "selector_domain_sha256": _digest("7"),
                "finalizer_sha256": _digest("8"),
                "input_complex_count": 2,
                "output_complex_count": 2,
                "source_bytes": 4096,
                "machine_code_bytes": 100,
                "optimization_level": 3,
            }
        ],
        "islands": [
            {
                "island_id": 0,
                "kernel_id": 0,
                "current_ids": [10],
                "selector_partition_sha256": _digest("9"),
                "invocations": [
                    {
                        "invocation_id": 0,
                        "evaluation_group_id": 7,
                        "attachment_start": 0,
                        "attachment_count": 1,
                    },
                    {
                        "invocation_id": 1,
                        "evaluation_group_id": 8,
                        "attachment_start": 1,
                        "attachment_count": 1,
                    },
                ],
                "attachments": [
                    {
                        "attachment_id": 0,
                        "invocation_id": 0,
                        "current_id": 10,
                        "evaluation_group_id": 7,
                        "operation": "overwrite",
                        "destination_complex_count": 2,
                        "factor_id": 0,
                    },
                    {
                        "attachment_id": 1,
                        "invocation_id": 1,
                        "current_id": 10,
                        "evaluation_group_id": 8,
                        "operation": "accumulate",
                        "destination_complex_count": 2,
                        "factor_id": 0,
                    },
                ],
                "factor_catalog": [{"factor_id": 0, "factor_sha256": _digest("a")}],
                "plane_bindings": [
                    {
                        "plane_id": index,
                        "role": role,
                        "canonical_index": index,
                        "permutation_index": index,
                    }
                    for index, role in enumerate(
                        ("current", "momentum", "parameter", "factor")
                    )
                ],
                "dependency_certificate": _certificate(
                    "complete-current-independence-v1",
                    [{"current_id": 10, "predecessor_current_ids": [1, 2]}],
                ),
                "order_certificate": _certificate(
                    "evaluation-group-order-v1",
                    [{"current_id": 10, "evaluation_group_ids": [7, 8]}],
                ),
                "semantic_row_bytes": 32,
                "arena_bytes": 64,
            }
        ],
        "finalizers": [
            {
                "current_id": 10,
                "island_id": 0,
                "identity_sha256": _digest("8"),
            }
        ],
        "diagnostics": _diagnostics(),
    }


def _residual_only_stage() -> dict[str, object]:
    stage = _stage_plan()
    stage["table_current_ids"] = []
    stage["residual_current_ids"] = [10, 11]
    stage["residual_leaves"] = [
        {
            "leaf_id": 0,
            "current_ids": [10, 11],
            "source_sha256": _digest("1"),
            "source_abi": acceptance.DIRECT_APPLICATION_ABI,
            "source_bytes": 240,
            "machine_code_bytes": 75,
            "optimization_level": 3,
        }
    ]
    stage["kernels"] = []
    stage["islands"] = []
    stage["finalizers"] = []
    stage["diagnostics"] = _diagnostics(
        islands=0,
        kernels=0,
        invocations=0,
        attachments=0,
        table_code=0,
        residual_code=75,
        semantic_bytes=0,
        arena_bytes=0,
    )
    return stage


def _table_only_stage() -> dict[str, object]:
    stage = _stage_plan()
    stage["output_current_ids"] = [10]
    stage["residual_current_ids"] = []
    stage["residual_leaves"] = []
    diagnostics = stage["diagnostics"]
    assert isinstance(diagnostics, dict)
    diagnostics["residual_machine_code_bytes"] = 0
    return stage


def _authorized_extended_stage_plan() -> dict[str, object]:
    vector_current_ids = list(range(26))
    three_vector_current_ids = list(range(26, 31))
    residual_current_ids = list(range(31, 55))
    contribution_counts = [width for width in range(1, 7) for _ in range(4)]
    contribution_counts.extend((7, 7))
    assert len(contribution_counts) == 26
    assert sum(contribution_counts) == 98

    vector_invocations: list[dict[str, object]] = []
    vector_attachments: list[dict[str, object]] = []
    vector_order_rows: list[dict[str, object]] = []
    group_id = 0
    for current_id, contribution_count in zip(
        vector_current_ids,
        contribution_counts,
        strict=True,
    ):
        current_groups: list[int] = []
        for contribution_index in range(contribution_count):
            invocation_id = len(vector_invocations)
            vector_invocations.append(
                {
                    "invocation_id": invocation_id,
                    "evaluation_group_id": group_id,
                    "attachment_start": invocation_id,
                    "attachment_count": 1,
                }
            )
            vector_attachments.append(
                {
                    "attachment_id": invocation_id,
                    "invocation_id": invocation_id,
                    "current_id": current_id,
                    "evaluation_group_id": group_id,
                    "operation": (
                        "overwrite" if contribution_index == 0 else "accumulate"
                    ),
                    "destination_complex_count": 2,
                    "factor_id": 0,
                }
            )
            current_groups.append(group_id)
            group_id += 1
        vector_order_rows.append(
            {
                "current_id": current_id,
                "evaluation_group_ids": current_groups,
            }
        )
    assert group_id == 98

    three_vector_invocations = [
        {
            "invocation_id": index,
            "evaluation_group_id": 98 + index,
            "attachment_start": index,
            "attachment_count": 1,
        }
        for index in range(5)
    ]
    three_vector_attachments = [
        {
            "attachment_id": index,
            "invocation_id": index,
            "current_id": current_id,
            "evaluation_group_id": 98 + index,
            "operation": "overwrite",
            "destination_complex_count": 4,
            "factor_id": 0,
        }
        for index, current_id in enumerate(three_vector_current_ids)
    ]

    def bindings() -> list[dict[str, object]]:
        return [
            {
                "plane_id": index,
                "role": role,
                "canonical_index": index,
                "permutation_index": index,
            }
            for index, role in enumerate(("current", "momentum", "parameter", "factor"))
        ]

    vector_finalizer = _digest("8")
    three_vector_finalizer = _digest("d")
    return {
        "kind": acceptance.STAGE_PLAN_KIND,
        "schema_version": acceptance.STAGE_PLAN_SCHEMA_VERSION,
        "stage_id": "stage-0",
        "direct_application_abi": acceptance.DIRECT_APPLICATION_ABI,
        "direct_table_binding_abi": acceptance.DIRECT_TABLE_BINDING_ABI,
        "direct_table_descriptor_abi": acceptance.DIRECT_TABLE_DESCRIPTOR_ABI,
        "output_current_ids": list(range(55)),
        "table_current_ids": vector_current_ids + three_vector_current_ids,
        "residual_current_ids": residual_current_ids,
        "residual_leaves": [
            {
                "leaf_id": 0,
                "current_ids": residual_current_ids,
                "source_sha256": _digest("1"),
                "source_abi": acceptance.DIRECT_APPLICATION_ABI,
                "source_bytes": 120,
                "machine_code_bytes": 50,
                "optimization_level": 3,
            }
        ],
        "kernels": [
            {
                "kernel_id": 0,
                "kernel_family": acceptance.VECTOR_WEYL_KERNEL_FAMILY,
                "motif_sha256": _digest("2"),
                "source_sha256": _digest("3"),
                "descriptor_sha256": _digest("4"),
                "source_abi": acceptance.DIRECT_APPLICATION_ABI,
                "binding_abi": acceptance.DIRECT_TABLE_BINDING_ABI,
                "descriptor_abi": acceptance.DIRECT_TABLE_DESCRIPTOR_ABI,
                "canonical_input_order": [f"input-{index}" for index in range(6)],
                "input_permutation": list(range(6)),
                "result_signature": "vector-weyl:2",
                "mutable_parameter_sha256": _digest("5"),
                "coupling_provenance_sha256": _digest("6"),
                "selector_domain_sha256": _digest("7"),
                "finalizer_sha256": vector_finalizer,
                "input_complex_count": 6,
                "output_complex_count": 2,
                "source_bytes": 4096,
                "machine_code_bytes": 100,
                "optimization_level": 3,
            },
            {
                "kernel_id": 1,
                "kernel_family": (acceptance.THREE_VECTOR_SINGLETON_KERNEL_FAMILY),
                "motif_sha256": _digest("9"),
                "source_sha256": _digest("a"),
                "descriptor_sha256": _digest("b"),
                "source_abi": acceptance.DIRECT_APPLICATION_ABI,
                "binding_abi": acceptance.DIRECT_TABLE_BINDING_ABI,
                "descriptor_abi": acceptance.DIRECT_TABLE_DESCRIPTOR_ABI,
                "canonical_input_order": [f"input-{index}" for index in range(16)],
                "input_permutation": list(range(16)),
                "result_signature": "three-vector:4",
                "mutable_parameter_sha256": _digest("c"),
                "coupling_provenance_sha256": _digest("d"),
                "selector_domain_sha256": _digest("e"),
                "finalizer_sha256": three_vector_finalizer,
                "input_complex_count": 16,
                "output_complex_count": 4,
                "source_bytes": 8192,
                "machine_code_bytes": 120,
                "optimization_level": 3,
            },
        ],
        "islands": [
            {
                "island_id": 0,
                "kernel_id": 0,
                "current_ids": vector_current_ids,
                "selector_partition_sha256": _digest("9"),
                "invocations": vector_invocations,
                "attachments": vector_attachments,
                "factor_catalog": [{"factor_id": 0, "factor_sha256": _digest("a")}],
                "plane_bindings": bindings(),
                "dependency_certificate": _certificate(
                    "complete-current-independence-v1",
                    [
                        {
                            "current_id": current_id,
                            "predecessor_current_ids": [100 + current_id],
                        }
                        for current_id in vector_current_ids
                    ],
                ),
                "order_certificate": _certificate(
                    "evaluation-group-order-v1",
                    vector_order_rows,
                ),
                "semantic_row_bytes": 3136,
                "arena_bytes": 1664,
            },
            {
                "island_id": 1,
                "kernel_id": 1,
                "current_ids": three_vector_current_ids,
                "selector_partition_sha256": _digest("e"),
                "invocations": three_vector_invocations,
                "attachments": three_vector_attachments,
                "factor_catalog": [{"factor_id": 0, "factor_sha256": _digest("f")}],
                "plane_bindings": bindings(),
                "dependency_certificate": _certificate(
                    "complete-current-independence-v1",
                    [
                        {
                            "current_id": current_id,
                            "predecessor_current_ids": [200 + current_id],
                        }
                        for current_id in three_vector_current_ids
                    ],
                ),
                "order_certificate": _certificate(
                    "evaluation-group-order-v1",
                    [
                        {
                            "current_id": current_id,
                            "evaluation_group_ids": [98 + index],
                        }
                        for index, current_id in enumerate(three_vector_current_ids)
                    ],
                ),
                "semantic_row_bytes": 160,
                "arena_bytes": 320,
            },
        ],
        "finalizers": [
            {
                "current_id": current_id,
                "island_id": 0 if current_id in vector_current_ids else 1,
                "identity_sha256": (
                    vector_finalizer
                    if current_id in vector_current_ids
                    else three_vector_finalizer
                ),
            }
            for current_id in vector_current_ids + three_vector_current_ids
        ],
        "diagnostics": _diagnostics(
            islands=2,
            kernels=2,
            invocations=103,
            attachments=103,
            table_code=220,
            residual_code=50,
            semantic_bytes=3296,
            arena_bytes=1984,
        ),
    }


def _active_prepared_kernel_occurrences() -> list[dict[str, object]]:
    prepared_kernel_ids = (
        [7] * 49 + [24] * 49 + [4] * 5 + [4] * 30 + [36] * 40 + [41] * 30
    )
    assert len(prepared_kernel_ids) == acceptance.TARGET_MATERIALIZED_INTERACTIONS
    return [
        {
            "occurrence_id": f"stage-0:{index}",
            "prepared_kernel_id": prepared_kernel_id,
        }
        for index, prepared_kernel_id in enumerate(prepared_kernel_ids)
    ]


def _complex_values(batch: int, value: float = 1.0) -> list[list[float]]:
    return [[value, 0.0] for _ in range(batch)]


def _resolved_values(batch: int, value: float = 1.0) -> list[list[list[float]]]:
    return [[[value, 0.0]] for _ in range(batch)]


def _correctness(batch: int) -> dict[str, object]:
    return {
        lane: {
            "evaluate": _complex_values(batch),
            "resolved_total": _complex_values(batch),
            "resolved_contributions": _resolved_values(batch),
        }
        for lane in ("baseline", "candidate")
    }


def _measurements(
    *,
    candidate_ratio: float,
    allocations: int = 0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair_index in range(acceptance.MIN_SAMPLE_PAIRS):
        order = (
            ("baseline", "candidate")
            if pair_index % 2 == 0
            else ("candidate", "baseline")
        )
        baseline = 1.0 + 0.001 * (pair_index % 3)
        for lane in order:
            rows.append(
                {
                    "pair_index": pair_index,
                    "sequence_index": len(rows),
                    "lane": lane,
                    "duration_seconds": 5.25,
                    "seconds_per_point": (
                        baseline if lane == "baseline" else baseline * candidate_ratio
                    ),
                    "warmed_allocation_count": (
                        allocations if lane == "candidate" else 0
                    ),
                }
            )
    return rows


def _artifact_metrics(
    *,
    generation_seconds: float,
    artifact_bytes: int,
    load_seconds: float,
    peak_rss_bytes: int,
    machine_code_bytes: int,
    digest_character: str,
    code_size_metric_available: bool = True,
) -> dict[str, object]:
    return {
        "artifact_sha256": _digest(digest_character),
        "build_identity_sha256": _digest(digest_character),
        "generation_seconds": generation_seconds,
        "artifact_bytes": artifact_bytes,
        "load_seconds": load_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "code_size_metric_available": code_size_metric_available,
        "code_size_metric_kind": (
            acceptance.EXACT_CODE_SIZE_METRIC if code_size_metric_available else None
        ),
        "selected_machine_code_bytes": (
            machine_code_bytes if code_size_metric_available else None
        ),
        "portable_source_applications": [
            {
                "source_application_sha256": _digest(digest_character),
                "source_application_bytes": 4096,
                "source_application_abi": acceptance.SOURCE_APPLICATION_ABI,
            }
        ],
    }


def _campaign() -> dict[str, object]:
    point_digest = _digest("b")
    stage = _authorized_extended_stage_plan()
    stage_diagnostics = copy.deepcopy(stage["diagnostics"])
    campaign: dict[str, object] = {
        "kind": acceptance.CAMPAIGN_KIND,
        "schema_version": acceptance.CAMPAIGN_SCHEMA_VERSION,
        "dependency": {
            "repository": acceptance.DEPENDENCY_REPOSITORY,
            "version": acceptance.DEPENDENCY_VERSION,
            "revision": acceptance.DEPENDENCY_REVISION,
            "local_patch_count": 0,
            "direct_application_abi": acceptance.DIRECT_APPLICATION_ABI,
            "direct_table_binding_abi": acceptance.DIRECT_TABLE_BINDING_ABI,
            "direct_table_descriptor_abi": acceptance.DIRECT_TABLE_DESCRIPTOR_ABI,
        },
        "workload": {
            "process": acceptance.TARGET_PROCESS,
            "color_accuracy": acceptance.TARGET_COLOR_ACCURACY,
            "lc_flow_layout": acceptance.TARGET_LAYOUT,
            "selected_flow": acceptance.TARGET_FLOW,
            "helicity_mode": acceptance.TARGET_HELICITY_MODE,
            "source_sha256": _digest("c"),
            "model_sha256": _digest("d"),
            "point_set_sha256": point_digest,
            "runtime_target_sha256": _digest("e"),
            "host_sha256": _digest("f"),
        },
        "baseline": _artifact_metrics(
            generation_seconds=100.0,
            artifact_bytes=1000,
            load_seconds=10.0,
            peak_rss_bytes=10_000,
            machine_code_bytes=1000,
            digest_character="1",
        ),
        "candidate": {
            "metrics": _artifact_metrics(
                generation_seconds=105.0,
                artifact_bytes=1050,
                load_seconds=10.5,
                peak_rss_bytes=10_500,
                machine_code_bytes=700,
                digest_character="2",
            ),
            "stage_plans": [stage],
            "diagnostics": stage_diagnostics,
            "census": {
                "denominator_contract": acceptance.CENSUS_DENOMINATOR_CONTRACT,
                "active_non_source_current_slots": (
                    acceptance.TARGET_ACTIVE_NON_SOURCE_CURRENT_SLOTS
                ),
                "materialized_interaction_count": (
                    acceptance.TARGET_MATERIALIZED_INTERACTIONS
                ),
                "two_component_current_slots": (
                    acceptance.TARGET_TWO_COMPONENT_CURRENT_SLOTS
                ),
                "four_component_three_vector_singleton_slots": (
                    acceptance.TARGET_THREE_VECTOR_SINGLETON_CURRENT_SLOTS
                ),
                "active_prepared_kernel_occurrences": (
                    _active_prepared_kernel_occurrences()
                ),
                "eligible_prepared_kernel_occurrence_ids": [
                    f"stage-0:{index}"
                    for index in range(
                        acceptance.TARGET_ELIGIBLE_PREPARED_KERNEL_OCCURRENCES
                    )
                ],
                "proof_dag_non_source_current_slots": (
                    acceptance.TARGET_PROOF_DAG_NON_SOURCE_CURRENT_SLOTS
                ),
                "proof_dag_interaction_count": (
                    acceptance.TARGET_PROOF_DAG_INTERACTIONS
                ),
                "projected_generated_text_bytes": 12_000,
                "projected_replaced_text_bytes": 100_000,
            },
        },
        "batches": {
            str(batch): {
                "point_set_sha256": point_digest,
                "correctness": _correctness(batch),
                "measurements": _measurements(
                    candidate_ratio=(
                        0.85
                        if batch in acceptance.PRIMARY_BATCHES
                        else 1.04
                        if batch == 1
                        else 0.95
                    )
                ),
            }
            for batch in acceptance.REQUIRED_BATCHES
        },
        "regression_cases": [
            {
                "name": name,
                "passed": True,
                "evidence_sha256": _digest("a"),
            }
            for name in sorted(acceptance.REQUIRED_REGRESSION_CASES)
        ],
        "non_target_performance": [
            {
                "name": name,
                "baseline_seconds_per_point": [1.0] * 7,
                "candidate_seconds_per_point": [1.01] * 7,
            }
            for name in sorted(acceptance.REQUIRED_NON_TARGET_PERFORMANCE)
        ],
    }
    return _seal(campaign)


def _seal(campaign: dict[str, object]) -> dict[str, object]:
    campaign.pop("content_sha256", None)
    campaign["content_sha256"] = acceptance.canonical_sha256(campaign)
    return campaign


def _candidate(campaign: dict[str, object]) -> dict[str, object]:
    value = campaign["candidate"]
    assert isinstance(value, dict)
    return value


def _batches(campaign: dict[str, object]) -> dict[str, object]:
    value = campaign["batches"]
    assert isinstance(value, dict)
    return value


def test_valid_campaign_recomputes_every_landing_gate() -> None:
    result = acceptance.audit_campaign(_campaign())

    assert result["passes"] is True
    assert set(result["batches"]) == {
        str(batch) for batch in acceptance.REQUIRED_BATCHES
    }
    gates = result["gates"]
    assert isinstance(gates, dict)
    assert gates["primary_gain:batch_128"]["passes"] is True
    assert gates["primary_gain:batch_1024"]["passes"] is True
    assert gates["selected_machine_code_reduction"]["passes"] is True
    assert gates["zero_warmed_allocations"]["passes"] is True


def test_residual_only_and_table_only_v2_stages_are_valid() -> None:
    residual = acceptance.audit_stage_plan_v2(_residual_only_stage())
    table = acceptance.audit_stage_plan_v2(_table_only_stage())

    assert residual["residual_only"] is True
    assert residual["table_only"] is False
    assert table["residual_only"] is False
    assert table["table_only"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 1),
        ("direct_application_abi", "symjit-direct-application-storage-v0"),
        ("direct_table_binding_abi", "symjit-direct-table-binding-v0"),
        ("direct_table_descriptor_abi", "symjit-direct-table-descriptor-v0"),
    ],
)
def test_v1_and_incompatible_abis_fail_closed(field: str, value: object) -> None:
    stage = _stage_plan()
    stage[field] = value

    with pytest.raises(acceptance.AcceptanceError, match="regenerate"):
        acceptance.audit_stage_plan_v2(stage)


def test_unknown_fields_and_split_destinations_fail_closed() -> None:
    stage = _stage_plan()
    stage["silent_fallback"] = True
    with pytest.raises(acceptance.AcceptanceError, match="field mismatch"):
        acceptance.audit_stage_plan_v2(stage)

    stage = _stage_plan()
    stage["residual_current_ids"] = [10, 11]
    with pytest.raises(acceptance.AcceptanceError, match="may not be split"):
        acceptance.audit_stage_plan_v2(stage)


def test_order_certificate_is_authenticated_and_matches_attachments() -> None:
    stage = _stage_plan()
    island = stage["islands"][0]  # type: ignore[index]
    assert isinstance(island, dict)
    island["order_certificate"] = _certificate(
        "evaluation-group-order-v1",
        [{"current_id": 10, "evaluation_group_ids": [8, 7]}],
    )

    with pytest.raises(acceptance.AcceptanceError, match="attachment order differs"):
        acceptance.audit_stage_plan_v2(stage)

    stage = _stage_plan()
    island = stage["islands"][0]  # type: ignore[index]
    assert isinstance(island, dict)
    certificate = island["order_certificate"]
    assert isinstance(certificate, dict)
    certificate["sha256"] = _digest("0")
    with pytest.raises(acceptance.AcceptanceError, match="SHA-256 mismatch"):
        acceptance.audit_stage_plan_v2(stage)


def test_noncontiguous_ranges_and_partial_destination_writes_fail_closed() -> None:
    stage = _stage_plan()
    island = stage["islands"][0]  # type: ignore[index]
    assert isinstance(island, dict)
    invocation = island["invocations"][1]  # type: ignore[index]
    assert isinstance(invocation, dict)
    invocation["attachment_start"] = 0
    with pytest.raises(acceptance.AcceptanceError, match="outside its invocation"):
        acceptance.audit_stage_plan_v2(stage)

    stage = _stage_plan()
    island = stage["islands"][0]  # type: ignore[index]
    assert isinstance(island, dict)
    attachment = island["attachments"][0]  # type: ignore[index]
    assert isinstance(attachment, dict)
    attachment["destination_complex_count"] = 1
    with pytest.raises(acceptance.AcceptanceError, match="complete"):
        acceptance.audit_stage_plan_v2(stage)


def test_dependency_certificate_rejects_grouped_dependent_currents() -> None:
    stage = _stage_plan()
    stage["output_current_ids"] = [10, 11, 12]
    stage["table_current_ids"] = [10, 12]
    island = stage["islands"][0]  # type: ignore[index]
    assert isinstance(island, dict)
    island["current_ids"] = [10, 12]
    island["invocations"] = [
        {
            "invocation_id": 0,
            "evaluation_group_id": 7,
            "attachment_start": 0,
            "attachment_count": 2,
        },
        {
            "invocation_id": 1,
            "evaluation_group_id": 8,
            "attachment_start": 2,
            "attachment_count": 2,
        },
    ]
    island["attachments"] = [
        {
            "attachment_id": index,
            "invocation_id": index // 2,
            "current_id": 10 if index % 2 == 0 else 12,
            "evaluation_group_id": 7 if index < 2 else 8,
            "operation": "overwrite" if index < 2 else "accumulate",
            "destination_complex_count": 2,
            "factor_id": 0,
        }
        for index in range(4)
    ]
    island["dependency_certificate"] = _certificate(
        "complete-current-independence-v1",
        [
            {"current_id": 10, "predecessor_current_ids": [12]},
            {"current_id": 12, "predecessor_current_ids": []},
        ],
    )
    island["order_certificate"] = _certificate(
        "evaluation-group-order-v1",
        [
            {"current_id": 10, "evaluation_group_ids": [7, 8]},
            {"current_id": 12, "evaluation_group_ids": [7, 8]},
        ],
    )
    stage["finalizers"] = [
        {
            "current_id": current_id,
            "island_id": 0,
            "identity_sha256": _digest("8"),
        }
        for current_id in (10, 12)
    ]
    diagnostics = stage["diagnostics"]
    assert isinstance(diagnostics, dict)
    diagnostics["attachment_count"] = 4

    with pytest.raises(acceptance.AcceptanceError, match="dependent destinations"):
        acceptance.audit_stage_plan_v2(stage)


def test_declared_diagnostics_and_hard_bounds_are_recomputed() -> None:
    stage = _stage_plan()
    diagnostics = stage["diagnostics"]
    assert isinstance(diagnostics, dict)
    diagnostics["attachment_count"] = 3
    with pytest.raises(acceptance.AcceptanceError, match="recomputed"):
        acceptance.audit_stage_plan_v2(stage)

    stage = _stage_plan()
    kernel = stage["kernels"][0]  # type: ignore[index]
    assert isinstance(kernel, dict)
    kernel["source_bytes"] = acceptance.MAX_KERNEL_SOURCE_BYTES + 1
    with pytest.raises(acceptance.AcceptanceError, match="64 KiB"):
        acceptance.audit_stage_plan_v2(stage)


def test_census_rejects_proof_dag_substitution_and_incomplete_occurrences() -> None:
    campaign = _campaign()
    census = _candidate(campaign)["census"]
    assert isinstance(census, dict)
    census["active_non_source_current_slots"] = (
        acceptance.TARGET_PROOF_DAG_NON_SOURCE_CURRENT_SLOTS
    )
    _seal(campaign)
    with pytest.raises(acceptance.AcceptanceError, match="frozen target census"):
        acceptance.audit_campaign(campaign)

    campaign = _campaign()
    census = _candidate(campaign)["census"]
    assert isinstance(census, dict)
    active = census["active_prepared_kernel_occurrences"]
    assert isinstance(active, list)
    active.pop()
    _seal(campaign)
    with pytest.raises(acceptance.AcceptanceError, match="enumerate all 203"):
        acceptance.audit_campaign(campaign)

    campaign = _campaign()
    census = _candidate(campaign)["census"]
    assert isinstance(census, dict)
    eligible = census["eligible_prepared_kernel_occurrence_ids"]
    assert isinstance(eligible, list)
    eligible.pop()
    _seal(campaign)
    with pytest.raises(acceptance.AcceptanceError, match="exactly match"):
        acceptance.audit_campaign(campaign)


def test_authorized_kernel_families_enforce_family_specific_output_caps() -> None:
    summary = acceptance.audit_stage_plan_v2(_authorized_extended_stage_plan())
    counts = summary["table_destination_family_counts"]
    assert counts == {
        acceptance.VECTOR_WEYL_KERNEL_FAMILY: 26,
        acceptance.THREE_VECTOR_SINGLETON_KERNEL_FAMILY: 5,
    }

    stage = _stage_plan()
    kernel = stage["kernels"][0]  # type: ignore[index]
    assert isinstance(kernel, dict)
    kernel["output_complex_count"] = 4
    kernel["result_signature"] = "three-vector:4"
    with pytest.raises(acceptance.AcceptanceError, match="vector-weyl"):
        acceptance.audit_stage_plan_v2(stage)

    stage = _authorized_extended_stage_plan()
    kernel = stage["kernels"][1]  # type: ignore[index]
    assert isinstance(kernel, dict)
    kernel["input_complex_count"] = 15
    kernel["canonical_input_order"] = [f"input-{index}" for index in range(15)]
    kernel["input_permutation"] = list(range(15))
    with pytest.raises(acceptance.AcceptanceError, match="exactly 16 inputs"):
        acceptance.audit_stage_plan_v2(stage)


def test_three_vector_family_rejects_non_singleton_currents() -> None:
    stage = _authorized_extended_stage_plan()
    island = stage["islands"][1]  # type: ignore[index]
    assert isinstance(island, dict)
    attachments = island["attachments"]
    invocations = island["invocations"]
    assert isinstance(attachments, list)
    assert isinstance(invocations, list)
    invocations.append(
        {
            "invocation_id": 5,
            "evaluation_group_id": 103,
            "attachment_start": 5,
            "attachment_count": 1,
        }
    )
    attachments.append(
        {
            "attachment_id": 5,
            "invocation_id": 5,
            "current_id": 26,
            "evaluation_group_id": 103,
            "operation": "accumulate",
            "destination_complex_count": 4,
            "factor_id": 0,
        }
    )
    island["order_certificate"] = _certificate(
        "evaluation-group-order-v1",
        [
            {
                "current_id": current_id,
                "evaluation_group_ids": (
                    [98, 103] if current_id == 26 else [98 + index]
                ),
            }
            for index, current_id in enumerate(range(26, 31))
        ],
    )
    with pytest.raises(acceptance.AcceptanceError, match="singleton"):
        acceptance.audit_stage_plan_v2(stage)


def test_odd_tail_batch_or_numerical_mismatch_is_not_accepted() -> None:
    campaign = _campaign()
    _batches(campaign).pop("129")
    _seal(campaign)
    with pytest.raises(acceptance.AcceptanceError, match="exactly batches"):
        acceptance.audit_campaign(campaign)

    campaign = _campaign()
    batch = _batches(campaign)["127"]
    assert isinstance(batch, dict)
    correctness = batch["correctness"]
    assert isinstance(correctness, dict)
    candidate = correctness["candidate"]
    assert isinstance(candidate, dict)
    for field in ("evaluate", "resolved_total", "resolved_contributions"):
        values = candidate[field]
        assert isinstance(values, list)
        if field == "resolved_contributions":
            values[0][0][0] += 1.0e-6
        else:
            values[0][0] += 1.0e-6
    _seal(campaign)
    with pytest.raises(acceptance.AcceptanceError, match="candidate_vs_baseline"):
        acceptance.audit_campaign(campaign)


def test_resolved_contributions_must_sum_to_the_declared_total() -> None:
    campaign = _campaign()
    batch = _batches(campaign)["1"]
    assert isinstance(batch, dict)
    correctness = batch["correctness"]
    assert isinstance(correctness, dict)
    candidate = correctness["candidate"]
    assert isinstance(candidate, dict)
    resolved = candidate["resolved_contributions"]
    assert isinstance(resolved, list)
    resolved[0][0][0] += 1.0e-6
    _seal(campaign)

    with pytest.raises(
        acceptance.AcceptanceError,
        match="resolved_contributions_vs_total",
    ):
        acceptance.audit_campaign(campaign)


def test_alternating_samples_may_start_with_either_lane() -> None:
    campaign = _campaign()
    for batch in _batches(campaign).values():
        assert isinstance(batch, dict)
        rows = batch["measurements"]
        assert isinstance(rows, list)
        for pair_index in range(acceptance.MIN_SAMPLE_PAIRS):
            start = 2 * pair_index
            rows[start], rows[start + 1] = rows[start + 1], rows[start]
            rows[start]["sequence_index"] = start
            rows[start + 1]["sequence_index"] = start + 1
    _seal(campaign)

    assert acceptance.audit_campaign(campaign)["passes"] is True


def _set_candidate_ratio(
    campaign: dict[str, object],
    batch: int,
    ratio: float,
) -> None:
    batch_value = _batches(campaign)[str(batch)]
    assert isinstance(batch_value, dict)
    rows = batch_value["measurements"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        if row["lane"] == "candidate":
            pair_index = int(row["pair_index"])
            row["seconds_per_point"] = (1.0 + 0.001 * (pair_index % 3)) * ratio


def _miss_primary_gain(campaign: dict[str, object]) -> str:
    _set_candidate_ratio(campaign, 128, 0.95)
    return "primary_gain:batch_128"


def _regress_batch_one(campaign: dict[str, object]) -> str:
    _set_candidate_ratio(campaign, 1, 1.06)
    return "batch_1_regression"


def _miss_code_size(campaign: dict[str, object]) -> str:
    metrics = _candidate(campaign)["metrics"]
    assert isinstance(metrics, dict)
    metrics["selected_machine_code_bytes"] = 800
    return "selected_machine_code_reduction"


def _regress_generation(campaign: dict[str, object]) -> str:
    metrics = _candidate(campaign)["metrics"]
    assert isinstance(metrics, dict)
    metrics["generation_seconds"] = 111.0
    return "resource:generation_seconds"


def _allocate_when_warmed(campaign: dict[str, object]) -> str:
    candidate = _candidate(campaign)
    diagnostics = candidate["diagnostics"]
    assert isinstance(diagnostics, dict)
    diagnostics["warmed_allocation_count"] = 1
    stage_plans = candidate["stage_plans"]
    assert isinstance(stage_plans, list)
    stage = stage_plans[0]
    assert isinstance(stage, dict)
    stage_diagnostics = stage["diagnostics"]
    assert isinstance(stage_diagnostics, dict)
    stage_diagnostics["warmed_allocation_count"] = 1
    for batch in _batches(campaign).values():
        assert isinstance(batch, dict)
        rows = batch["measurements"]
        assert isinstance(rows, list)
        for row in rows:
            assert isinstance(row, dict)
            if row["lane"] == "candidate":
                row["warmed_allocation_count"] = 1
    return "zero_warmed_allocations"


def _regress_eager(campaign: dict[str, object]) -> str:
    rows = campaign["non_target_performance"]
    assert isinstance(rows, list)
    eager = next(row for row in rows if row["name"] == "eager")
    eager["candidate_seconds_per_point"] = [1.03] * 7
    return "non_target_performance:eager"


@pytest.mark.parametrize(
    "mutation",
    [
        _miss_primary_gain,
        _regress_batch_one,
        _miss_code_size,
        _regress_generation,
        _allocate_when_warmed,
        _regress_eager,
    ],
)
def test_well_formed_threshold_misses_return_failed_gates(
    mutation: Callable[[dict[str, object]], str],
) -> None:
    campaign = _campaign()
    expected_gate = mutation(campaign)
    _seal(campaign)

    result = acceptance.audit_campaign(campaign)

    assert result["passes"] is False
    gates = result["gates"]
    assert isinstance(gates, dict)
    assert gates[expected_gate]["passes"] is False


def test_unavailable_exact_code_size_fails_closed_without_using_source_bytes() -> None:
    campaign = _campaign()
    baseline = campaign["baseline"]
    assert isinstance(baseline, dict)
    baseline["code_size_metric_available"] = False
    baseline["code_size_metric_kind"] = None
    baseline["selected_machine_code_bytes"] = None
    metrics = _candidate(campaign)["metrics"]
    assert isinstance(metrics, dict)
    metrics["code_size_metric_available"] = False
    metrics["code_size_metric_kind"] = None
    metrics["selected_machine_code_bytes"] = None
    _seal(campaign)

    result = acceptance.audit_campaign(campaign)

    assert result["passes"] is False
    gate = result["gates"]["selected_machine_code_reduction"]
    assert gate["passes"] is False
    assert gate["metric_available"] is False
    diagnostics = result["code_size_diagnostics"]
    assert diagnostics["portable_source_metric_is_machine_code"] is False
    assert diagnostics["baseline_portable_source_application_bytes"] == 4096


def test_code_size_metric_must_be_exact_executed_machine_code() -> None:
    campaign = _campaign()
    metrics = _candidate(campaign)["metrics"]
    assert isinstance(metrics, dict)
    metrics["code_size_metric_kind"] = "portable-source-application-bytes-v1"
    _seal(campaign)

    with pytest.raises(acceptance.AcceptanceError, match="exact executed code"):
        acceptance.audit_campaign(campaign)


def test_content_digest_duplicate_keys_and_cli_exit_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    campaign = _campaign()
    campaign["content_sha256"] = _digest("0")
    with pytest.raises(acceptance.AcceptanceError, match="computed"):
        acceptance.audit_campaign(campaign)

    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text('{"kind": 1, "kind": 2}', encoding="utf-8")
    with pytest.raises(acceptance.AcceptanceError, match="duplicate"):
        acceptance.load_campaign(duplicate_path)

    valid_path = tmp_path / "valid.json"
    valid_path.write_text(json.dumps(_campaign()), encoding="utf-8")
    assert acceptance.main([str(valid_path)]) == 0
    captured = capsys.readouterr()
    assert acceptance.RESULT_KIND in captured.out
