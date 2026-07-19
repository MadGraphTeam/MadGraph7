# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections import Counter

from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.eager_lowering import (
    EAGER_RUNTIME_KIND,
    MappingEagerKernelResolver,
    PreparedCatalogEagerKernelResolver,
    lower_eager_execution_tables,
)
from pyamplicol.generation.eager_tables import MISSING_U32
from pyamplicol.generation.runtime_schema import build_runtime_schema
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.prepared_catalog import build_prepared_kernel_catalog


def _gluon_scattering_tables():
    model = BuiltinSMModel()
    dag = compile_generic_dag(build_process_ir("g g > g g"), model=model)
    schema = build_runtime_schema(dag, model, process_id="gg_gg")
    propagated = {
        (int(slot["particle_id"]), int(slot["chirality"]))
        for slot in schema["value_storage"]["value_slots"]
        if slot["variant"] == "propagated"
    }
    resolver = MappingEagerKernelResolver(
        vertex_kernels={kind: 100 + kind for kind in dag.required_vertex_kinds},
        propagator_kernels={key: 1000 + index for index, key in enumerate(propagated)},
        closure_kernels={},
    )
    tables = lower_eager_execution_tables(dag, model, schema, resolver)
    return model, dag, schema, tables


def test_eager_lowering_preserves_compiled_evaluation_groups_and_fanout() -> None:
    _model, dag, _schema, tables = _gluon_scattering_tables()

    assert tables.process_key == "gg_gg"
    assert tables.invocation_count == dag.interaction_evaluation_count
    assert tables.attachment_count == len(dag.interactions)
    assert sum(len(stage.finalizations) for stage in tables.stages) == len(
        {interaction.result_id for interaction in dag.interactions}
    )

    dag_fanout = Counter(
        interaction.evaluation_group_id for interaction in dag.interactions
    )
    eager_fanout = Counter(
        invocation.attachment_count
        for stage in tables.stages
        for invocation in stage.invocations
    )
    assert eager_fanout[2] == sum(size == 2 for size in dag_fanout.values())
    assert eager_fanout[1] == sum(size == 1 for size in dag_fanout.values())


def test_eager_attachment_factors_use_the_compiled_representative_ratio() -> None:
    _model, dag, schema, tables = _gluon_scattering_tables()
    interaction_by_stage = {
        int(stage["stage_index"]): [
            dag.interactions[int(interaction_id)]
            for interaction_id in stage["interaction_ids"]
        ]
        for stage in schema["stages"]
    }

    for stage in tables.stages:
        grouped: dict[tuple[str, int], list[object]] = {}
        for interaction in interaction_by_stage[stage.stage_index]:
            key = (
                ("group", int(interaction.evaluation_group_id))
                if interaction.evaluation_group_id is not None
                else ("interaction", interaction.id)
            )
            grouped.setdefault(key, []).append(interaction)
        for invocation, interactions in zip(
            stage.invocations,
            grouped.values(),
            strict=True,
        ):
            representative_factor = complex(*interactions[0].evaluation_factor)
            actual = stage.attachments[
                invocation.attachment_start : invocation.attachment_start
                + invocation.attachment_count
            ]
            expected = tuple(
                complex(*interaction.color_weight)
                * complex(*interaction.evaluation_factor)
                / representative_factor
                for interaction in interactions
            )
            assert (
                tuple(complex(row.factor_real, row.factor_imag) for row in actual)
                == expected
            )


def test_eager_lowering_emits_standalone_binary_table_metadata() -> None:
    _model, _dag, _schema, tables = _gluon_scattering_tables()

    metadata = tables.to_metadata()
    payloads = tables.binary_payloads()

    assert metadata["kind"] == EAGER_RUNTIME_KIND
    assert metadata["process_key"] == "gg_gg"
    assert metadata["required_runtime_capabilities"] == [
        "rusticol.eager-dag.complex-f64.v1"
    ]
    assert set(payloads) == {
        "eager/couplings.bin",
        "eager/closures.bin",
        *{
            f"eager/stage-{stage.stage_index}-{kind}.bin"
            for stage in tables.stages
            for kind in ("invocations", "attachments", "finalizations")
        },
    }
    for stage, record in zip(tables.stages, metadata["stages"], strict=True):
        assert len(payloads[record["invocations"]["path"]]) == (
            len(stage.invocations) * record["invocations"]["row_size"]
        )


def test_direct_contractions_remain_native_and_couplings_are_parameterized() -> None:
    _model, _dag, _schema, tables = _gluon_scattering_tables()

    assert tables.closures
    assert all(row.kernel_id == MISSING_U32 for row in tables.closures)
    assert all(
        row.applies_kernel is row.stores_propagated
        for stage in tables.stages
        for row in stage.finalizations
    )
    assert all(row.coupling_slot_id == MISSING_U32 for row in tables.closures)
    assert tables.couplings
    assert all(row.real_parameter_id != MISSING_U32 for row in tables.couplings)


def test_prepared_catalog_resolves_every_real_dag_orientation() -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(build_process_ir("g g > g g"), model=model)
    schema = build_runtime_schema(dag, model, process_id="gg_gg")
    catalog = build_prepared_kernel_catalog(model)
    resolver = PreparedCatalogEagerKernelResolver(dag, catalog.resolver_manifest())

    tables = lower_eager_execution_tables(dag, model, schema, resolver)

    known_kernel_ids = set(catalog.by_id)
    assert tables.invocation_count == dag.interaction_evaluation_count
    assert {
        invocation.kernel_id
        for stage in tables.stages
        for invocation in stage.invocations
    } <= known_kernel_ids

    checked_reflected_gluon = False
    for stage, stage_record in zip(tables.stages, schema["stages"], strict=True):
        groups: dict[int, list[object]] = {}
        for interaction_id in stage_record["interaction_ids"]:
            interaction = dag.interactions[int(interaction_id)]
            groups.setdefault(int(interaction.evaluation_group_id), []).append(
                interaction
            )
        input_slots = {
            int(slot["current_id"]): int(slot["value_slot_id"])
            for slot in schema["value_storage"]["value_slots"]
            if int(slot["value_slot_id"]) in stage_record["input_value_slot_ids"]
        }
        for invocation, interactions in zip(
            stage.invocations, groups.values(), strict=True
        ):
            representative = interactions[0]
            if representative.vertex_kind != 3:
                continue
            resolution = resolver.vertex_kernel(representative)
            assert resolution.canonical_input_order == (1, 0)
            assert resolution.normalization_factor == (-1.0, 0.0)
            assert invocation.left_value_slot_id == input_slots[
                representative.right_id
            ]
            assert invocation.right_value_slot_id == input_slots[
                representative.left_id
            ]
            first_attachment = stage.attachments[invocation.attachment_start]
            expected = (
                complex(*representative.color_weight)
                * complex(*resolution.normalization_factor)
            )
            assert complex(
                first_attachment.factor_real, first_attachment.factor_imag
            ) == expected
            checked_reflected_gluon = True
            break
        if checked_reflected_gluon:
            break
    assert checked_reflected_gluon
