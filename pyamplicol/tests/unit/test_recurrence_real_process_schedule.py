# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path

from pyamplicol.color.plan import (
    build_color_plan,
    build_lc_topology_replay_plan,
)
from pyamplicol.generation.dag_algorithms import (
    infer_minimal_coupling_order_limits,
)
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.recurrence_columnar import (
    ExactComplexRationalV1,
    RecurrenceNormalizationV1,
    build_recurrence_builder_input_v1,
)
from pyamplicol.generation.recurrence_physics import (
    build_recurrence_runtime_metadata,
)
from pyamplicol.generation.recurrence_projection import (
    project_recurrence_process_v1,
)
from pyamplicol.generation.recurrence_template_columnar import (
    build_recurrence_template_input_v1,
)
from pyamplicol.models import (
    BuiltinSMModel,
    CompiledUFOModel,
    compile_model_source,
)
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.prepared_catalog import build_prepared_kernel_catalog
from pyamplicol.models.recurrence_catalog_builder import (
    build_recurrence_template_catalog,
)
from pyamplicol.processes.model import build_model_process_ir

_COMPILED_MODEL_DIGEST = "a" * 64
_PREPARED_PACK_DIGEST = "b" * 64
_UFO_SM_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "sm"
)


def test_ufo_full_ddbar_to_ee_prepares_source_only_electron_mass() -> None:
    compiled = compile_model_source(
        _UFO_SM_ROOT / "sm.json",
        restriction=str((_UFO_SM_ROOT / "restrict_default.json").resolve()),
        use_cache=True,
    )
    model = CompiledUFOModel(compiled)
    process = build_model_process_ir(
        "d d~ > e+ e-",
        compiled.ir,
        color_accuracy="full",
    )
    prepared_catalog = build_prepared_kernel_catalog(model)
    recurrence_catalog = build_recurrence_template_catalog(
        model,
        prepared_catalog,
        compiled_model_digest=_COMPILED_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PREPARED_PACK_DIGEST,
    )
    color_plan = build_color_plan(
        process,
        color_accuracy="full",
        fold_trace_reflections=False,
    )
    logical = project_recurrence_process_v1(
        process,
        color_plan,
        recurrence_catalog,
        layout="contracted-color-union",
        normalization=RecurrenceNormalizationV1(
            ExactComplexRationalV1(1),
            "source-only-parameter-canary-v1",
            "c" * 64,
        ),
        coupling_order_limits=infer_minimal_coupling_order_limits(
            process,
            model=model,
        ),
        model=model,
    )

    electron_mass = next(
        row for row in logical.parameter_projection if row.runtime_name == "Me"
    )
    assert electron_mass.prepared_parameter_id is not None
    metadata = build_recurrence_runtime_metadata(
        logical,
        recurrence_catalog,
        model,
        {
            "averaging_factor": 1.0,
            "coupling_factor": 1.0,
            "symmetry_factor": 1.0,
            "source": "source-only-parameter-canary-v1",
        },
    )
    assert metadata["prepared_parameter_defaults"][
        electron_mass.prepared_parameter_id
    ] == [0.0, 0.0]


def test_builtin_charged_current_all_flow_union_retains_chiral_sources() -> None:
    model = BuiltinSMModel()
    process = build_process_ir("u d~ > e+ ve", color_accuracy="lc")
    recurrence_catalog = build_recurrence_template_catalog(
        model,
        build_prepared_kernel_catalog(model),
        compiled_model_digest=_COMPILED_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PREPARED_PACK_DIGEST,
    )
    color_plan = build_color_plan(
        process,
        color_accuracy="lc",
        fold_trace_reflections=model.lc_trace_reflection_equivalence_is_proven(process),
    )
    logical = project_recurrence_process_v1(
        process,
        color_plan,
        recurrence_catalog,
        layout="all-flow-union",
        normalization=RecurrenceNormalizationV1(
            ExactComplexRationalV1(1),
            "runtime-chiral-source-canary-v1",
            "e" * 64,
        ),
        coupling_order_limits=infer_minimal_coupling_order_limits(
            process,
            model=model,
        ),
        model=model,
    )

    assert tuple(leg.public_label for leg in logical.external_legs) == (1, 2, 3, 4)
    assert all(
        {state.spin_state for state in leg.source_states} == {-1, 1}
        for leg in logical.external_legs
    )
    assert all(len(leg.source_states) == 2 for leg in logical.external_legs)
    columnar = build_recurrence_builder_input_v1(logical)
    assert columnar.table("source_states").row_count == 8


def test_builtin_ddbar_to_ttbar_exposes_legacy_crossed_lc_flow() -> None:
    model = BuiltinSMModel()
    process = build_process_ir("d d~ > t t~", color_accuracy="lc")
    recurrence_catalog = build_recurrence_template_catalog(
        model,
        build_prepared_kernel_catalog(model),
        compiled_model_digest=_COMPILED_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PREPARED_PACK_DIGEST,
    )
    color_plan = build_color_plan(
        process,
        color_accuracy="lc",
        fold_trace_reflections=model.lc_trace_reflection_equivalence_is_proven(process),
    )
    replay = build_lc_topology_replay_plan(color_plan, model)
    expected_public_flows = (
        ("flow:2,1,3,4", (2, 1, 3, 4)),
        ("flow:3,1,2,4", (3, 1, 2, 4)),
    )

    for layout in ("topology-replay", "all-flow-union"):
        logical = project_recurrence_process_v1(
            process,
            color_plan,
            recurrence_catalog,
            layout=layout,
            normalization=RecurrenceNormalizationV1(
                ExactComplexRationalV1(1),
                "legacy-two-line-flow-canary-v1",
                "f" * 64,
            ),
            topology_replay=replay if layout == "topology-replay" else None,
            coupling_order_limits=infer_minimal_coupling_order_limits(
                process,
                model=model,
            ),
            model=model,
        )

        assert (
            tuple(
                (
                    flow.public_id,
                    tuple(
                        process.legs[source_slot].label
                        for source_slot in flow.word_source_slots
                    ),
                )
                for flow in logical.public_flows
            )
            == expected_public_flows
        )
        # Construction remains on the canonical fundamental-to-
        # antifundamental sector word; only its public selector is aliased.
        assert tuple(
            tuple(
                process.legs[source_slot].label
                for source_slot in sector.word_source_slots
            )
            for sector in logical.physical_sectors
        ) == ((2, 1, 3, 4), (2, 4, 3, 1))
        build_recurrence_builder_input_v1(logical)


def test_builtin_two_line_contracted_color_keeps_unique_construction_flows() -> None:
    model = BuiltinSMModel()
    recurrence_catalog = build_recurrence_template_catalog(
        model,
        build_prepared_kernel_catalog(model),
        compiled_model_digest=_COMPILED_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PREPARED_PACK_DIGEST,
    )

    for expression in ("d d~ > t t~", "d d~ > d d~"):
        for accuracy in ("nlc", "full"):
            process = build_process_ir(expression, color_accuracy=accuracy)
            color_plan = build_color_plan(
                process,
                color_accuracy=accuracy,
                fold_trace_reflections=model.lc_trace_reflection_equivalence_is_proven(
                    process
                ),
            )
            logical = project_recurrence_process_v1(
                process,
                color_plan,
                recurrence_catalog,
                layout="contracted-color-union",
                normalization=RecurrenceNormalizationV1(
                    ExactComplexRationalV1(1),
                    "contracted-two-line-flow-canary-v1",
                    "0" * 64,
                ),
                coupling_order_limits=infer_minimal_coupling_order_limits(
                    process,
                    model=model,
                ),
                model=model,
            )

            public_ids = tuple(flow.public_id for flow in logical.public_flows)
            construction_ids = tuple(
                sector.public_id for sector in logical.physical_sectors
            )
            assert public_ids == construction_ids
            assert len(set(public_ids)) == len(public_ids)
            assert tuple(
                (
                    flow.construction_sector_id,
                    flow.word_source_slots,
                )
                for flow in logical.public_flows
            ) == tuple(
                (
                    sector.sector_id,
                    sector.word_source_slots,
                )
                for sector in logical.physical_sectors
            )
            build_recurrence_builder_input_v1(logical)


def test_sm_process_projects_model_generic_topology_replay_input() -> None:
    summaries: dict[str, dict[str, int]] = {}
    dag_shapes: dict[str, tuple[int, int]] = {}
    for model_source in ("built-in", "ufo-sm"):
        if model_source == "built-in":
            model = BuiltinSMModel()
            process = build_process_ir("d d~ > z g", color_accuracy="lc")
        else:
            compiled = compile_model_source(
                _UFO_SM_ROOT / "sm.json",
                restriction=str((_UFO_SM_ROOT / "restrict_default.json").resolve()),
                use_cache=True,
            )
            model = CompiledUFOModel(compiled)
            process = build_model_process_ir("d d~ > z g", compiled.ir)

        prepared_catalog = build_prepared_kernel_catalog(model)
        recurrence_catalog = build_recurrence_template_catalog(
            model,
            prepared_catalog,
            compiled_model_digest=_COMPILED_MODEL_DIGEST,
            prepared_kernel_pack_digest=_PREPARED_PACK_DIGEST,
        )
        color_plan = build_color_plan(
            process,
            color_accuracy="lc",
            fold_trace_reflections=model.lc_trace_reflection_equivalence_is_proven(
                process
            ),
        )
        replay = build_lc_topology_replay_plan(color_plan, model)
        coupling_order_limits = infer_minimal_coupling_order_limits(
            process,
            model=model,
        )
        dag = compile_generic_dag(
            process,
            model=model,
            max_coupling_orders=coupling_order_limits,
        )
        logical = project_recurrence_process_v1(
            process,
            color_plan,
            recurrence_catalog,
            layout="topology-replay",
            normalization=RecurrenceNormalizationV1(
                ExactComplexRationalV1(1),
                "structural-canary-v1",
                "c" * 64,
            ),
            topology_replay=replay,
            coupling_order_limits=coupling_order_limits,
            model=model,
        )

        sources_by_numeric_id = tuple(
            sorted(recurrence_catalog.sources, key=lambda row: row.template_id)
        )
        states_by_numeric_id = tuple(
            sorted(recurrence_catalog.current_states, key=lambda row: row.template_id)
        )
        states_by_template_id = {
            state.template_id: state for state in recurrence_catalog.current_states
        }
        crossed_internal_spin_pairs = tuple(
            (
                state.spin_state,
                sources_by_numeric_id[state.source_template_id].spin_state,
            )
            for leg in logical.external_legs
            if leg.is_initial
            for state in leg.source_states
        )
        assert any(
            public != canonical for public, canonical in crossed_internal_spin_pairs
        )
        crossed_internal_chirality_pairs = tuple(
            (
                states_by_numeric_id[state.current_state_template_id].chirality,
                states_by_template_id[
                    sources_by_numeric_id[state.source_template_id].state_template_id
                ].chirality,
            )
            for leg in logical.external_legs
            if leg.is_initial
            for state in leg.source_states
        )
        assert any(
            effective != canonical
            for effective, canonical in crossed_internal_chirality_pairs
        )
        assert all(
            state.chirality
            == states_by_numeric_id[state.current_state_template_id].chirality
            for leg in logical.external_legs
            for state in leg.source_states
        )

        columnar = build_recurrence_builder_input_v1(logical)
        build_recurrence_template_input_v1(recurrence_catalog)
        assert len(columnar.digest) == 64
        assert len(columnar.fermion_pairing_digest or "") == 64
        assert columnar.table("external_legs").row_count == len(process.legs)
        assert columnar.table("physical_lc_sectors").row_count == (
            color_plan.sector_count
        )
        assert columnar.table("public_lc_flows").row_count == color_plan.sector_count
        assert columnar.table("replay_partitions").row_count == len(
            logical.replay_partitions
        )
        assert columnar.table("replay_targets").row_count == sum(
            len(partition.targets) for partition in logical.replay_partitions
        )
        assert columnar.table("source_states").row_count > len(process.legs)
        summaries[model_source] = {
            table.name: table.row_count
            for table in columnar.tables
            if table.name
            in {
                "external_legs",
                "physical_lc_sectors",
                "public_lc_flows",
                "replay_partitions",
                "replay_targets",
                "source_states",
            }
        }
        dag_shapes[model_source] = (len(dag.currents), len(dag.interactions))

    assert summaries["built-in"] == summaries["ufo-sm"]
    assert dag_shapes == {"built-in": (31, 34), "ufo-sm": (31, 34)}


def test_sm_recurrence_closure_projections_match_without_forest_aliases() -> None:
    """Exercise every LC closure family before public recurrence dispatch."""

    expressions = (
        "g g > g g",
        "d d~ > z g",
        "u d~ > w+",
        "d d~ > u u~",
        "d d~ > d d~",
        "d d~ > u u~ s s~",
    )
    summaries: dict[str, dict[str, tuple[int, ...]]] = {}
    compiled_ufo = compile_model_source(
        _UFO_SM_ROOT / "sm.json",
        restriction=str((_UFO_SM_ROOT / "restrict_default.json").resolve()),
        use_cache=True,
    )
    for model_source, model in (
        ("built-in", BuiltinSMModel()),
        ("ufo-sm", CompiledUFOModel(compiled_ufo)),
    ):
        prepared_catalog = build_prepared_kernel_catalog(model)
        recurrence_catalog = build_recurrence_template_catalog(
            model,
            prepared_catalog,
            compiled_model_digest=_COMPILED_MODEL_DIGEST,
            prepared_kernel_pack_digest=_PREPARED_PACK_DIGEST,
        )
        summaries[model_source] = {}
        for expression in expressions:
            process = (
                build_process_ir(expression, color_accuracy="lc")
                if model_source == "built-in"
                else build_model_process_ir(expression, compiled_ufo.ir)
            )
            color_plan = build_color_plan(
                process,
                color_accuracy="lc",
                fold_trace_reflections=model.lc_trace_reflection_equivalence_is_proven(
                    process
                ),
            )
            replay = build_lc_topology_replay_plan(color_plan, model)
            logical = project_recurrence_process_v1(
                process,
                color_plan,
                recurrence_catalog,
                layout="topology-replay",
                normalization=RecurrenceNormalizationV1(
                    ExactComplexRationalV1(1),
                    "closure-family-canary-v1",
                    "d" * 64,
                ),
                topology_replay=replay,
                coupling_order_limits=infer_minimal_coupling_order_limits(
                    process,
                    model=model,
                ),
                model=model,
            )
            columnar = build_recurrence_builder_input_v1(logical)
            pairing = logical.fermion_pairing_catalog
            assert pairing is not None
            assert pairing.source_count == len(process.legs)
            assert len(columnar.fermion_pairing_digest or "") == 64
            assert len(pairing.topology_digest) == 64
            assert len(pairing.semantic_digest) == 64
            assert len(pairing.rules) >= 1
            assert columnar.table("physical_lc_sectors").row_count == (
                color_plan.sector_count
            )
            assert columnar.table("replay_targets").row_count == sum(
                len(partition.targets) for partition in logical.replay_partitions
            )
            if expression == "g g > g g":
                assert not pairing.endpoints
                assert not pairing.pairing_classes
                assert len(pairing.rules) == 1
            elif expression == "d d~ > u u~":
                assert len(pairing.endpoints) == 4
                assert len(pairing.pairing_classes) == 1
                assert len(pairing.rules) == 2
            elif expression == "u d~ > w+":
                assert len(pairing.endpoints) == 2
                assert len(pairing.pairing_classes) == 1
                assert len(pairing.rules) == 1
            elif expression == "d d~ > d d~":
                assert len(pairing.endpoints) == 4
                assert len(pairing.pairing_classes) == 1
                assert len(pairing.rules) == 2
            elif expression == "d d~ > u u~ s s~":
                assert len(pairing.endpoints) == 6
                assert len(pairing.pairing_classes) == 2
                assert len(pairing.rules) == 2
            summaries[model_source][expression] = (
                len(pairing.endpoints),
                len(pairing.pairing_classes),
                len(pairing.rules),
                color_plan.sector_count,
                columnar.table("replay_targets").row_count,
            )

    assert summaries["built-in"] == summaries["ufo-sm"]
    assert summaries["built-in"]["d d~ > u u~ s s~"][3] > 1
