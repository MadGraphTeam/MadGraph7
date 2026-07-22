# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path

from pyamplicol import _rusticol
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


def test_sm_process_constructs_model_generic_topology_replay_schedule() -> None:
    summaries: dict[str, dict[str, object]] = {}
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

        result = _rusticol._validate_recurrence_builder_input_v1(
            build_recurrence_builder_input_v1(logical),
            build_recurrence_template_input_v1(recurrence_catalog),
            construct_schedule=True,
        )

        assert result["composite_authenticated"] is True
        assert result["schedule_constructed"] is True
        schedule = result["inspection_summary"]["schedule"]
        assert schedule["source_current_count"] > 0
        assert schedule["current_count"] > schedule["source_current_count"]
        assert schedule["current_count_by_support_size"][3] > 0
        assert schedule["contribution_count"] > 0
        assert len(schedule["referenced_quantum_flow_template_ids"]) >= 2
        assert schedule["finalization_count"] > 0
        assert (
            schedule["identity_finalization_count_by_support_size"][3]
            == schedule["current_count_by_support_size"][3]
        )
        assert schedule["propagated_finalization_count_by_support_size"][3] == 0
        assert schedule["target_sector_count"] == color_plan.sector_count
        assert schedule["resolved_helicity_count"] > 1
        assert (
            schedule["retained_helicity_count"]
            >= schedule["resolved_helicity_count"]
        )
        assert schedule["structural_zero_helicity_count"] == (
            schedule["retained_helicity_count"] - schedule["resolved_helicity_count"]
        )
        assert schedule["structural_zero_helicity_count"] > 0
        assert schedule["amplitude_destination_count"] <= (
            schedule["target_sector_count"] * schedule["resolved_helicity_count"]
        )
        assert schedule["closure_term_count"] > 0
        summaries[model_source] = schedule
        dag_shapes[model_source] = (len(dag.currents), len(dag.interactions))
        assert schedule["current_count"] == len(dag.currents)
        assert schedule["contribution_count"] == len(dag.interactions)

    for field in (
        "current_count_by_support_size",
        "contribution_count",
        "finalization_count",
        "identity_finalization_count_by_support_size",
        "propagated_finalization_count_by_support_size",
        "retained_helicity_count",
        "resolved_helicity_count",
        "structural_zero_helicity_count",
        "amplitude_destination_count",
        "target_sector_count",
        "closure_term_count",
    ):
        assert summaries["built-in"][field] == summaries["ufo-sm"][field]
    assert dag_shapes == {"built-in": (31, 34), "ufo-sm": (31, 34)}


def test_sm_recurrence_closure_topologies_match_without_forest_aliases() -> None:
    """Exercise every LC closure family before public recurrence dispatch."""

    expressions = (
        "g g > g g",
        "d d~ > z g",
        "d d~ > u u~",
        "d d~ > d d~",
        "d d~ > u u~ s s~",
    )
    summaries: dict[str, dict[str, dict[str, object]]] = {}
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
        template_input = build_recurrence_template_input_v1(recurrence_catalog)
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
            )
            result = _rusticol._validate_recurrence_builder_input_v1(
                build_recurrence_builder_input_v1(logical),
                template_input,
                construct_schedule=True,
            )
            pairing = result["inspection_summary"]["fermion_pairing"]
            assert pairing is not None
            assert pairing["source_count"] == len(process.legs)
            assert len(pairing["columnar_digest"]) == 64
            assert len(pairing["topology_digest"]) == 64
            assert len(pairing["semantic_digest"]) == 64
            assert pairing["rule_count"] >= 1
            schedule = result["inspection_summary"]["schedule"]
            assert schedule["closure_term_count"] >= schedule["target_sector_count"]
            assert schedule["amplitude_destination_count"] > 0
            if expression == "d d~ > u u~":
                assert schedule["current_count_by_support_size"][3] > 0
                assert pairing["endpoint_count"] == 4
                assert pairing["pairing_class_count"] == 2
                assert pairing["rule_count"] == 1
            elif expression == "d d~ > d d~":
                assert pairing["endpoint_count"] == 4
                assert pairing["pairing_class_count"] == 1
                assert pairing["rule_count"] == 2
            elif expression == "d d~ > u u~ s s~":
                assert pairing["endpoint_count"] == 6
                assert pairing["pairing_class_count"] == 3
                assert pairing["rule_count"] == 1
            summaries[model_source][expression] = schedule

    for expression in expressions:
        for field in (
            "current_count_by_support_size",
            "contribution_count",
            "finalization_count",
            "resolved_helicity_count",
            "structural_zero_helicity_count",
            "amplitude_destination_count",
            "target_sector_count",
            "closure_term_count",
        ):
            assert summaries["built-in"][expression][field] == summaries["ufo-sm"][
                expression
            ][field], (
                expression,
                field,
                summaries["built-in"][expression][field],
                summaries["ufo-sm"][expression][field],
            )

    three_line = summaries["built-in"]["d d~ > u u~ s s~"]
    assert three_line["target_sector_count"] > 1
    assert three_line["closure_term_count"] > three_line["amplitude_destination_count"]
