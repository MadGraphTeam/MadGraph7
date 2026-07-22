# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from pyamplicol.color.plan import (
    GenericColorPlan,
    LCColorSector,
    LCColorSectorReplayPartition,
    LCColorTopologyReplayPlan,
    LCOpenColorLine,
)
from pyamplicol.generation.recurrence_columnar import (
    ExactComplexRationalV1 as BuilderExact,
)
from pyamplicol.generation.recurrence_columnar import (
    RecurrenceNormalizationV1,
    build_recurrence_builder_input_v1,
)
from pyamplicol.generation.recurrence_projection import (
    RecurrenceGenerationSliceV1,
    RecurrenceProjectionError,
    project_recurrence_process_v1,
)
from pyamplicol.models.recurrence_template import (
    CurrentStateTemplateV1,
    EvaluatorBindingV1,
    ExactComplexRationalV1,
    ParameterTemplateV1,
    RecurrenceTemplateCatalog,
    SourceTemplateV1,
)
from pyamplicol.processes.ir import (
    CanonicalProcessIR,
    ColorEndpointSummary,
    ProcessLegIR,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _crossing(
    *,
    momentum_transform: str = "negate-four-momentum",
    phase: tuple[int, int] = (1, 0),
) -> str:
    return json.dumps(
        {
            "chirality_factor": -1,
            "helicity_factor": -1,
            "momentum_transform": momentum_transform,
            "phase": {
                "imag_denominator": "1",
                "imag_numerator": str(phase[1]),
                "real_denominator": "1",
                "real_numerator": str(phase[0]),
            },
            "spin_state_factor": -1,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _state(
    template_id: str,
    particle_id: int,
    *,
    family: str,
    dimension: int,
) -> CurrentStateTemplateV1:
    return CurrentStateTemplateV1(
        template_id=template_id,
        particle_id=particle_id,
        anti_particle_id=-particle_id if abs(particle_id) == 1 else particle_id,
        species_id=f"species:{abs(particle_id)}",
        orientation=(
            "self-conjugate"
            if particle_id == 21
            else "particle"
            if particle_id > 0
            else "antiparticle"
        ),
        statistics="fermion" if family == "fermion" else "boson",
        color_representation=8 if particle_id == 21 else 3,
        basis=family,
        tensor_ordering=tuple(f"{family}:c{index}" for index in range(dimension)),
        dimension=dimension,
        chirality=0,
        auxiliary_kind=None,
        mass_parameter_id=None,
        width_parameter_id=None,
    )


def _source(
    state: CurrentStateTemplateV1,
    *,
    helicity: int,
    family: str,
) -> tuple[SourceTemplateV1, EvaluatorBindingV1]:
    token = f"source:{state.particle_id}:{helicity}"
    resolver = f"resolver:{state.particle_id}:{helicity}"
    signature = _sha(f"callable:{token}")
    source = SourceTemplateV1(
        template_id=token,
        state_template_id=state.template_id,
        crossing=_crossing(),
        wavefunction_family=family,
        helicity=helicity,
        spin_state=helicity,
        wavefunction_expression_digest=signature,
        evaluator_resolver_key=resolver,
    )
    output_layout = tuple(
        f"source-component:{index}" for index in range(state.dimension)
    )
    binding = EvaluatorBindingV1(
        resolver_key=resolver,
        prepared_kernel_id=None,
        callable_kind="rusticol-template",
        runtime_template=(
            f"rusticol.source-fill.{family}.v1:{signature[:24]}"
        ),
        contract_kind="source",
        callable_signature=signature,
        input_state_template_ids=(),
        output_state_template_id=state.template_id,
        input_layout=("momentum:energy",),
        output_layout=output_layout,
        exact_expression_digests=tuple(
            _sha(f"expression:{token}:{index}") for index in range(state.dimension)
        ),
        semantic_template_ids=(source.template_id,),
    )
    return source, binding


def _catalog(*, omit_particle: int | None = None) -> RecurrenceTemplateCatalog:
    states = (
        _state("state:anti-u", -1, family="fermion", dimension=2),
        _state("state:u", 1, family="fermion", dimension=2),
        _state("state:g", 21, family="vector", dimension=4),
    )
    source_pairs = tuple(
        _source(
            state,
            helicity=helicity,
            family="vector" if state.particle_id == 21 else "fermion",
        )
        for state in states
        if state.particle_id != omit_particle
        for helicity in (-1, 1)
    )
    parameter = ParameterTemplateV1(
        template_id="parameter:alpha_s",
        name="alpha_s",
        parameter_kind="external",
        value_type="real",
        mutable=True,
        default_value=ExactComplexRationalV1(1, 10, 0, 1),
        exact_expression_digest=None,
        dependency_parameter_ids=(),
    )
    return RecurrenceTemplateCatalog.create(
        compiled_model_digest=_sha("compiled-model"),
        prepared_kernel_pack_digest=_sha("prepared-kernel-pack"),
        parameters=(parameter,),
        current_states=states,
        sources=tuple(pair[0] for pair in source_pairs),
        evaluator_bindings=tuple(pair[1] for pair in source_pairs),
    )


def _process(*, color_accuracy: str = "lc") -> CanonicalProcessIR:
    return CanonicalProcessIR(
        process="u u~ > g g",
        key="u_ubar_to_g_g",
        color_accuracy=color_accuracy,
        legs=(
            ProcessLegIR(
                1,
                "initial",
                "u",
                "u~",
                1,
                -1,
                "fermion",
                "fermion",
                "antifundamental",
                "antiparticle",
            ),
            ProcessLegIR(
                2,
                "initial",
                "u~",
                "u",
                -1,
                1,
                "fermion",
                "fermion",
                "fundamental",
                "particle",
            ),
            ProcessLegIR(
                3,
                "final",
                "g",
                "g",
                21,
                21,
                "boson",
                "vector",
                "adjoint",
                "self-conjugate",
            ),
            ProcessLegIR(
                4,
                "final",
                "g",
                "g",
                21,
                21,
                "boson",
                "vector",
                "adjoint",
                "self-conjugate",
            ),
        ),
        color_endpoints=ColorEndpointSummary(1, 1, 1),
    )


def _pure_gluon_process() -> CanonicalProcessIR:
    legs = tuple(
        ProcessLegIR(
            label,
            "initial" if label <= 2 else "final",
            "g",
            "g",
            21,
            21,
            "boson",
            "vector",
            "adjoint",
            "self-conjugate",
        )
        for label in range(1, 5)
    )
    return CanonicalProcessIR(
        process="g g > g g",
        key="g_g_to_g_g",
        color_accuracy="lc",
        legs=legs,
        color_endpoints=ColorEndpointSummary(0, 0, 0),
    )


def _color_plan(
    process: CanonicalProcessIR,
    *,
    truncated: bool = False,
    bad_label: bool = False,
) -> GenericColorPlan:
    final_label = 99 if bad_label else 4
    return GenericColorPlan(
        process=process,
        color_accuracy=process.color_accuracy,
        sectors=(
            LCColorSector(
                id=0,
                kind="open-lines",
                open_color_lines=(LCOpenColorLine(2, 1, (3, final_label)),),
                word_labels=(2, 3, final_label, 1),
            ),
            LCColorSector(
                id=1,
                kind="open-lines",
                open_color_lines=(LCOpenColorLine(2, 1, (final_label, 3)),),
                word_labels=(2, final_label, 3, 1),
            ),
        ),
        truncated=truncated,
    )


def _replay() -> LCColorTopologyReplayPlan:
    return LCColorTopologyReplayPlan(
        physical_sector_ids=(0, 1),
        partitions=(
            LCColorSectorReplayPartition(
                representative_sector_id=0,
                materialized_sector_id=0,
                active_sector_ids=(0, 1),
                label_permutations=((), ((3, 4), (4, 3))),
                replay_weights=(1.0, 1.0),
                replay_signs=(1, -1),
                proof_algorithm="canonical-recurrence-replay-witness-v1",
                proof_digest=_sha("replay-proof"),
            ),
        ),
        residual_sector_ids=(),
    )


def _normalization() -> RecurrenceNormalizationV1:
    return RecurrenceNormalizationV1(
        factor=BuilderExact(1, 4),
        convention="spin-colour-average-v1",
        semantic_digest=_sha("normalization"),
    )


def test_topology_replay_projects_exact_axes_and_generation_coverage() -> None:
    process = _process()
    logical = project_recurrence_process_v1(
        process,
        _color_plan(process),
        _catalog(),
        layout="topology-replay",
        topology_replay=_replay(),
        normalization=_normalization(),
        generation_slice=RecurrenceGenerationSliceV1(
            selected_public_flow_ids=(1,),
            # Initial-state crossing maps the catalog's -1 source to +1.
            selected_source_helicities=((1, 1),),
        ),
        coupling_order_limits={"qcd": 4},
    )

    assert logical.process_id == "u_ubar_to_g_g"
    assert tuple(leg.public_label for leg in logical.external_legs) == (1, 2, 3, 4)
    assert tuple(len(leg.source_states) for leg in logical.external_legs) == (
        2,
        2,
        2,
        2,
    )
    assert tuple(leg.is_initial for leg in logical.external_legs) == (
        True,
        True,
        False,
        False,
    )
    assert tuple(
        state.public_helicity for state in logical.external_legs[0].source_states
    ) == (-1, 1)
    assert all(
        state.crossing_phase == BuilderExact(1)
        for state in logical.external_legs[0].source_states
    )
    assert all(
        state.momentum_sign == -1 for state in logical.external_legs[0].source_states
    )
    assert logical.selected_public_flow_ids == (1,)
    assert logical.selected_source_coverage is not None
    assert logical.selected_source_coverage[0].source_slot == 0
    assert logical.selected_source_coverage[0].source_state_indices == (1,)
    assert tuple(sector.public_id for sector in logical.physical_sectors) == (
        "flow:2,3,4,1",
        "flow:2,4,3,1",
    )
    assert len(logical.replay_partitions) == 1
    replayed = sorted(
        logical.replay_partitions[0].targets,
        key=lambda target: target.sector_id,
    )
    assert replayed[0].external_permutation == (0, 1, 2, 3)
    assert replayed[1].external_permutation == (0, 1, 3, 2)
    assert replayed[1].source_slot_permutation == (0, 1, 3, 2)
    assert replayed[1].fermion_sign == -1
    assert logical.coupling_limits[0].name == "QCD"
    assert logical.parameter_projection[0].runtime_name == "alpha_s"
    assert logical.parameter_projection[0].prepared_parameter_id is None

    # The result is accepted unchanged by the existing deterministic encoder.
    assert len(build_recurrence_builder_input_v1(logical).canonical_digest) == 64


def test_all_flow_union_retains_all_sectors_without_replay_partitions() -> None:
    process = _process()
    logical = project_recurrence_process_v1(
        process,
        _color_plan(process),
        _catalog(),
        layout="all-flow-union",
        topology_replay=_replay(),
        normalization=_normalization(),
    )

    assert logical.selected_public_flow_ids is None
    assert tuple(sector.sector_id for sector in logical.physical_sectors) == (0, 1)
    assert logical.replay_partitions == ()
    assert len(build_recurrence_builder_input_v1(logical).canonical_digest) == 64


def test_folded_trace_reflection_remains_a_public_runtime_flow() -> None:
    process = _pure_gluon_process()
    plan = GenericColorPlan(
        process=process,
        color_accuracy="lc",
        sectors=(
            LCColorSector(
                id=0,
                kind="single-trace",
                trace_labels=(1, 2, 3, 4),
                word_labels=(1, 2, 3, 4),
            ),
        ),
        trace_reflections_folded=True,
    )
    logical = project_recurrence_process_v1(
        process,
        plan,
        _catalog(),
        layout="all-flow-union",
        normalization=_normalization(),
    )

    assert tuple(flow.public_id for flow in logical.public_flows) == (
        "flow:1,2,3,4",
        "flow:1,4,3,2",
    )
    assert tuple(flow.construction_sector_id for flow in logical.public_flows) == (
        0,
        0,
    )
    assert logical.public_flows[1].source_slot_permutation == (0, 3, 2, 1)


@pytest.mark.parametrize("flow_id", [0, 1])
def test_folded_trace_flows_can_be_specialized_independently(flow_id: int) -> None:
    process = _pure_gluon_process()
    plan = GenericColorPlan(
        process=process,
        color_accuracy="lc",
        sectors=(
            LCColorSector(
                id=0,
                kind="single-trace",
                trace_labels=(1, 2, 3, 4),
                word_labels=(1, 2, 3, 4),
            ),
        ),
        trace_reflections_folded=True,
    )
    logical = project_recurrence_process_v1(
        process,
        plan,
        _catalog(),
        layout="topology-replay",
        normalization=_normalization(),
        generation_slice=RecurrenceGenerationSliceV1(
            selected_public_flow_ids=(flow_id,)
        ),
    )
    columns = build_recurrence_builder_input_v1(logical)

    assert logical.selected_public_flow_ids == (flow_id,)
    assert tuple(flow.flow_id for flow in logical.public_flows) == (0, 1)
    assert tuple(
        int(value)
        for value in columns.table("selected_public_flow_coverage").column("flow_id")
    ) == (flow_id,)


def test_crossed_source_carries_effective_momentum_transform_and_phase() -> None:
    base = _catalog()
    source = next(
        item
        for item in base.sources
        if item.state_template_id == "state:anti-u" and item.helicity == -1
    )
    changed = replace(
        source,
        crossing=_crossing(momentum_transform="identity", phase=(0, 1)),
        semantic_digest="",
    )
    catalog = RecurrenceTemplateCatalog.create(
        compiled_model_digest=base.header.compiled_model_digest,
        prepared_kernel_pack_digest=base.header.prepared_kernel_pack_digest,
        parameters=base.parameters,
        current_states=base.current_states,
        sources=tuple(changed if item == source else item for item in base.sources),
        evaluator_bindings=base.evaluator_bindings,
    )
    process = _process()
    logical = project_recurrence_process_v1(
        process,
        _color_plan(process),
        catalog,
        layout="all-flow-union",
        normalization=_normalization(),
    )

    crossed = next(
        item
        for item in logical.external_legs[0].source_states
        if item.source_template_id
        == sorted(template.template_id for template in catalog.sources).index(
            changed.template_id
        )
    )
    assert crossed.momentum_sign == 1
    assert crossed.crossing_phase == BuilderExact(0, 1, 1, 1)
    assert all(
        state.momentum_sign == 1 and state.crossing_phase == BuilderExact(1)
        for state in logical.external_legs[2].source_states
    )


def test_projection_and_columnar_digest_are_deterministic() -> None:
    process = _process()
    arguments = dict(
        process=process,
        color_plan=_color_plan(process),
        template_catalog=_catalog(),
        layout="topology-replay",
        topology_replay=_replay(),
        normalization=_normalization(),
        coupling_order_limits={"QED": 2, "QCD": 4},
    )
    first = project_recurrence_process_v1(**arguments)
    second = project_recurrence_process_v1(**arguments)

    assert first == second
    first_columns = build_recurrence_builder_input_v1(first)
    second_columns = build_recurrence_builder_input_v1(second)
    assert first_columns.canonical_digest == second_columns.canonical_digest


@pytest.mark.parametrize("accuracy", ["nlc", "full"])
def test_non_lc_projection_fails_closed(accuracy: str) -> None:
    process = _process(color_accuracy=accuracy)
    with pytest.raises(RecurrenceProjectionError, match="supports LC only"):
        project_recurrence_process_v1(
            process,
            _color_plan(process),
            _catalog(),
            layout="all-flow-union",
            normalization=_normalization(),
        )


def test_projection_rejects_truncated_plan_and_allows_residual_only_replay() -> None:
    process = _process()
    with pytest.raises(RecurrenceProjectionError, match="truncated"):
        project_recurrence_process_v1(
            process,
            _color_plan(process, truncated=True),
            _catalog(),
            layout="all-flow-union",
            normalization=_normalization(),
        )
    residual = project_recurrence_process_v1(
        process,
        _color_plan(process),
        _catalog(),
        layout="topology-replay",
        normalization=_normalization(),
    )
    assert residual.replay_partitions == ()


def test_projection_rejects_missing_source_semantics_and_malformed_references() -> None:
    process = _process()
    with pytest.raises(RecurrenceProjectionError, match="no supported source"):
        project_recurrence_process_v1(
            process,
            _color_plan(process),
            _catalog(omit_particle=-1),
            layout="all-flow-union",
            normalization=_normalization(),
        )
    with pytest.raises(RecurrenceProjectionError, match="unknown external label"):
        project_recurrence_process_v1(
            process,
            _color_plan(process, bad_label=True),
            _catalog(),
            layout="all-flow-union",
            normalization=_normalization(),
        )


def test_union_rejects_generation_selected_flow() -> None:
    process = _process()
    with pytest.raises(RecurrenceProjectionError, match="retain every public"):
        project_recurrence_process_v1(
            process,
            _color_plan(process),
            _catalog(),
            layout="all-flow-union",
            normalization=_normalization(),
            generation_slice=RecurrenceGenerationSliceV1(selected_public_flow_ids=(0,)),
        )


def test_projection_does_not_depend_on_generic_dag_types() -> None:
    import pyamplicol.generation.recurrence_projection as projection

    names = set(projection.__dict__)
    assert "GenericDAG" not in names
    assert "compile_generic_dag" not in names


def test_mismatched_color_plan_process_is_rejected() -> None:
    process = _process()
    other = replace(process, key="different_process")
    with pytest.raises(RecurrenceProjectionError, match="does not belong"):
        project_recurrence_process_v1(
            process,
            _color_plan(other),
            _catalog(),
            layout="all-flow-union",
            normalization=_normalization(),
        )


def test_projection_rejects_ambiguous_builder_axes() -> None:
    process = _process()
    common = dict(
        process=process,
        color_plan=_color_plan(process),
        template_catalog=_catalog(),
        layout="all-flow-union",
        normalization=_normalization(),
    )
    with pytest.raises(RecurrenceProjectionError, match="support mask"):
        project_recurrence_process_v1(**common, process_support_mask=1.0)
    with pytest.raises(RecurrenceProjectionError, match="unique"):
        project_recurrence_process_v1(
            **common,
            coupling_order_limits={"QCD": 4, "qcd": 4},
        )


def test_complex_parameters_use_authoritative_prepared_slots() -> None:
    base = _catalog()
    complex_parameter = ParameterTemplateV1(
        template_id="parameter:complex_coupling",
        name="complex_coupling",
        parameter_kind="external",
        value_type="complex",
        mutable=True,
        default_value=ExactComplexRationalV1(1, 10, 1, 20),
        exact_expression_digest=None,
        dependency_parameter_ids=(),
        prepared_parameter_id=17,
    )
    catalog = RecurrenceTemplateCatalog.create(
        compiled_model_digest=base.header.compiled_model_digest,
        prepared_kernel_pack_digest=base.header.prepared_kernel_pack_digest,
        parameters=(*base.parameters, complex_parameter),
        current_states=base.current_states,
        sources=base.sources,
        evaluator_bindings=base.evaluator_bindings,
    )
    process = _process()
    logical = project_recurrence_process_v1(
        process,
        _color_plan(process),
        catalog,
        layout="all-flow-union",
        normalization=_normalization(),
    )

    complex_rows = tuple(
        item
        for item in logical.parameter_projection
        if item.runtime_name == "complex_coupling"
    )
    assert tuple(item.component for item in complex_rows) == (0, 1)
    assert tuple(item.prepared_parameter_id for item in complex_rows) == (17, 17)
    assert tuple(item.runtime_slot for item in logical.parameter_projection) == tuple(
        range(len(logical.parameter_projection))
    )
