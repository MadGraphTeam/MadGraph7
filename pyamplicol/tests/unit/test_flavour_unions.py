# SPDX-License-Identifier: 0BSD
"""Flavour-union discovery: templates, member slots, pairing patterns, factors."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import fields
from fractions import Fraction
from typing import Any

import pytest

import pyamplicol.generation.service as generation_service
import pyamplicol.licensing as licensing
from pyamplicol import ProcessSet
from pyamplicol.api.errors import ArtifactError, ConfigurationError
from pyamplicol.artifacts.process_groups import validate_process_groups
from pyamplicol.color.plan import build_color_plan, build_lc_topology_replay_plan
from pyamplicol.config import (
    Action,
    ColorConfig,
    EvaluatorConfig,
    GenerationConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.generation.born_orders import RuntimeNormalizationRequest
from pyamplicol.generation.dag_algorithms import infer_minimal_coupling_order_limits
from pyamplicol.generation.process_grouping import (
    FlavourUnion,
    GroupingBasis,
    GroupingProcess,
    GroupMember,
    ParticleTable,
    ProcessGroupMap,
    flavor_class_key,
    flavour_unions,
    is_union_quark_type,
)
from pyamplicol.generation.process_grouping_symmetry import VertexSymmetryTable
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.generation.recurrence_columnar import (
    ExactComplexRationalV1,
    RecurrenceNormalizationV1,
)
from pyamplicol.generation.recurrence_fermion_pairing import (
    build_recurrence_fermion_pairing_catalog_v1,
)
from pyamplicol.generation.recurrence_projection import project_recurrence_process_v1
from pyamplicol.licensing import SymbolicaLicenseState
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.prepared_catalog import build_prepared_kernel_catalog
from pyamplicol.models.recurrence_catalog_builder import (
    build_recurrence_template_catalog,
)

QCD_ONLY = {"QED": 0}


def _restricted_license(**_kwargs: object) -> SymbolicaLicenseState:
    return SymbolicaLicenseState(licensed=False, restricted=True)


@pytest.fixture(scope="module")
def model() -> BuiltinSMModel:
    return BuiltinSMModel()


@pytest.fixture(scope="module")
def template_catalog(model: BuiltinSMModel) -> Any:
    return build_recurrence_template_catalog(
        model,
        build_prepared_kernel_catalog(model),
        compiled_model_digest="a" * 64,
        prepared_kernel_pack_digest="b" * 64,
    )


@pytest.fixture(scope="module")
def pairing_rules(template_catalog: Any) -> Callable[[GroupingProcess], Any]:
    # The call recurrence_projection.project_recurrence_process_v1 makes.
    def rules(template: GroupingProcess) -> Any:
        process = build_process_ir(template.expression, color_accuracy="lc")
        assert tuple(leg.pdg for leg in process.legs) == template.pdgs
        return build_recurrence_fermion_pairing_catalog_v1(
            process,
            template_catalog.current_states,
            quantum_flows=template_catalog.quantum_flows,
        ).rules

    return rules


@pytest.fixture(scope="module")
def symmetry(model: BuiltinSMModel) -> VertexSymmetryTable:
    return VertexSymmetryTable.from_builtin_model(model)


_EXPANSIONS: dict[
    tuple[str, int], tuple[list[GroupingProcess], int, ParticleTable]
] = {}


def _expansion(
    expression: str, flavor_scheme: int
) -> tuple[list[GroupingProcess], int, ParticleTable]:
    """Every concrete subprocess of a strict flavour-blind expansion, the
    number of flavour classes (generated entries) and the particle table."""

    key = (expression, flavor_scheme)
    if key not in _EXPANSIONS:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(licensing, "detect_symbolica_license", _restricted_license)
            backend = generation_service.GenerationBackend(
                RunConfig(
                    action=Action.GENERATE,
                    process=ProcessConfig(
                        flavor_blind_strict=True, flavor_scheme=flavor_scheme
                    ),
                    evaluator=EvaluatorConfig(execution_mode="compiled"),
                ),
                None,
            )
            resolved = backend._resolve_model_for_plan(
                backend._configured_model_source()
            )
            entries = backend._expand_process_set(
                ProcessSet.from_expressions((expression,)),
                resolved,
                PhaseHandle("test", None, 1),
            )
            table = backend._grouping_particle_table(resolved)
        processes = [generation_service._grouping_process(entry) for entry in entries]
        processes += [
            member.process for entry in entries for member in entry.group_members
        ]
        _EXPANSIONS[key] = (processes, len(entries), table)
    return _EXPANSIONS[key]


def _unions(
    processes: Sequence[GroupingProcess],
    table: ParticleTable,
    model: BuiltinSMModel,
    pairing_rules: Callable[[GroupingProcess], Any],
    symmetry: VertexSymmetryTable,
    *,
    limits: dict[str, int] = QCD_ONLY,
    **options: Any,
) -> list[FlavourUnion]:
    return flavour_unions(
        processes,
        table,
        pairing_rules=pairing_rules,
        normalization=model.external_state_normalization,
        union_warning=lambda template, flavours, conjugated: (
            symmetry.flavour_union_warning(
                template, flavours, table, limits=limits, conjugated=conjugated
            )
        ),
        **options,
    )


def _by_expression(union: FlavourUnion) -> dict[str, GroupMember]:
    return {member.expression: member for member in union.members}


# pp > n jets with QED=0: (unions, flavour classes, classes joined by unions)
_STATISTICS = {
    ("p p > j j", 4): (2, 9, 5),
    ("p p > j j", 5): (2, 9, 5),
    ("p p > j j j", 4): (3, 11, 7),
    ("p p > j j j", 5): (3, 11, 7),
    ("p p > j j j j", 4): (6, 23, 19),
    ("p p > j j j j", 5): (6, 23, 19),
}


@pytest.mark.parametrize(("expression", "flavor_scheme"), sorted(_STATISTICS))
def test_jet_unions_join_the_quark_flavour_classes(
    expression: str,
    flavor_scheme: int,
    model: BuiltinSMModel,
    pairing_rules: Callable[[GroupingProcess], Any],
    symmetry: VertexSymmetryTable,
) -> None:
    processes, class_count, table = _expansion(expression, flavor_scheme)
    unions = _unions(processes, table, model, pairing_rules, symmetry)
    by_id = {process.process_id: process for process in processes}

    members = [member for union in unions for member in union.members]
    assert len({member.process_id for member in members}) == len(members)
    joined = {
        flavor_class_key(
            by_id[member.process_id], table, consider_charge_conjugation=True
        )
        for member in members
    }
    assert (len(unions), class_count, len(joined)) == _STATISTICS[
        (expression, flavor_scheme)
    ]
    for union in unions:
        _check_union(union, by_id, table, model)


def _check_union(
    union: FlavourUnion,
    by_id: dict[str, GroupingProcess],
    table: ParticleTable,
    model: BuiltinSMModel,
) -> None:
    template = union.template
    count = template.initial_count
    union_slots = [is_union_quark_type(table[pdg].type) for pdg in template.pdgs]
    flavours = {
        abs(pdg) for pdg, union in zip(template.pdgs, union_slots, strict=True) if union
    }
    assert len(flavours) == 1, "the template carries one union flavour"
    assert template.process_id in by_id
    assert by_id[template.process_id].pdgs == template.pdgs
    classes = {
        flavor_class_key(
            by_id[member.process_id], table, consider_charge_conjugation=True
        )
        for member in union.members
    }
    assert len(classes) >= 2
    template_norm = math.prod(
        model.external_state_normalization(template.pdgs[:count], template.pdgs[count:])
    )
    for member in union.members:
        assert member.basis is GroupingBasis.FLAVOUR_UNION
        assert not member.generated
        source = by_id[member.process_id]
        assert (member.expression, member.pdgs) == (source.expression, source.pdgs)
        assert sorted(member.permutation) == list(range(len(template.pdgs)))
        # Slot r of the template takes member leg permutation[r].
        for slot, leg in enumerate(member.permutation):
            pdg = member.pdgs[leg]
            if member.conjugated and not table[pdg].type.self_conjugate:
                pdg = -pdg
            assert (slot < count) == (leg < member.initial_count)
            assert table[pdg].type == table[template.pdgs[slot]].type
            if union_slots[slot]:
                assert (pdg > 0) == (template.pdgs[slot] > 0)
            else:
                assert pdg == template.pdgs[slot]
        # Flavour-respecting pairings: prod_f n_f! of them.
        flavour = [abs(member.pdgs[leg]) for leg in member.permutation]
        fundamental = Counter(
            flavour[slot]
            for slot, pdg in enumerate(template.pdgs)
            if union_slots[slot] and (pdg > 0) != (slot < count)
        )
        assert member.patterns is not None and member.pattern_signs is not None
        assert len(member.patterns) == math.prod(
            math.factorial(n) for n in fundamental.values()
        )
        assert member.pattern_signs == (1,) * len(member.patterns)
        assert all(0 <= pattern < union.pattern_count for pattern in member.patterns)
        expected = Fraction(
            template_norm,
            math.prod(
                model.external_state_normalization(
                    member.pdgs[:count], member.pdgs[count:]
                )
            ),
        )
        assert member.factor == expected
        if member.process_id == template.process_id:
            assert member.patterns == tuple(range(union.pattern_count))
            assert member.permutation == tuple(range(len(template.pdgs)))
            assert member.factor == 1 and not member.conjugated


def test_dijet_member_patterns(
    model: BuiltinSMModel,
    pairing_rules: Callable[[GroupingProcess], Any],
    symmetry: VertexSymmetryTable,
) -> None:
    processes, _, table = _expansion("p p > j j", 4)
    unions = {
        union.template.expression: union
        for union in _unions(processes, table, model, pairing_rules, symmetry)
    }
    assert set(unions) == {"d d > d d", "d d~ > d d~"}

    # Template rules: 0 = (2,0)(3,1), 1 = (2,1)(3,0).
    same = _by_expression(unions["d d > d d"])
    assert unions["d d > d d"].pattern_count == 2
    assert same["u u > u u"].patterns == (0, 1)
    assert same["u u > u u"].factor == 1
    assert same["d u > d u"].patterns == (0,)
    assert same["d u > d u"].factor == 2  # 2! identical d in the template
    assert same["d~ u~ > d~ u~"].conjugated
    assert same["d~ u~ > d~ u~"].patterns == (0,)
    assert len(same) == 20

    # Template rules: 0 = (1,0)(2,3) annihilation, 1 = (1,3)(2,0) scattering.
    mixed = _by_expression(unions["d d~ > d d~"])
    assert mixed["d d~ > d d~"].patterns == (0, 1)
    assert mixed["u u~ > s s~"].patterns == (0,)
    assert mixed["d u~ > d u~"].patterns == (1,)
    assert mixed["u d~ > u d~"].patterns == (1,)
    assert not any(member.conjugated for member in mixed.values())
    assert all(member.factor == 1 for member in mixed.values())


def test_three_line_member_patterns(
    model: BuiltinSMModel,
    pairing_rules: Callable[[GroupingProcess], Any],
    symmetry: VertexSymmetryTable,
) -> None:
    processes, _, table = _expansion("p p > j j j j", 4)
    union = next(
        union
        for union in _unions(processes, table, model, pairing_rules, symmetry)
        if union.template.expression == "d d~ > d d d~ d~"
    )
    members = _by_expression(union)
    assert union.pattern_count == 6
    # Rule 0 = (1,0)(2,4)(3,5); 4 = (1,5)(2,0)(3,4); 5 = (1,5)(2,4)(3,0).
    assert members["d d~ > u s u~ s~"].patterns == (0,)
    assert members["d d~ > u s u~ s~"].factor == 4
    assert members["d u~ > d d d~ u~"].patterns == (4, 5)
    assert members["d u~ > d d d~ u~"].factor == 2
    assert members["u u~ > u u u~ u~"].patterns == tuple(range(6))
    assert members["u u~ > u u u~ u~"].factor == 1


def test_member_slots_follow_the_template_orientation(
    model: BuiltinSMModel,
    pairing_rules: Callable[[GroupingProcess], Any],
    symmetry: VertexSymmetryTable,
) -> None:
    _, _, table = _expansion("p p > j j", 4)
    processes = [
        GroupingProcess("dd", "d d~ > d d~", (1, -1, 1, -1), 2),
        GroupingProcess("uu", "u~ u > d d~", (-2, 2, 1, -1), 2),
        GroupingProcess("ud", "u d > d u", (2, 1, 1, 2), 2),
        GroupingProcess("dd2", "d d > d d", (1, 1, 1, 1), 2),
    ]
    unions = {
        union.template.process_id: _by_expression(union)
        for union in _unions(processes, table, model, pairing_rules, symmetry)
    }
    assert set(unions) == {"dd", "dd2"}
    swapped = unions["dd"]["u~ u > d d~"]
    assert swapped.permutation == (1, 0, 2, 3)
    assert swapped.patterns == (0,)
    # A side permutation of "d u > d u" pairs the other way round.
    assert unions["dd2"]["u d > d u"].permutation == (0, 1, 2, 3)
    assert unions["dd2"]["u d > d u"].patterns == (1,)


def test_template_rule_ids_match_the_recurrence_projection(
    model: BuiltinSMModel,
    template_catalog: Any,
    pairing_rules: Callable[[GroupingProcess], Any],
    symmetry: VertexSymmetryTable,
) -> None:
    processes, _, table = _expansion("p p > j j j j", 4)
    unions = _unions(processes, table, model, pairing_rules, symmetry)
    assert len(unions) == 6
    for union in unions:
        process = build_process_ir(union.template.expression, color_accuracy="lc")
        plan = build_color_plan(
            process,
            color_accuracy="lc",
            fold_trace_reflections=model.lc_trace_reflection_equivalence_is_proven(
                process
            ),
        )
        logical = project_recurrence_process_v1(
            process,
            plan,
            template_catalog,
            layout="topology-replay",
            topology_replay=build_lc_topology_replay_plan(plan, model),
            normalization=RecurrenceNormalizationV1(
                ExactComplexRationalV1(1), "test-v1", "4" * 64
            ),
            coupling_order_limits=infer_minimal_coupling_order_limits(
                process, model=model
            ),
            model=model,
        )
        projected = [
            (rule.rule_id, rule.endpoint_pairings)
            for rule in logical.fermion_pairing_catalog.rules
        ]
        used = [
            (rule.rule_id, rule.endpoint_pairings)
            for rule in pairing_rules(union.template)
        ]
        assert projected == used
        assert len(projected) == union.pattern_count


@pytest.mark.parametrize("limits", [{}, {"QED": 2}])
def test_electroweak_couplings_prevent_unions(
    limits: dict[str, int],
    model: BuiltinSMModel,
    pairing_rules: Callable[[GroupingProcess], Any],
    symmetry: VertexSymmetryTable,
) -> None:
    processes, _, table = _expansion("p p > j j", 4)
    assert not _unions(processes, table, model, pairing_rules, symmetry, limits=limits)
    template = GroupingProcess("t", "d d > d d", (1, 1, 1, 1), 2)
    warning = symmetry.flavour_union_warning(
        template, frozenset({1, 2}), table, limits=limits
    )
    assert warning is not None
    assert "changes a union quark flavour" in warning  # the W vertex
    assert "relabelling" in warning  # photon and Z couplings
    assert (
        symmetry.flavour_union_warning(
            template, frozenset({1, 2, 3, 4}), table, limits=QCD_ONLY, conjugated=True
        )
        is None
    )


def test_refused_conjugation_splits_a_union_by_orientation(
    model: BuiltinSMModel,
    pairing_rules: Callable[[GroupingProcess], Any],
) -> None:
    processes, _, table = _expansion("p p > j j", 4)
    unions = flavour_unions(
        processes,
        table,
        pairing_rules=pairing_rules,
        normalization=model.external_state_normalization,
        union_warning=lambda _t, _f, conjugated: "no C" if conjugated else None,
    )
    templates = sorted(union.template.expression for union in unions)
    assert templates == ["d d > d d", "d d~ > d d~", "d~ d~ > d~ d~"]
    assert not any(member.conjugated for union in unions for member in union.members)


def test_without_conjugation_antiquark_unions_stand_alone(
    model: BuiltinSMModel,
    pairing_rules: Callable[[GroupingProcess], Any],
    symmetry: VertexSymmetryTable,
) -> None:
    processes, _, table = _expansion("p p > j j", 4)
    unions = _unions(
        processes,
        table,
        model,
        pairing_rules,
        symmetry,
        consider_charge_conjugation=False,
    )
    templates = sorted(union.template.expression for union in unions)
    assert templates == ["d d > d d", "d d~ > d d~", "d~ d~ > d~ d~"]


def test_union_members_round_trip_through_process_groups(
    model: BuiltinSMModel,
    pairing_rules: Callable[[GroupingProcess], Any],
    symmetry: VertexSymmetryTable,
) -> None:
    processes, _, table = _expansion("p p > j j", 4)
    union = next(
        union
        for union in _unions(processes, table, model, pairing_rules, symmetry)
        if union.template.expression == "d d > d d"
    )
    group = union.group()
    assert group[0] == GroupMember.identity(union.template)
    assert group[0].patterns is None and group[0].generated
    assert len(group) == len(union.members)  # the template replaces its member

    group_map = ProcessGroupMap()
    group_map.add_group(group)
    restored = ProcessGroupMap.from_json(group_map.to_json())
    assert list(restored.members()) == group
    validate_process_groups(restored, [union.template.process_id])

    member = group[1]
    record = member.as_dict()
    assert record["flavour_union_patterns"] == list(member.patterns or ())
    assert record["flavour_union_pattern_signs"] == [1] * len(member.patterns or ())
    assert GroupMember.from_dict(record) == member
    # Plain members keep their exact records.
    plain = GroupMember.identity(union.template).as_dict()
    assert "flavour_union_patterns" not in plain
    assert "flavour_union_pattern_signs" not in plain


def test_invalid_union_members_are_rejected(
    model: BuiltinSMModel,
    pairing_rules: Callable[[GroupingProcess], Any],
    symmetry: VertexSymmetryTable,
) -> None:
    processes, _, table = _expansion("p p > j j", 4)
    union = next(
        union
        for union in _unions(processes, table, model, pairing_rules, symmetry)
        if union.template.expression == "d d > d d"
    )
    template, member = union.group()[:2]
    with pytest.raises(ValueError, match="one sign"):
        GroupMember(**{**_fields(member), "pattern_signs": (1, 1, 1)})
    with pytest.raises(ValueError, match="given together"):
        GroupMember(**{**_fields(member), "pattern_signs": None})
    with pytest.raises(ValueError, match="other than"):
        GroupMember(**{**_fields(member), "pattern_signs": (2,)})
    with pytest.raises(ValueError, match="do not compose"):
        member.through(template)
    with pytest.raises(ValueError, match="not invertible"):
        member.inverted_to(template)

    wrong_basis = GroupMember(
        **{**_fields(member), "basis": GroupingBasis.FLAVOR_BLIND}
    )
    for bad in (wrong_basis, GroupMember(**{**_fields(member), "generated": True})):
        group_map = ProcessGroupMap()
        group_map.add_group([template, bad])
        with pytest.raises(ArtifactError):
            validate_process_groups(
                group_map,
                [template.process_id] + ([bad.process_id] if bad.generated else []),
            )


def _fields(member: GroupMember) -> dict[str, Any]:
    return {field.name: getattr(member, field.name) for field in fields(member)}


def test_external_state_normalization_is_the_runtime_normalization(
    model: BuiltinSMModel,
) -> None:
    for expression, orders in (
        ("d d > d d", 2),
        ("u d~ > u u d~ u~", 4),
        ("g g > d d d~ d~", 4),
    ):
        process = build_process_ir(expression, color_accuracy="lc")
        payload = model.runtime_normalization_payload(
            RuntimeNormalizationRequest(process, lambda o=orders: [(("QCD", o),)])
        )
        assert model.external_state_normalization(
            process.initial_pdgs, process.final_pdgs
        ) == (payload["average_factor"], payload["identical_factor"])


def test_flavour_unions_config_requires_strict_lc_recurrence() -> None:
    with pytest.raises(ConfigurationError, match="flavor_blind_strict"):
        ProcessConfig(flavor_unions=True)
    process = ProcessConfig(flavor_blind_strict=True, flavor_unions=True)
    no_bundle = GenerationConfig(emit_api_bundle=False)
    recurrence = EvaluatorConfig(execution_mode="recurrence")
    RunConfig(
        action=Action.GENERATE,
        process=process,
        evaluator=recurrence,
        generation=no_bundle,
    )
    for evaluator, color, generation in (
        (EvaluatorConfig(execution_mode="compiled"), ColorConfig(), no_bundle),
        (recurrence, ColorConfig(accuracy="full"), no_bundle),
        (recurrence, ColorConfig(lc_flow_layout="all-flow-union"), no_bundle),
        (recurrence, ColorConfig(), GenerationConfig()),
    ):
        with pytest.raises(ConfigurationError, match="flavor_unions"):
            RunConfig(
                action=Action.GENERATE,
                process=process,
                evaluator=evaluator,
                color=color,
                generation=generation,
            )
