# SPDX-License-Identifier: 0BSD
"""Flavour-union discovery by all-outgoing content (``flavor_union_crossing``).

One template serves every crossing of its quark and gluon content: a member is
routed uncrossed where it can be (directly, else conjugated) and crossed
otherwise, with sigma from its sides and rho from its source states.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import replace
from fractions import Fraction
from typing import Any

import pytest

import pyamplicol.generation.service as generation_service
import pyamplicol.licensing as licensing
from pyamplicol import ProcessSet
from pyamplicol.api.errors import ConfigurationError
from pyamplicol.color.plan import build_color_plan, build_lc_topology_replay_plan
from pyamplicol.config import (
    Action,
    ColorConfig,
    EvaluatorConfig,
    GenerationConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.config.resolver import config_to_dict, config_to_toml, resolve_config
from pyamplicol.generation.dag_algorithms import infer_minimal_coupling_order_limits
from pyamplicol.generation.process_grouping import (
    FlavourUnion,
    GroupingProcess,
    GroupMember,
    ParticleTable,
    ProcessGroupMap,
    _legs,
    _mapping,
    crossing_signs,
    flavour_unions,
    is_union_quark_type,
)
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.generation.recurrence_columnar import (
    ExactComplexRationalV1,
    RecurrenceNormalizationV1,
)
from pyamplicol.generation.recurrence_fermion_pairing import (
    build_recurrence_fermion_pairing_catalog_v1,
)
from pyamplicol.generation.recurrence_projection import (
    project_external_source_legs,
    project_recurrence_process_v1,
)
from pyamplicol.generation.recurrence_schedule_sharing import (
    exact_recurrence_process_bijection,
    external_crossing_relation,
)
from pyamplicol.licensing import SymbolicaLicenseState
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.builtin.validation import generic_validation_point
from pyamplicol.models.prepared_catalog import build_prepared_kernel_catalog
from pyamplicol.models.recurrence_catalog_builder import (
    build_recurrence_template_catalog,
)


def _restricted_license(**_kwargs: object) -> SymbolicaLicenseState:
    return SymbolicaLicenseState(licensed=False, restricted=True)


def _run_config(*, crossing: bool, **process: Any) -> RunConfig:
    return RunConfig(
        action=Action.GENERATE,
        process=ProcessConfig(
            flavor_blind_strict=True,
            flavor_unions=True,
            flavor_union_crossing=crossing,
            flavor_scheme=process.pop("flavor_scheme", 4),
            max_coupling_orders={"QED": 0},
            **process,
        ),
        evaluator=EvaluatorConfig(execution_mode="recurrence"),
        generation=GenerationConfig(emit_api_bundle=False),
    )


_ENTRIES: dict[tuple[Any, ...], tuple[list[Any], ParticleTable, int]] = {}


def _entries(
    expressions: Sequence[str], *, crossing: bool = True
) -> tuple[list[Any], ParticleTable, int]:
    """Expanded entries as generation groups them, the particle table and the
    number of members whose crossing relation was refused."""

    key = (tuple(expressions), crossing)
    if key not in _ENTRIES:
        refused = 0
        original = generation_service.GenerationBackend._union_crossing_relation

        def counted(self: Any, *args: Any) -> Any:
            relation = original(self, *args)

            def wrapped(*call: Any) -> Any:
                nonlocal refused
                result = relation(*call)
                refused += result is None
                return result

            return wrapped

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(licensing, "detect_symbolica_license", _restricted_license)
            patch.setattr(
                generation_service.GenerationBackend,
                "_union_crossing_relation",
                counted,
            )
            backend = generation_service.GenerationBackend(
                _run_config(crossing=crossing), None
            )
            resolved = backend._resolve_model_for_plan(
                backend._configured_model_source()
            )
            entries = list(
                backend._expand_process_set(
                    ProcessSet.from_expressions(tuple(expressions)),
                    resolved,
                    PhaseHandle("test", None, 1),
                )
            )
            table = backend._grouping_particle_table(resolved)
        _ENTRIES[key] = (entries, table, refused)
    return _ENTRIES[key]


def _subprocesses(entries: Sequence[Any]) -> list[GroupingProcess]:
    processes = [generation_service._grouping_process(entry) for entry in entries]
    return processes + [m.process for e in entries for m in e.group_members]


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


def _check_crossed_union(
    entry: Any, table: ParticleTable, model: BuiltinSMModel
) -> None:
    """Every member: slot types by all-outgoing orientation, sigma from the
    sides, rho -1 exactly on crossed quark slots, the normalization factor."""

    template = generation_service._grouping_process(entry)
    count = template.initial_count
    norm = math.prod(
        model.external_state_normalization(template.pdgs[:count], template.pdgs[count:])
    )
    for member in entry.group_members:
        sigma = crossing_signs(member.permutation, count, member.initial_count)
        assert member.crossed == (-1 in sigma)
        assert not (member.crossed and member.conjugated)
        for slot, leg in enumerate(member.permutation):
            pdg = member.pdgs[leg]
            if member.conjugated and not table[pdg].type.self_conjugate:
                pdg = -pdg
            assert table[pdg].type == table[template.pdgs[slot]].type
            self_conjugate = table[pdg].type.self_conjugate
            outgoing = (
                -pdg if leg < member.initial_count and not self_conjugate else pdg
            )
            target = template.pdgs[slot]
            target = -target if slot < count and not self_conjugate else target
            if is_union_quark_type(table[pdg].type):
                assert (outgoing > 0) == (target > 0)
            else:
                assert outgoing == target
        if member.crossed:
            assert member.crossing_momentum_signs == sigma
            assert member.crossing_helicity_signs == tuple(
                -1 if s == -1 and member.pdgs[leg] != 21 else 1
                for s, leg in zip(sigma, member.permutation, strict=True)
            )
        else:
            assert member.crossing_helicity_signs is None
        initial, final = (
            member.pdgs[: member.initial_count],
            member.pdgs[member.initial_count :],
        )
        assert member.factor == Fraction(
            norm, math.prod(model.external_state_normalization(initial, final))
        )


# pp > n jets: programs with crossing (all-outgoing keys), of which unions
_PROGRAMS = {
    "p p > j j": (3, 2),
    "p p > j j j": (3, 2),
    "p p > j j j j": (4, 3),
    "p p > j j j j j": (4, 3),
}


@pytest.mark.parametrize("expression", sorted(_PROGRAMS))
def test_crossing_gives_one_program_per_all_outgoing_content(
    expression: str, model: BuiltinSMModel
) -> None:
    entries, table, refused = _entries((expression,))
    programs, unions = _PROGRAMS[expression]
    assert len(entries) == programs
    assert sum(entry.flavour_union_template for entry in entries) == unions
    assert refused == 0
    # every quark subprocess belongs to exactly one union; gluons stay alone
    seen: list[str] = []
    for entry in entries:
        if entry.flavour_union_template:
            seen += [entry.request.name, *(m.process_id for m in entry.group_members)]
            _check_crossed_union(entry, table, model)
        else:
            assert set(generation_service._grouping_process(entry).pdgs) == {21}
            assert not entry.group_members
    quark = [
        process
        for process in _subprocesses(entries)
        if any(is_union_quark_type(table[pdg].type) for pdg in process.pdgs)
    ]
    assert sorted(seen) == sorted(process.process_id for process in quark)
    assert any(m.crossed for e in entries for m in e.group_members)
    # schema 2 round trip of the crossed groups
    group_map = ProcessGroupMap()
    for entry in entries:
        if entry.flavour_union_template:
            group_map.add_group(
                [
                    GroupMember.identity(generation_service._grouping_process(entry)),
                    *entry.group_members,
                ],
                entry.flavour_union_pattern_pairs,
            )
    document = group_map.as_document()
    assert document["schema_version"] == 2
    assert ProcessGroupMap.from_document(document).as_document() == document


def test_crossed_member_factors_follow_the_external_normalization() -> None:
    entries, _, _ = _entries(("p p > j j j",))
    union = next(e for e in entries if e.request.expression == "d g > d g g")
    members = {m.expression: m for m in union.group_members}
    assert members["d d~ > g g g"].factor == Fraction(8, 9)
    assert members["d d~ > g g g"].crossed
    assert members["g g > d d~ g"].factor == Fraction(3, 4)
    assert members["u~ g > u~ g g"].factor == 1
    assert members["u~ g > u~ g g"].conjugated
    entries, _, _ = _entries(("p p > j j j j",))
    union = next(e for e in entries if e.request.expression == "d g > d g g g")
    members = {m.expression: m for m in union.group_members}
    assert members["d d~ > g g g g"].factor == Fraction(2, 3)
    assert members["g g > d d~ g g"].factor == Fraction(9, 8)


def test_uncrossed_members_keep_their_side_aware_routes() -> None:
    """Where both modes pick the same template, its uncrossed members get the
    same permutation, conjugation and patterns as without crossing."""

    compared = 0
    for expression in ("p p > j j j", "p p > j j j j"):
        crossed, _, _ = _entries((expression,), crossing=True)
        plain, _, _ = _entries((expression,), crossing=False)
        before = {
            e.request.name: {m.process_id: m for m in e.group_members}
            for e in plain
            if e.flavour_union_template
        }
        for entry in crossed:
            if entry.request.name not in before:
                continue
            for member in entry.group_members:
                if member.crossed:
                    continue
                old = before[entry.request.name][member.process_id]
                assert member == old
                compared += 1
    assert compared > 50


def test_shuffled_requests_give_identical_process_groups() -> None:
    entries, _, _ = _entries(("p p > j j",))
    expressions = sorted({p.expression for p in _subprocesses(entries)})

    def document(order: Sequence[str]) -> str:
        shuffled, _, refused = _entries(tuple(order))
        assert refused == 0
        group_map = ProcessGroupMap()
        for entry in sorted(shuffled, key=lambda e: e.request.name):
            group_map.add_group(
                [
                    GroupMember.identity(generation_service._grouping_process(entry)),
                    *sorted(entry.group_members, key=lambda m: m.process_id),
                ],
                entry.flavour_union_pattern_pairs,
            )
        return group_map.to_json()

    reference = document(expressions)
    assert '"crossing_momentum_signs"' in reference
    for seed in (1, 2):
        order = list(expressions)
        random.Random(seed).shuffle(order)
        assert document(order) == reference


def _processes(*expressions: str) -> list[GroupingProcess]:
    return [
        GroupingProcess(
            f"p{index}",
            expression,
            tuple(int(leg.pdg) for leg in generic_validation_point(expression)),
            len(expression.split(">")[0].split()),
        )
        for index, expression in enumerate(expressions)
    ]


def _relation(catalog: Any) -> Callable[..., Any]:
    """The crossing relation of same-flavour processes (no substitution)."""

    def relation(
        template: GroupingProcess, member: GroupingProcess, permutation: Any
    ) -> Any:
        legs = [
            project_external_source_legs(
                build_process_ir(process.expression, color_accuracy="lc"), catalog
            )
            for process in (template, member)
        ]
        return external_crossing_relation(*legs, permutation)

    return relation


def _pairing_rules(catalog: Any) -> Callable[[GroupingProcess], Any]:
    def rules(template: GroupingProcess) -> Any:
        process = build_process_ir(template.expression, color_accuracy="lc")
        return build_recurrence_fermion_pairing_catalog_v1(
            process, catalog.current_states, quantum_flows=catalog.quantum_flows
        ).rules

    return rules


@pytest.mark.parametrize("refuse_conjugation", [False, True])
def test_conjugation_is_preferred_and_crossing_replaces_a_refused_one(
    refuse_conjugation: bool, model: BuiltinSMModel, template_catalog: Any
) -> None:
    _, table, _ = _entries(("p p > j j",))
    processes = _processes("u g > u g", "u~ g > u~ g", "g g > u u~")
    unions = flavour_unions(
        processes,
        table,
        pairing_rules=_pairing_rules(template_catalog),
        normalization=model.external_state_normalization,
        union_warning=lambda _t, _f, conjugated: (
            "no C" if conjugated and refuse_conjugation else None
        ),
        crossing=True,
        crossing_relation=_relation(template_catalog),
    )
    (union,) = unions
    assert isinstance(union, FlavourUnion)
    assert union.template.expression == "u g > u g"
    members = {m.expression: m for m in union.members}
    anti = members["u~ g > u~ g"]
    if refuse_conjugation:
        # template slot 0 (initial u) takes the final u~, slot 2 the initial u~
        assert not anti.conjugated and anti.crossed
        assert anti.permutation == (2, 1, 0, 3)
        assert anti.crossing_momentum_signs == (-1, 1, -1, 1)
        assert anti.crossing_helicity_signs == (-1, 1, -1, 1)
    else:
        assert anti.conjugated and not anti.crossed
        assert anti.permutation == (0, 1, 2, 3)
    # slot 0 (initial u) takes the final u~, slot 3 (final g) an initial gluon
    gluons = members["g g > u u~"]
    assert gluons.permutation == (3, 0, 2, 1)
    assert gluons.crossing_momentum_signs == (-1, 1, 1, -1)
    assert gluons.crossing_helicity_signs == (-1, 1, 1, 1)


def test_a_refused_crossing_relation_keeps_the_member_routing(
    model: BuiltinSMModel, template_catalog: Any
) -> None:
    _, table, _ = _entries(("p p > j j",))
    processes = _processes("u g > u g", "u~ g > u~ g", "g g > u u~", "u u~ > g g")
    relation = _relation(template_catalog)
    unions = flavour_unions(
        processes,
        table,
        pairing_rules=_pairing_rules(template_catalog),
        normalization=model.external_state_normalization,
        crossing=True,
        crossing_relation=lambda t, m, p: (
            None if m.expression == "g g > u u~" else relation(t, m, p)
        ),
    )
    (union,) = unions
    assert {m.expression for m in union.members} == {
        "u g > u g",
        "u~ g > u~ g",
        "u u~ > g g",
    }
    with pytest.raises(ValueError, match="momentum signs"):
        flavour_unions(
            processes,
            table,
            pairing_rules=_pairing_rules(template_catalog),
            normalization=model.external_state_normalization,
            crossing=True,
            crossing_relation=lambda t, m, p: ((1,) * len(p), (1,) * len(p)),
        )
    with pytest.raises(ValueError, match="crossing relation"):
        flavour_unions(
            processes,
            table,
            pairing_rules=_pairing_rules(template_catalog),
            normalization=model.external_state_normalization,
            crossing=True,
        )


def test_crossing_keeps_processes_with_other_legs_side_aware(
    model: BuiltinSMModel, template_catalog: Any
) -> None:
    _, table, _ = _entries(("p p > j j",))
    # a photon makes the content ineligible: no crossing, the side-aware union
    processes = _processes(
        "d d~ > d d~ a", "d d~ > u u~ a", "d d > d d a", "u u~ > g a"
    )
    kwargs: dict[str, Any] = {
        "pairing_rules": _pairing_rules(template_catalog),
        "normalization": model.external_state_normalization,
    }
    plain = flavour_unions(processes, table, **kwargs)
    crossed = flavour_unions(
        processes,
        table,
        crossing=True,
        crossing_relation=_relation(template_catalog),
        **kwargs,
    )
    assert crossed == plain
    assert [union.template.expression for union in plain] == ["d d~ > d d~ a"]
    assert not any(m.crossed for union in crossed for m in union.members)


# (root, target) pairs that schedule sharing relates across a crossing in pp > 4j
@pytest.mark.parametrize(
    ("root", "target"),
    [
        ("d u > d u g g", "d d~ > u u~ g g"),
        ("d u > d u s s~", "d d~ > u s u~ s~"),
    ],
)
def test_crossing_signs_equal_the_schedule_sharing_remap(
    root: str, target: str, model: BuiltinSMModel, template_catalog: Any
) -> None:
    def logical(expression: str) -> Any:
        process = build_process_ir(expression, color_accuracy="lc")
        plan = build_color_plan(
            process,
            color_accuracy="lc",
            fold_trace_reflections=model.lc_trace_reflection_equivalence_is_proven(
                process
            ),
        )
        return project_recurrence_process_v1(
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

    remap = exact_recurrence_process_bijection(
        logical(root), logical(target), direct_executor_count=1, parameter_slot_count=1
    )
    assert remap is not None
    assert -1 in remap.source_momentum_signs
    relation = _relation(template_catalog)(
        _processes(root)[0], _processes(target)[0], remap.source_slots
    )
    assert relation == (remap.source_momentum_signs, remap.source_helicity_signs)
    assert remap.source_momentum_signs == (1, -1, -1, 1, 1, 1)
    assert remap.source_momentum_signs == crossing_signs(remap.source_slots, 2, 2)


def test_leg_mappings_still_refuse_side_changes() -> None:
    _, table, _ = _entries(("p p > j j",))
    (template,) = _processes("u g > u g")
    legs = _legs(template, table)
    assert _mapping(legs, legs, conjugated=False, method="x").permutation == (
        0,
        1,
        2,
        3,
    )
    # the same legs with the gluons on swapped sides
    swapped = [
        replace(leg, initial=not leg.initial) if leg.pdg == 21 else leg for leg in legs
    ]
    with pytest.raises(ValueError, match="same process shape"):
        _mapping(swapped, legs, conjugated=False, method="x")


def test_crossing_config_is_leading_colour_without_api_bundle() -> None:
    bundle_off = GenerationConfig(emit_api_bundle=False)
    recurrence = EvaluatorConfig(execution_mode="recurrence")
    process = ProcessConfig(
        flavor_blind_strict=True, flavor_unions=True, flavor_union_crossing=True
    )
    config = RunConfig(
        action=Action.GENERATE,
        process=process,
        evaluator=recurrence,
        generation=bundle_off,
    )
    with pytest.raises(ConfigurationError, match=r"requires process\.flavor_unions"):
        ProcessConfig(flavor_blind_strict=True, flavor_union_crossing=True)
    with pytest.raises(ConfigurationError, match="selected source"):
        ProcessConfig(
            flavor_blind_strict=True,
            flavor_unions=True,
            flavor_union_crossing=True,
            selected_source_helicities={"1": -1},
        )
    with pytest.raises(ConfigurationError, match="full-colour flavour-union crossing"):
        RunConfig(
            action=Action.GENERATE,
            process=process,
            evaluator=recurrence,
            generation=bundle_off,
            color=ColorConfig(accuracy="full"),
        )
    with pytest.raises(ConfigurationError, match="emit_api_bundle"):
        RunConfig(action=Action.GENERATE, process=process, evaluator=recurrence)
    # omitted from serialized configs at its default, written when on
    assert "flavor_union_crossing" not in config_to_toml(
        RunConfig(action=Action.GENERATE)
    )
    assert (
        "flavor_union_crossing"
        not in config_to_dict(RunConfig(action=Action.GENERATE))["process"]
    )
    assert "flavor_union_crossing = true" in config_to_toml(config)
    restored = resolve_config(config_to_dict(config)).effective
    assert restored.process.flavor_union_crossing


def test_crossing_refuses_generation_selected_helicities() -> None:
    selection = generation_service._ProcessSelection(selected_source_helicities={1: 1})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(licensing, "detect_symbolica_license", _restricted_license)
        backend = generation_service.GenerationBackend(
            _run_config(crossing=True), None, process_selection=selection
        )
        resolved = backend._resolve_model_for_plan(backend._configured_model_source())
        with pytest.raises(Exception, match="selected"):
            backend._expand_process_set(
                ProcessSet.from_expressions(("p p > j j",)),
                resolved,
                PhaseHandle("test", None, 1),
            )


def test_crossed_unions_that_absorb_an_aliased_entry_are_skipped() -> None:
    """All-outgoing unions are larger, so they meet aliased entries more often:
    such a union is skipped and every entry keeps its routing."""

    from types import SimpleNamespace

    from pyamplicol.api.requests import ProcessRequest
    from pyamplicol.generation.process_grouping import GroupingBasis
    from pyamplicol.generation.service import _apply_flavour_unions, _ExpandedProcess

    def entry(name: str, expression: str, aliases: tuple[Any, ...] = ()) -> Any:
        pdgs = tuple(int(leg.pdg) for leg in generic_validation_point(expression))
        return _ExpandedProcess(
            request=ProcessRequest.parse(expression, name=name),
            process_ir=SimpleNamespace(legs=[SimpleNamespace(pdg=p) for p in pdgs]),
            aliases=aliases,
        )

    template = GroupingProcess("ug", "u g > u g", (2, 21, 2, 21), 2)
    crossed = GroupMember(
        process_id="gg",
        expression="g g > u u~",
        pdgs=(21, 21, 2, -2),
        initial_count=2,
        permutation=(3, 0, 2, 1),
        basis=GroupingBasis.FLAVOUR_UNION,
        generated=False,
        patterns=(0,),
        pattern_signs=(1,),
        crossing_momentum_signs=(-1, 1, 1, -1),
        crossing_helicity_signs=(-1, 1, 1, 1),
    )
    union = FlavourUnion(
        template, (GroupMember.identity(template), crossed), 1, (((2, 0),),)
    )
    plain = [entry("ug", "u g > u g"), entry("gg", "g g > u u~")]
    (applied,) = _apply_flavour_unions(plain, [union], lambda t: None)
    assert applied.flavour_union_template
    assert [m.process_id for m in applied.group_members] == ["gg"]
    assert applied.group_members[0].crossed
    aliased = [entry("ug", "u g > u g"), entry("gg", "g g > u u~", ({"id": "a"},))]
    assert _apply_flavour_unions(aliased, [union], lambda t: None) == aliased
