# SPDX-License-Identifier: 0BSD
"""Exact all-flow-union execution over synthetic Direct-Arena v2 tables."""

from __future__ import annotations

from dataclasses import astuple, replace
from decimal import Decimal

import pytest

from pyamplicol.api.errors import ArtifactError, EvaluationError
from pyamplicol.runtime.recurrence_exact import _executor as executor_module
from pyamplicol.runtime.recurrence_exact._execution import (
    _evaluate_union_point,
    _execute_union_source,
    _kernel_inputs,
)
from pyamplicol.runtime.recurrence_exact._executor import RecurrenceExactExecutor
from pyamplicol.runtime.recurrence_exact._plan import (
    _RecurrenceExactPlan,
    _SourceTemplate,
)
from pyamplicol.runtime.recurrence_exact._plan_v2 import (
    DIRECT_NONE_U32,
    RECURRENCE_EXACT_SECTIONS_ABI,
    RECURRENCE_RUNTIME_LAYOUT_V2_ABI,
    _AmplitudeDestination,
    _Closure,
    _Contribution,
    _ExactFactor,
    _Executor,
    _MomentumForm,
    _MomentumTerm,
    _parse_exact_sections,
    _RecurrenceExactSectionsV1,
    _ResolvedHelicity,
    _ResolvedSourceSelection,
    _RowGroup,
    _Source,
    _SourceDispatchVariant,
    _SourceEmbedding,
    _SourceProjection,
    _SourceStateAssignment,
)
from pyamplicol.runtime.symbolica_exact import _quark_weyl

_ZERO = Decimal(0)
_ONE = Decimal(1)


def _factor(real: int, imaginary: int = 0) -> _ExactFactor:
    return _ExactFactor(real, 1, imaginary, 1)


def _scalar_union_plan() -> _RecurrenceExactPlan:
    exact_factors = (
        _factor(1),
        _factor(0),
        _factor(2),
        _factor(3),
        _factor(0),
        _factor(1),
        _factor(0),
        _factor(2),
    )
    sources = (
        _Source(0, 0, 0, 0, 0, 0, 0),
        _Source(1, 2, 1, 1, 0, 0, 0),
    )
    variants = (
        _SourceDispatchVariant(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1),
        _SourceDispatchVariant(2, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 2, 1),
    )
    sections = _RecurrenceExactSectionsV1(
        process_id="synthetic_union",
        strategy="all-flow-union",
        semantic_digest="0" * 64,
        runtime_layout_digest="1" * 64,
        current_arena_components=4,
        amplitude_destination_count=2,
        parameter_value_count=0,
        external_source_count=2,
        currents=(),
        sources=sources,
        contributions=(),
        finalizations=(),
        closures=(
            _Closure(0, 2, 0, 1, 0, 0, 4, 2, 0, 0),
            _Closure(0, 2, 0, 1, 1, 0, 6, 2, 0, 0),
        ),
        row_groups=(
            _RowGroup(0, 0, 0, DIRECT_NONE_U32, 0, 2),
            _RowGroup(1, 3, 3, 1, 0, 2),
        ),
        momentum_forms=(_MomentumForm(0, 1), _MomentumForm(1, 1)),
        momentum_terms=(_MomentumTerm(0, 1), _MomentumTerm(1, 1)),
        replay_targets=(),
        source_permutations=(),
        amplitude_destinations=(
            _AmplitudeDestination(0, 0, 10, DIRECT_NONE_U32, 1, 0),
            _AmplitudeDestination(1, 1, 20, DIRECT_NONE_U32, 1, 0),
        ),
        resolved_helicities=(
            _ResolvedHelicity(0, 0, 0, 0, 2, 2, 2, 0),
        ),
        source_state_assignments=(
            _SourceStateAssignment(0, 0),
            _SourceStateAssignment(1, 0),
        ),
        source_dispatch_variants=variants,
        source_embeddings=(
            _SourceEmbedding(0, DIRECT_NONE_U32, 1),
            _SourceEmbedding(1, 0, 2),
            _SourceEmbedding(0, DIRECT_NONE_U32, 1),
            _SourceEmbedding(1, 0, 3),
        ),
        source_projections=(
            _SourceProjection(0, 1),
            _SourceProjection(0, 1),
        ),
        resolved_source_selections=(
            _ResolvedSourceSelection(0, 0),
            _ResolvedSourceSelection(1, 1),
        ),
        public_helicities=(0, 0),
        exact_factors=exact_factors,
        public_flow_ids=(10, 20),
        executors=(
            _Executor(0, "source", "initialize", (), 1, 1, None, "source"),
            _Executor(
                1,
                "closure",
                "closure-add",
                (2, 2),
                1,
                2,
                None,
                "componentwise-dot",
            ),
        ),
    )
    return _RecurrenceExactPlan(
        sections=sections,
        kernels={},
        executors={row.executor_id: row for row in sections.executors},
        source_templates={
            0: _SourceTemplate(
                0, 1, 0, 0, 0, "scalar", "self-conjugate", None, 1, 1, 1
            ),
            1: _SourceTemplate(
                1, 1, 0, 0, 0, "scalar", "self-conjugate", None, 1, 1, 1
            ),
        },
        initial_source_slots=frozenset(),
        executor_couplings={},
        prepared_defaults=(),
        parameter_projection=(),
        parameter_derivation=None,
    )


def _point() -> tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...]:
    return (
        (Decimal(10), Decimal(1), Decimal(2), Decimal(3)),
        (Decimal(20), Decimal(4), Decimal(5), Decimal(6)),
    )


def test_union_schedule_returns_independent_flow_amplitudes() -> None:
    plan = _scalar_union_plan()
    amplitudes = _evaluate_union_point(
        plan,
        _point(),
        plan.sections.resolved_helicities[0],
        (),
        50,
    )
    assert amplitudes == ((Decimal(6), _ZERO), (Decimal(12), _ZERO))


def test_union_source_zeroes_inactive_weyl_embedding_components() -> None:
    plan = _scalar_union_plan()
    sections = plan.sections
    source = _Source(0, 0, 0, 0, 1, 0, 0)
    variant = _SourceDispatchVariant(
        0,
        0,
        0,
        0,
        0,
        0,
        7,
        7,
        9,
        1,
        0,
        0,
        4,
        2,
    )
    plan.sections = replace(
        sections,
        sources=(source,),
        source_dispatch_variants=(variant,),
        source_embeddings=(
            _SourceEmbedding(0, 0, 0),
            _SourceEmbedding(1, DIRECT_NONE_U32, 1),
            _SourceEmbedding(2, 1, 0),
            _SourceEmbedding(3, DIRECT_NONE_U32, 1),
        ),
        source_projections=(
            _SourceProjection(0, 0),
            _SourceProjection(1, 2),
        ),
    )
    plan.source_templates = {
        7: _SourceTemplate(
            7,
            2,
            1,
            1,
            1,
            "fermion",
            "particle",
            None,
            1,
            1,
            1,
        )
    }
    momentum = (Decimal(10), Decimal(1), Decimal(2), Decimal(9))
    arena = [(_ONE * 99, _ONE * 99) for _ in range(4)]
    _execute_union_source(plan, source, variant, (momentum,), (), arena)
    expected = _quark_weyl(momentum, 1, 1)
    assert arena == [expected[0], (_ZERO, _ZERO), expected[1], (_ZERO, _ZERO)]


def test_exact_kernel_inputs_use_executor_binding_coupling() -> None:
    plan = _scalar_union_plan()
    plan.executor_couplings = {7: (Decimal("1.25"), Decimal("-0.5"))}
    executor = _Executor(
        7,
        "contribution",
        "add",
        (1, 1),
        1,
        2,
        3,
        None,
    )
    row = _Contribution(0, 0, 0, 0, 0, 0, 0, 0)

    inputs = _kernel_inputs(
        plan,
        (
            {"component": 0, "role": "coupling-real"},
            {"component": 1, "role": "coupling-imag"},
        ),
        executor,
        row,
        (),
        ((Decimal(99), Decimal(101)),),
        (),
    )

    assert inputs == (
        (Decimal("1.25"), _ZERO),
        (Decimal("-0.5"), _ZERO),
    )


def test_union_exact_resolved_values_sum_incoherently_by_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _scalar_union_plan()
    executor = object.__new__(RecurrenceExactExecutor)
    executor._plan = plan
    executor._physics = {
        "helicities": [
            {
                "id": "h:0,0",
                "values": [0, 0],
                "computed": True,
                "structural_zero": False,
                "coefficient": 1,
            }
        ],
        "color_components": [
            {"id": "flow:a", "coefficient": 1},
            {"id": "flow:b", "coefficient": 1},
        ],
        "color_accuracy": "lc",
    }
    executor._permutation = None
    executor._native_runtime = object()
    executor._replay_by_color = ()
    executor._destination_helicities = ()
    executor._union_destination_by_color = executor._union_destinations_by_color()
    executor._union_helicity_by_physics = executor._union_helicities_by_physics()
    monkeypatch.setattr(executor_module, "_prepare_points", lambda *_: (_point(),))
    monkeypatch.setattr(
        executor_module,
        "_runtime_state",
        lambda _: {"model_parameter_values": [], "normalization_factor": 1},
    )

    resolved = executor.evaluate_resolved(
        (((10, 1, 2, 3), (20, 4, 5, 6)),),
        helicities=None,
        color_flows=None,
        precision=40,
    )
    assert resolved.values == (((Decimal(36), Decimal(144)),),)
    assert resolved.total() == (Decimal(180),)
    assert resolved.total() != (Decimal((6 + 12) ** 2),)

    selected = executor.evaluate_resolved(
        (((10, 1, 2, 3), (20, 4, 5, 6)),),
        helicities=("h:0,0",),
        color_flows=("flow:b",),
        precision=40,
    )
    assert selected.values == (((Decimal(144),),),)
    with pytest.raises(EvaluationError, match="unknown resolved helicity ID"):
        executor.evaluate_resolved(
            (((10, 1, 2, 3), (20, 4, 5, 6)),),
            helicities=("missing",),
            color_flows=None,
            precision=40,
        )
    with pytest.raises(EvaluationError, match="unknown resolved color component ID"):
        executor.evaluate_resolved(
            (((10, 1, 2, 3), (20, 4, 5, 6)),),
            helicities=None,
            color_flows=("missing",),
            precision=40,
        )


def test_union_native_section_adapter_parses_dispatch_tables() -> None:
    sections = _scalar_union_plan().sections
    raw = {
        "abi": RECURRENCE_EXACT_SECTIONS_ABI,
        "runtime_layout_abi": RECURRENCE_RUNTIME_LAYOUT_V2_ABI,
        "process_id": sections.process_id,
        "strategy": sections.strategy,
        "semantic_digest": sections.semantic_digest,
        "runtime_layout_digest": sections.runtime_layout_digest,
        "counts": (
            sections.current_arena_components,
            sections.amplitude_destination_count,
            sections.parameter_value_count,
            sections.external_source_count,
        ),
        "currents": [astuple(row) for row in sections.currents],
        "sources": [astuple(row) for row in sections.sources],
        "contributions": [astuple(row) for row in sections.contributions],
        "finalizations": [astuple(row) for row in sections.finalizations],
        "closures": [astuple(row) for row in sections.closures],
        "row_groups": [astuple(row) for row in sections.row_groups],
        "momentum_forms": [astuple(row) for row in sections.momentum_forms],
        "momentum_terms": [astuple(row) for row in sections.momentum_terms],
        "replay_targets": [],
        "source_permutations": [],
        "amplitude_destinations": [
            astuple(row) for row in sections.amplitude_destinations
        ],
        "resolved_helicities": [
            astuple(row) for row in sections.resolved_helicities
        ],
        "source_state_assignments": [
            astuple(row) for row in sections.source_state_assignments
        ],
        "source_dispatch_variants": [
            astuple(row) for row in sections.source_dispatch_variants
        ],
        "source_embeddings": [astuple(row) for row in sections.source_embeddings],
        "source_projections": [astuple(row) for row in sections.source_projections],
        "resolved_source_selections": [
            astuple(row) for row in sections.resolved_source_selections
        ],
        "public_helicities": list(sections.public_helicities),
        "exact_factors": [
            tuple(str(value) for value in astuple(row))
            for row in sections.exact_factors
        ],
        "public_flow_ids": list(sections.public_flow_ids),
        "executors": [
            (
                row.executor_id,
                row.role,
                row.destination_operation,
                row.parent_component_counts,
                row.destination_component_count,
                row.momentum_operand_count,
                row.prepared_kernel_id,
                row.runtime_template,
            )
            for row in sections.executors
        ],
    }
    parsed = _parse_exact_sections(raw, sections.process_id)
    assert parsed.strategy == "all-flow-union"
    assert parsed.source_dispatch_variants == sections.source_dispatch_variants
    assert parsed.resolved_source_selections == sections.resolved_source_selections

    raw["resolved_source_selections"] = [(0, 0)]
    with pytest.raises(ArtifactError, match="does not cover every external source"):
        _parse_exact_sections(raw, sections.process_id)
