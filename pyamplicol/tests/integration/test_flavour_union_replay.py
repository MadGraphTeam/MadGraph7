# SPDX-License-Identifier: 0BSD
"""Topology replay that moves fermion slots of flavour-union templates.

A replay route evaluates its representative at permuted sources, so the
representative's pattern ``(f, a)`` lands on the flow's pattern
``(perm[f], perm[a])``. Templates generated with such routes must equal
templates generated without them (every fermion slot fixed) for every
assignment, helicity and colour flow, with fewer program rows.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from pyamplicol import Generator, ProcessSet, Runtime
from pyamplicol.artifacts.process_groups import read_process_groups
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.generation import recurrence_projection
from pyamplicol.generation.process_grouping import GroupingBasis
from pyamplicol.models.builtin.validation import generic_validation_point

_EXPRESSIONS = (
    "d d~ > d d d~ d~",
    "d d~ > d u d~ u~",
    "d d~ > u s u~ s~",
    "u u~ > u u u~ u~",
    "d d > d d g g",
    "d u > d u g g",
    "d d~ > d d~ g g",
    "u u~ > d d~ g g",
)


def _config() -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc"),
        process=ProcessConfig(
            flavor_scheme=4,
            flavor_blind_strict=True,
            flavor_unions=True,
            max_coupling_orders={"QED": 0},
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode="recurrence",
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Path]]:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    root = tmp_path_factory.mktemp("flavour-union-replay")
    paths = {"replay": root / "replay", "fixed": root / "fixed"}
    monkeypatch = pytest.MonkeyPatch()
    try:
        for kind, path in paths.items():
            monkeypatch.setattr(
                recurrence_projection,
                "_PATTERN_RESOLVED_FERMION_REPLAY",
                kind == "replay",
            )
            Generator(_config()).generate(
                ProcessSet.from_expressions(_EXPRESSIONS), path
            )
        yield paths
    finally:
        monkeypatch.undo()
        shutil.rmtree(root, ignore_errors=True)


def _templates(artifact: Path) -> list[str]:
    groups = read_process_groups(artifact)
    assert groups is not None
    return sorted(
        template
        for template, group in groups.groups.items()
        if any(member.basis is GroupingBasis.FLAVOUR_UNION for member in group.members)
    )


def _row_count(artifact: Path, process_id: str) -> int:
    execution = json.loads(
        (artifact / "processes" / process_id / "execution.json").read_text(
            encoding="utf-8"
        )
    )
    certificate = execution["plan"]["inspection_summary"].get(
        "selector_work_certificate"
    )
    return int((certificate or {}).get("persisted_union", {}).get("row_count", 0))


def _point(expression: str, angle: float) -> tuple[tuple[float, ...], ...]:
    cos, sin = math.cos(angle), math.sin(angle)
    return tuple(
        (
            float(p[0]),
            cos * float(p[1]) - sin * float(p[2]),
            sin * float(p[1]) + cos * float(p[2]),
            float(p[3]),
        )
        for p in (
            particle.momentum for particle in generic_validation_point(expression)
        )
    )


def test_fermion_replay_templates_equal_fixed_fermion_templates(
    artifacts: dict[str, Path],
) -> None:
    templates = _templates(artifacts["replay"])
    assert templates == _templates(artifacts["fixed"])
    fewer = False
    for template_id in templates:
        replay = Runtime.load(artifacts["replay"], process=template_id)
        fixed = Runtime.load(artifacts["fixed"], process=template_id)
        assert replay.flavour_union_assignments == fixed.flavour_union_assignments
        assert replay.physics.helicity_ids == fixed.physics.helicity_ids
        assert replay.physics.color_ids == fixed.physics.color_ids
        expression = str(replay.flavour_union_assignments[0]["expression"])
        points = [_point(expression, angle) for angle in (0.3, 1.7)]
        for index in range(len(replay.flavour_union_assignments)):
            ours = replay.evaluate_assignments(
                points, [[index]] * len(points), output="helicity-flow"
            )
            theirs = fixed.evaluate_assignments(
                points, [[index]] * len(points), output="helicity-flow"
            )
            for our_point, their_point in zip(ours, theirs, strict=True):
                (our_rows,), (their_rows,) = our_point, their_point
                scale = max(abs(v) for row in their_rows for v in row)
                assert scale > 0.0
                for our_row, their_row in zip(our_rows, their_rows, strict=True):
                    assert list(our_row) == pytest.approx(
                        list(their_row), rel=1.0e-12, abs=1.0e-13 * scale
                    ), (template_id, index)
        rows = (
            _row_count(artifacts["replay"], template_id),
            _row_count(artifacts["fixed"], template_id),
        )
        assert rows[0] <= rows[1], (template_id, rows)
        fewer |= rows[0] < rows[1]
    assert fewer


def test_fermion_replay_members_equal_member_loads(
    artifacts: dict[str, Path],
) -> None:
    for template_id in _templates(artifacts["replay"]):
        template = Runtime.load(artifacts["replay"], process=template_id)
        for entry in template.flavour_union_assignments[1:]:
            member = Runtime.load(artifacts["replay"], process=str(entry["process_id"]))
            fixed = Runtime.load(artifacts["fixed"], process=str(entry["process_id"]))
            points = [_point(str(entry["expression"]), a) for a in (0.5, 2.5)]
            values = [complex(v).real for v in member.evaluate(points)]
            expected = [complex(v).real for v in fixed.evaluate(points)]
            assert values == pytest.approx(expected, rel=1.0e-12)
