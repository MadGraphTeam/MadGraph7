# SPDX-License-Identifier: 0BSD
"""Full-colour flavour-union artifacts against separately generated subprocesses.

A full-colour (contracted-colour) template keeps one amplitude per fermion
pairing pattern; a member sums its patterns per colour-basis element and
helicity before the colour contraction. Every subprocess, loaded by its key or
evaluated as an assignment of its template, must match the full-colour program
generated for it alone, per helicity.

``p p > j j j`` and ``p p > j j j j`` run only with
``PYAMPLICOL_RUN_FLAVOUR_UNION=pp3j,pp4j`` (either or both).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import shutil
import time
from collections.abc import Iterator, Sequence
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
from pyamplicol.generation.process_grouping import GroupingBasis
from pyamplicol.models.builtin.validation import generic_validation_point

_EXPLICIT = (
    "d d~ > d d d~ d~",
    "d d~ > d u d~ u~",
    "d d~ > u s u~ s~",
    "u u~ > u u u~ u~",
    "d d > d d g g",
    "d u > d u g g",
    "u~ d~ > u~ d~ g g",
    "d d~ > d d~ g g",
    "d d~ > u u~ g g",
    "d u~ > d u~ g g",
)
_CASES = {
    "explicit": _EXPLICIT,
    "pp2j": ("p p > j j",),
    "pp3j": ("p p > j j j",),
    "pp4j": ("p p > j j j j",),
}
_OPT_IN = {"pp3j", "pp4j"}
_OPT_IN_ENV = "PYAMPLICOL_RUN_FLAVOUR_UNION"
Point = tuple[tuple[float, float, float, float], ...]


def _config(*, unions: bool) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="full"),
        process=ProcessConfig(
            flavor_scheme=4,
            flavor_blind_strict=unions,
            flavor_unions=unions,
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


def _execution(artifact: Path, process_id: str) -> dict[str, object]:
    return json.loads(
        (artifact / "processes" / process_id / "execution.json").read_text(
            encoding="utf-8"
        )
    )


def _row_count(artifact: Path) -> int:
    raw = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    rows = 0
    for process in raw["processes"]:
        summary = _execution(artifact, process["id"])["plan"]["inspection_summary"]
        rows += int(summary["selector_work_certificate"]["persisted_union"]["row_count"])
    return rows


@pytest.fixture(scope="module", params=tuple(_CASES))
def artifacts(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, Path]]:
    if request.param in _OPT_IN and request.param not in os.environ.get(
        _OPT_IN_ENV, ""
    ).split(","):
        pytest.skip(f"set {_OPT_IN_ENV}={request.param} to run it")
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    root = tmp_path_factory.mktemp(f"full-colour-unions-{request.param}")
    paths = {"union": root / "union", "reference": root / "reference"}
    processes = ProcessSet.from_expressions(_CASES[request.param])
    try:
        for kind, path in paths.items():
            started = time.perf_counter()
            Generator(_config(unions=kind == "union")).generate(processes, path)
            print(
                f"{request.param} {kind}: {time.perf_counter() - started:.1f} s, "
                f"{_row_count(path)} rows"
            )
        yield paths
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _templates(artifact: Path) -> dict[str, list[str]]:
    groups = read_process_groups(artifact)
    assert groups is not None
    result = {
        template: [
            member.process_id
            for member in group.members[1:]
            if member.basis is GroupingBasis.FLAVOUR_UNION
        ]
        for template, group in groups.groups.items()
    }
    return {template: members for template, members in result.items() if members}


def _process_keys(artifact: Path) -> dict[str, str]:
    raw = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    return {
        str(process["id"]): str(process["expression"]) for process in raw["processes"]
    }


def _point(expression: str, angle: float = 0.0) -> Point:
    """The validation point rotated by ``angle`` about the beam axis."""

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


def _to_template(point: Point, permutation: Sequence[int]) -> Point:
    """Member momenta in template slot order: ``q[r] = p[permutation[r]]``."""

    return tuple(point[index] for index in permutation)


def _real(values: Sequence[object]) -> list[float]:
    return [complex(value).real for value in values]  # type: ignore[arg-type]


def _helicity_values(runtime: Runtime, point: Point) -> dict[str, float]:
    resolved = runtime.evaluate_resolved((point,))
    return {
        helicity: sum(complex(value).real for value in resolved.values[0][h])
        for h, helicity in enumerate(resolved.helicity_ids)
    }


def _assert_close(
    actual: Sequence[float], expected: Sequence[float], context: object
) -> None:
    scale = max(abs(value) for value in expected)
    assert scale > 0.0, context
    assert list(actual) == pytest.approx(
        list(expected), rel=1.0e-12, abs=1.0e-12 * scale
    ), context


def test_full_colour_unions_route_members_through_templates(
    artifacts: dict[str, Path],
) -> None:
    templates = _templates(artifacts["union"])
    assert templates
    assert len(_process_keys(artifacts["union"])) < len(
        _process_keys(artifacts["reference"])
    )
    for template in templates:
        execution = _execution(artifacts["union"], template)
        # Members cannot use a companion that sums every pattern.
        assert execution.get("helicity_selector_companion") is None
        assert execution["plan"]["inspection_summary"]["lc_flow_layout"] == (
            "contracted-color-union"
        )


def test_every_subprocess_matches_its_own_full_colour_program(
    artifacts: dict[str, Path],
) -> None:
    for key, expression in _process_keys(artifacts["reference"]).items():
        routed = Runtime.load(artifacts["union"], process=key)
        reference = Runtime.load(artifacts["reference"], process=key)
        assert routed.physics.process == expression
        point = _point(expression, 0.3)
        actual, expected = (
            _helicity_values(runtime, point) for runtime in (routed, reference)
        )
        assert actual.keys() == expected.keys(), expression
        _assert_close(
            [actual[key] for key in expected], list(expected.values()), expression
        )
        points = (point, _point(expression, 1.9))
        _assert_close(
            _real(routed.evaluate(points)), _real(reference.evaluate(points)), expression
        )
        # One helicity at a time: the template runs without a companion.
        helicity = max(expected, key=lambda key: expected[key])
        _assert_close(
            _real(routed.evaluate(points, helicities=[helicity])),
            _real(reference.evaluate(points, helicities=[helicity])),
            (expression, helicity),
        )


def test_assignments_equal_member_loads_and_own_programs(
    artifacts: dict[str, Path],
) -> None:
    for template_id in _templates(artifacts["union"]):
        template = Runtime.load(artifacts["union"], process=template_id)
        helicity_ids = list(template.physics.helicity_ids)
        for index, entry in enumerate(template.flavour_union_assignments):
            expression = str(entry["expression"])
            points = [_point(expression, angle) for angle in (0.0, 0.7, 2.1)]
            permutation = [int(value) for value in entry["permutation"]]
            template_points = [_to_template(p, permutation) for p in points]
            routed = template.evaluate_assignments(
                template_points, [[index]] * len(points)
            )
            member = Runtime.load(artifacts["union"], process=str(entry["process_id"]))
            # Same program, pattern sum, contraction and factor: bitwise.
            assert [row[0] for row in routed] == _real(member.evaluate(points))
            reference = Runtime.load(artifacts["reference"], process=expression)
            _assert_close(
                [row[0] for row in routed], _real(reference.evaluate(points)), expression
            )
            # Per helicity: the member's helicity labels are the template's.
            per_helicity = template.evaluate_assignments(
                template_points[:1], [[index]], output="helicity"
            )[0][0]
            own = _helicity_values(member, points[0])
            assert dict(zip(helicity_ids, per_helicity, strict=True)) == pytest.approx(
                own, rel=1.0e-12, abs=1.0e-12 * max(own.values())
            )


def test_per_point_helicities_and_mixed_requests(
    artifacts: dict[str, Path],
) -> None:
    generator = random.Random(20261119)
    for template_id in _templates(artifacts["union"]):
        template = Runtime.load(artifacts["union"], process=template_id)
        count = len(template.flavour_union_assignments)
        helicity_ids = list(template.physics.helicity_ids)
        expression = str(template.flavour_union_assignments[0]["expression"])
        points = [_point(expression, 0.37 * index) for index in range(9)]
        requests = [
            [generator.randrange(count) for _ in range(generator.randint(1, count + 1))]
            for _ in points
        ]
        requests[0] = list(range(count - 1, -1, -1))
        mixed = template.evaluate_assignments(points, requests)
        resolved = template.evaluate_assignments(points, requests, output="helicity")
        for rows, request, by_helicity in zip(mixed, requests, resolved, strict=True):
            assert len(rows) == len(request)
            for total, helicities in zip(rows, by_helicity, strict=True):
                assert sum(helicities) == pytest.approx(total, rel=1.0e-12)
        homogeneous = {
            index: [row[0] for row in template.evaluate_assignments(points, [[index]] * 9)]
            for index in range(count)
        }
        for point_index, (rows, request) in enumerate(zip(mixed, requests, strict=True)):
            assert list(rows) == pytest.approx(
                [homogeneous[index][point_index] for index in request], rel=1.0e-14
            )
        # Per-point helicities select one entry of the per-helicity rows.
        chosen = [generator.randrange(len(helicity_ids)) for _ in points]
        selected = template.evaluate_assignments(
            points,
            requests,
            helicity_by_point=[helicity_ids[index] for index in chosen],
        )
        for rows, by_helicity, helicity in zip(selected, resolved, chosen, strict=True):
            assert list(rows) == pytest.approx(
                [row[helicity] for row in by_helicity], rel=1.0e-12, abs=0.0
            )
        with pytest.raises(Exception, match="colou?r"):
            template.evaluate_assignments(
                points[:1],
                [[0]],
                color_flow_by_point=[str(template.physics.color_ids[0])],
            )
