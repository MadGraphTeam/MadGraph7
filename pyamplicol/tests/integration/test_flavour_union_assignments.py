# SPDX-License-Identifier: 0BSD
"""Multi-assignment evaluation of flavour-union templates.

``Runtime.evaluate_assignments`` evaluates several flavour assignments of one
pattern-resolved template per point (momenta in template slot order). A
single requested assignment must equal loading that member bitwise, a mixed
batch must equal per-request calls bitwise, and every assignment must match
its separately generated per-flavour program.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import math
import operator
import random
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Iterator, Sequence
from fractions import Fraction
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

_EXPRESSIONS = (
    "d d > d d g g",
    "d u > d u g g",
    "u u > u u g g",
    "u~ d~ > u~ d~ g g",
    "d d~ > d d d~ d~",
    "d d~ > d u d~ u~",
    "d d~ > u s u~ s~",
    "u u~ > u u u~ u~",
)
Point = tuple[tuple[float, float, float, float], ...]


def _config(*, unions: bool) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc"),
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


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Path]]:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    root = tmp_path_factory.mktemp("flavour-union-assignments")
    paths = {"union": root / "union", "reference": root / "reference"}
    try:
        for kind, path in paths.items():
            Generator(_config(unions=kind == "union")).generate(
                ProcessSet.from_expressions(_EXPRESSIONS), path
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


def test_assignment_table_lists_the_template_then_its_members(
    artifacts: dict[str, Path],
) -> None:
    templates = _templates(artifacts["union"])
    assert templates
    for template, members in templates.items():
        table = Runtime.load(artifacts["union"], process=template)
        ids = [entry["process_id"] for entry in table.flavour_union_assignments]
        assert ids == [template, *members]
        assert table.flavour_union_assignments[0]["permutation"] == list(
            range(len(table.flavour_union_assignments[0]["pdgs"]))
        )
        member = Runtime.load(artifacts["union"], process=members[0])
        assert [
            entry["process_id"] for entry in member.flavour_union_assignments
        ] == ids
        with pytest.raises(Exception, match="flavour-union"):
            _ = member.process_group_factor


def test_single_assignments_equal_member_loads_and_own_programs(
    artifacts: dict[str, Path],
) -> None:
    for template_id in _templates(artifacts["union"]):
        template = Runtime.load(artifacts["union"], process=template_id)
        for index, entry in enumerate(template.flavour_union_assignments):
            expression = str(entry["expression"])
            points = [_point(expression, angle) for angle in (0.0, 0.7, 2.1)]
            permutation = [int(value) for value in entry["permutation"]]
            template_points = [_to_template(p, permutation) for p in points]
            routed = template.evaluate_assignments(
                template_points, [[index]] * len(points)
            )
            member = Runtime.load(artifacts["union"], process=str(entry["process_id"]))
            # Bitwise: the same restricted program, combination and factor.
            assert [row[0] for row in routed] == _real(member.evaluate(points))
            resolved = template.evaluate_assignments(
                template_points[:1], [[index]], output="helicity-flow"
            )[0][0]
            member_resolved = member.evaluate_resolved(points[:1]).values[0]
            assert [list(map(float, row)) for row in resolved] == [
                [complex(value).real for value in row] for row in member_resolved
            ]
            reference = Runtime.load(artifacts["reference"], process=expression)
            assert [row[0] for row in routed] == pytest.approx(
                _real(reference.evaluate(points)), rel=1.0e-12
            ), expression
            factor = entry["factor"]
            assert Fraction(factor["numerator"], factor["denominator"]) > 0


def test_mixed_requests_match_per_request_calls(
    artifacts: dict[str, Path],
) -> None:
    """Rows follow the request; values agree with per-request calls.

    Points are regrouped by pattern mask, which moves them between SIMD
    lanes; like ordinary batches (lane-dependent rounding of the executors)
    this is exact up to a few ulp, and bitwise when positions are kept.
    """

    generator = random.Random(20261118)
    for template_id in _templates(artifacts["union"]):
        template = Runtime.load(artifacts["union"], process=template_id)
        count = len(template.flavour_union_assignments)
        expression = str(template.flavour_union_assignments[0]["expression"])
        points = [_point(expression, 0.37 * index) for index in range(9)]
        requests = []
        for _ in points:
            size = generator.randint(1, count + 1)
            requests.append([generator.randrange(count) for _ in range(size)])
        requests[0] = list(range(count - 1, -1, -1))  # all, out of order
        requests[1] = [requests[1][0]] * 3  # duplicates repeat rows
        mixed = template.evaluate_assignments(points, requests)
        homogeneous = [
            [row[0] for row in template.evaluate_assignments(points, [[index]] * 9)]
            for index in range(count)
        ]
        for point_index, (point, request, rows) in enumerate(
            zip(points, requests, mixed, strict=True)
        ):
            assert len(rows) == len(request)
            alone = template.evaluate_assignments([point], [request])[0]
            assert list(rows) == pytest.approx(list(alone), rel=1.0e-14, abs=0.0)
            # Same positions, different restriction masks: bitwise.
            same_positions = template.evaluate_assignments(
                points, [request] * len(points)
            )[point_index]
            assert list(same_positions) == [
                homogeneous[index][point_index] for index in request
            ]
        per_helicity = template.evaluate_assignments(
            points[:2], requests[:2], output="helicity"
        )
        flows = template.evaluate_assignments(
            points[:2], requests[:2], output="helicity-flow"
        )
        for helicity_rows, flow_rows, total_rows in zip(
            per_helicity, flows, mixed[:2], strict=True
        ):
            for helicities, by_flow, total in zip(
                helicity_rows, flow_rows, total_rows, strict=True
            ):
                # Plain left-to-right sums (Python's sum() is compensated).
                assert list(helicities) == [
                    functools.reduce(operator.add, row, 0.0) for row in by_flow
                ]
                assert sum(helicities) == pytest.approx(total, rel=1.0e-12)


def test_restricted_programs_match_the_full_program(
    artifacts: dict[str, Path],
) -> None:
    for template_id in _templates(artifacts["union"]):
        template = Runtime.load(artifacts["union"], process=template_id)
        expression = str(template.flavour_union_assignments[0]["expression"])
        point = _point(expression, 0.4)
        for index in range(1, len(template.flavour_union_assignments)):
            restricted = template.evaluate_assignments(
                [point], [[index]], output="helicity-flow"
            )[0][0]
            # Requesting the template too runs the complete program.
            full = template.evaluate_assignments(
                [point], [[0, index]], output="helicity-flow"
            )[0][1]
            scale = max(abs(value) for row in full for value in row)
            for restricted_row, full_row in zip(restricted, full, strict=True):
                assert list(restricted_row) == pytest.approx(
                    list(full_row), rel=1.0e-14, abs=1.0e-14 * scale
                )


def test_requests_are_validated_and_restricted_programs_bounded(
    artifacts: dict[str, Path],
) -> None:
    template_id = next(iter(_templates(artifacts["union"])))
    template = Runtime.load(artifacts["union"], process=template_id)
    expression = str(template.flavour_union_assignments[0]["expression"])
    point = _point(expression)
    count = len(template.flavour_union_assignments)
    with pytest.raises(Exception, match="at least one assignment"):
        template.evaluate_assignments([point, point], [[0], []])
    with pytest.raises(Exception, match="unknown flavour-union assignment"):
        template.evaluate_assignments([point], [[count]])
    helicities = [helicity.id for helicity in template.physics.helicities]
    member = count - 1
    by_point = template.evaluate_assignments(
        [point] * len(helicities),
        [[member]] * len(helicities),
        helicity_by_point=helicities,
    )
    reference = Runtime.load(
        artifacts["union"],
        process=str(template.flavour_union_assignments[member]["process_id"]),
    )
    permutation = template.flavour_union_assignments[member]["permutation"]
    member_point = [None] * len(point)
    for slot, leg in enumerate(permutation):
        member_point[leg] = point[slot]
    for helicity, row in zip(helicities, by_point, strict=True):
        index = helicities.index(helicity)
        expected = reference.evaluate(
            (tuple(member_point),),
            helicities=(reference.physics.helicities[index].id,),
        )
        assert row[0] == pytest.approx(complex(expected[0]).real, rel=1.0e-13)
    cache = template._backend._runtime._restricted_program_cache_len()  # type: ignore[attr-defined]
    assert 0 < cache <= 64


_C_PROBE = r"""
#include <stdio.h>
#include <stdlib.h>
#include "rusticol.h"

static void fail(const char *what) {
    char message[2048];
    size_t required = 0;
    rusticol_last_error_message(message, sizeof message, &required);
    fprintf(stderr, "%s: %s\n", what, message);
    exit(1);
}

int main(int argc, char **argv) {
    RusticolRuntimeHandle *handle = NULL;
    if (rusticol_runtime_load(argv[1], argv[2], NULL, &handle) != 0) fail("load");
    size_t count = 0;
    if (rusticol_runtime_assignment_count(handle, &count) != 0) fail("count");
    FILE *input = fopen(argv[3], "r");
    size_t legs = 0;
    if (fscanf(input, "%zu", &legs) != 1) return 2;
    double *momenta = malloc(sizeof(double) * legs * 4);
    for (size_t i = 0; i < legs * 4; ++i) {
        if (fscanf(input, "%lf", &momenta[i]) != 1) return 2;
    }
    fclose(input);
    size_t offsets[2] = {0, count};
    uint32_t *ids = malloc(sizeof(uint32_t) * count);
    for (size_t i = 0; i < count; ++i) ids[i] = (uint32_t)(count - 1 - i);
    double *output = malloc(sizeof(double) * count);
    if (rusticol_runtime_evaluate_assignments_f64(
            handle, momenta, legs * 4, 1, offsets, 2, ids, count,
            NULL, 0, NULL, 0, 0, output, count) != 0) fail("evaluate");
    printf("%zu\n", count);
    for (size_t i = 0; i < count; ++i) printf("%a\n", output[i]);
    char process[256];
    size_t required = 0;
    if (rusticol_runtime_assignment_process_id(
            handle, count - 1, process, sizeof process, &required) != 0) fail("id");
    printf("%s\n", process);
    rusticol_runtime_free(handle);
    return 0;
}
"""


def test_c_abi_evaluates_assignments(
    artifacts: dict[str, Path], tmp_path: Path
) -> None:
    configured = subprocess.run(
        [sys.executable, "-m", "rusticol_config", "--json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    cc = shutil.which("cc")
    if configured.returncode != 0 or cc is None:
        pytest.skip("the installed Rusticol SDK or a C compiler is unavailable")
    sdk = json.loads(configured.stdout)
    source = tmp_path / "assignments.c"
    source.write_text(textwrap.dedent(_C_PROBE), encoding="utf-8")
    binary = tmp_path / "assignments"
    subprocess.run(
        [
            cc,
            "-std=c11",
            *sdk["cflags"],
            str(source),
            "-o",
            str(binary),
            *sdk["link_flags"],
        ],
        check=True,
        timeout=300,
    )
    template_id, members = next(iter(_templates(artifacts["union"]).items()))
    template = Runtime.load(artifacts["union"], process=template_id)
    expression = str(template.flavour_union_assignments[0]["expression"])
    point = _point(expression, 1.3)
    momenta = tmp_path / "point.txt"
    momenta.write_text(
        f"{len(point)}\n" + "\n".join(" ".join(repr(v) for v in p) for p in point),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [str(binary), str(artifacts["union"]), template_id, str(momenta)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = completed.stdout.split()
    count = int(lines[0])
    values = [float.fromhex(value) for value in lines[1 : 1 + count]]
    expected = template.evaluate_assignments([point], [list(range(count))[::-1]])[0]
    assert values == list(expected)
    assert lines[1 + count] == members[-1]
