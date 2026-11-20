# SPDX-License-Identifier: 0BSD
"""Flavour-union assignments relabelled by template slot symmetries.

A permutation ``pi`` of template slots that keeps side and particle leaves the
template invariant, so per pairing pattern ``A_P(pi.q) = s_pi A_{pi(P)}(q)``
with a sign common to all patterns. ``Runtime.relabel_assignment`` returns the
assignment with patterns ``pi(S)``: at momenta ``q`` it must equal the original
assignment at ``q'[r] = q[pi[r]]``, and the member loaded with the relabelled
leg permutation. Members with several (but not all) patterns test the common
sign.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import shutil
import subprocess
import sys
import textwrap
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

_EXPRESSIONS = (
    "d d > d d g g",
    "d u > d u g g",
    "d d~ > d d d~ d~",
    "d d~ > d u d~ u~",
    "d d~ > u s u~ s~",
    "d d > d d d d~",
    "d d > d d u u~",
    "u d > u d d d~",
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
def artifact(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    root = tmp_path_factory.mktemp("flavour-union-relabelling")
    try:
        Generator(_config(unions=True)).generate(
            ProcessSet.from_expressions(_EXPRESSIONS), root / "union"
        )
        yield root / "union"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _templates(artifact: Path) -> list[str]:
    groups = read_process_groups(artifact)
    assert groups is not None
    return [
        template
        for template, group in groups.groups.items()
        if any(member.basis is GroupingBasis.FLAVOUR_UNION for member in group.members)
    ]


def _point(expression: str, angle: float = 0.0) -> Point:
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


def _six_quark_template(artifact: Path) -> str:
    groups = read_process_groups(artifact)
    assert groups is not None
    return next(
        template
        for template in _templates(artifact)
        if groups.groups[template].representative.expression == "d d~ > d d d~ d~"
    )


def _symmetries(pdgs: Sequence[int], initial: int) -> list[tuple[int, ...]]:
    """Every permutation of slots keeping side and PDG."""

    blocks: dict[tuple[bool, int], list[int]] = {}
    for slot, pdg in enumerate(pdgs):
        blocks.setdefault((slot < initial, pdg), []).append(slot)
    result = []
    for images in itertools.product(
        *(itertools.permutations(slots) for slots in blocks.values())
    ):
        permutation = list(range(len(pdgs)))
        for slots, image in zip(blocks.values(), images, strict=True):
            for slot, target in zip(slots, image, strict=True):
                permutation[slot] = target
        result.append(tuple(permutation))
    return result


def _initial_count(expression: str) -> int:
    return len(expression.split(">")[0].split())


def test_pattern_tables_reach_the_runtime(artifact: Path) -> None:
    groups = read_process_groups(artifact)
    assert groups is not None
    templates = _templates(artifact)
    assert len(templates) == 3
    for template_id in templates:
        group = groups.groups[template_id]
        runtime = Runtime.load(artifact, process=template_id)
        pairs = runtime.flavour_union_pattern_pairs
        assert pairs is not None and group.pattern_pairs is not None
        assert {tuple(sorted(p)) for p in pairs} == {
            tuple(sorted(p)) for p in group.pattern_pairs
        }
        assert [sorted(p) for p in pairs] == [sorted(p) for p in group.pattern_pairs]
        # Each member's patterns are the rules joining equal flavours.
        for entry in runtime.flavour_union_assignments[1:]:
            flavour = [abs(entry["pdgs"][leg]) for leg in entry["permutation"]]
            expected = [
                pattern
                for pattern, rule in enumerate(pairs)
                if all(flavour[f] == flavour[a] for f, a in rule)
            ]
            assert sorted(entry["patterns"]) == expected


def test_relabelled_assignments_equal_permuted_evaluations(artifact: Path) -> None:
    for template_id in _templates(artifact):
        template = Runtime.load(artifact, process=template_id)
        assignments = template.flavour_union_assignments
        expression = str(assignments[0]["expression"])
        pdgs = [int(pdg) for pdg in assignments[0]["pdgs"]]
        points = [_point(expression, angle) for angle in (0.0, 0.9, 2.3)]
        symmetries = _symmetries(pdgs, _initial_count(expression))
        assert len(symmetries) >= 4
        pattern_count = len(assignments[0]["patterns"])
        several = 0
        for permutation in symmetries:
            derived = [
                template.relabel_assignment(index, permutation)
                for index in range(len(assignments))
            ]
            # One call for every relabelled assignment at q ...
            rows = template.evaluate_assignments(points, [derived] * len(points))
            for index, base in enumerate(assignments):
                if 1 < len(base["patterns"]) < pattern_count:
                    several += 1
                # ... equals the base assignment at q'[r] = q[pi[r]]
                permuted = [tuple(q[s] for s in permutation) for q in points]
                expected = template.evaluate_assignments(
                    permuted, [[index]] * len(points)
                )
                assert [row[index] for row in rows] == pytest.approx(
                    [row[0] for row in expected], rel=1.0e-12
                ), (template_id, base["process_id"], permutation)
        final = pdgs[_initial_count(expression) :]
        if any(final.count(pdg) >= 3 for pdg in final):
            # a 3-cycle of identical outgoing quarks
            assert any(sum(v != i for i, v in enumerate(p)) == 3 for p in symmetries)
        if pattern_count > 2:
            assert several, template_id


def test_relabelled_assignments_equal_member_loads(artifact: Path) -> None:
    for template_id in _templates(artifact):
        template = Runtime.load(artifact, process=template_id)
        assignments = template.flavour_union_assignments
        expression = str(assignments[0]["expression"])
        pdgs = [int(pdg) for pdg in assignments[0]["pdgs"]]
        for permutation in _symmetries(pdgs, _initial_count(expression))[1:4]:
            for index in range(1, len(assignments)):
                derived = template.relabel_assignment(index, permutation)
                entry = template.flavour_union_assignments[derived]
                if derived != index:
                    # interned: the first symmetry giving these patterns
                    assert entry["relabelled"]["assignment"] == index
                if int(entry.get("crossing", 0)):
                    # No member load; test_flavour_union_crossing.py is the oracle.
                    with pytest.raises(Exception, match="crossed flavour-union"):
                        Runtime.load(artifact, process=str(entry["process_id"]))
                    continue
                member = Runtime.load(artifact, process=str(entry["process_id"]))
                member_expression = str(entry["expression"])
                points = [_point(member_expression, a) for a in (0.2, 1.4)]
                legs = [int(leg) for leg in entry["permutation"]]
                template_points = [tuple(p[leg] for leg in legs) for p in points]
                values = template.evaluate_assignments(
                    template_points, [[derived]] * len(points)
                )
                assert [row[0] for row in values] == pytest.approx(
                    [complex(v).real for v in member.evaluate(points)], rel=1.0e-12
                )


def test_relabelling_is_interned_and_validated(artifact: Path) -> None:
    template_id = _six_quark_template(artifact)
    template = Runtime.load(artifact, process=template_id)
    count = len(template.flavour_union_assignments)
    pdgs = [int(pdg) for pdg in template.flavour_union_assignments[0]["pdgs"]]
    identity = list(range(len(pdgs)))
    swap = list(identity)
    first, second = (slot for slot in range(2, len(pdgs)) if pdgs[slot] == pdgs[2])
    swap[first], swap[second] = second, first
    assert template.relabel_assignment(0, swap) == 0
    assert template.relabel_assignment(count - 1, identity) == count - 1
    member = next(
        index
        for index, entry in enumerate(template.flavour_union_assignments)
        if template.relabel_assignment(index, swap) != index
    )
    derived = template.relabel_assignment(member, swap)
    assert derived >= count
    assert template.relabel_assignment(member, swap) == derived
    assert template.relabel_assignment(derived, swap) == template.relabel_assignment(
        derived, swap
    )
    crossing = list(identity)
    crossing[0], crossing[2] = 2, 0
    with pytest.raises(Exception, match="same side and particle"):
        template.relabel_assignment(member, crossing)
    with pytest.raises(Exception, match="permutation"):
        template.relabel_assignment(member, identity[:-1])
    with pytest.raises(Exception, match="unknown flavour-union assignment"):
        template.relabel_assignment(10_000, swap)
    plain = Runtime.load(artifact, process=template_id)
    assert len(plain.flavour_union_assignments) == count


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
    size_t patterns = 0, required = 0;
    if (rusticol_runtime_pattern_count(handle, &patterns) != 0) fail("count");
    printf("%zu\n", patterns);
    for (size_t p = 0; p < patterns; ++p) {
        uint32_t pairs[64];
        if (rusticol_runtime_pattern_pairs(handle, p, pairs, 64, &required) != 0)
            fail("pairs");
        for (size_t i = 0; i < required; ++i) printf("%u ", pairs[i]);
        printf("\n");
    }
    size_t n = (size_t)(argc - 4);
    size_t *permutation = malloc(sizeof(size_t) * n);
    for (size_t i = 0; i < n; ++i) permutation[i] = (size_t)atoi(argv[4 + i]);
    uint32_t relabelled = 0;
    if (rusticol_runtime_relabel_assignment(
            handle, (uint32_t)atoi(argv[3]), permutation, n, &relabelled) != 0)
        fail("relabel");
    printf("%u\n", relabelled);
    rusticol_runtime_free(handle);
    return 0;
}
"""


def test_c_abi_pattern_tables_and_relabelling(artifact: Path, tmp_path: Path) -> None:
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
    source = tmp_path / "relabel.c"
    source.write_text(textwrap.dedent(_C_PROBE), encoding="utf-8")
    binary = tmp_path / "relabel"
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
    template_id = _six_quark_template(artifact)
    template = Runtime.load(artifact, process=template_id)
    pdgs = [int(pdg) for pdg in template.flavour_union_assignments[0]["pdgs"]]
    swap = list(range(len(pdgs)))
    first, second = (slot for slot in range(2, len(pdgs)) if pdgs[slot] == pdgs[2])
    swap[first], swap[second] = second, first
    member = len(template.flavour_union_assignments) - 1
    lines = subprocess.run(
        [str(binary), str(artifact), template_id, str(member), *map(str, swap)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    ).stdout.splitlines()
    pairs = template.flavour_union_pattern_pairs
    assert pairs is not None
    assert int(lines[0]) == len(pairs)
    for line, pattern in zip(lines[1 : 1 + len(pairs)], pairs, strict=True):
        assert [int(v) for v in line.split()] == [s for pair in pattern for s in pair]
    # A fresh handle interns the same first relabelling id.
    fresh = Runtime.load(artifact, process=template_id)
    assert int(lines[-1]) == fresh.relabel_assignment(member, swap)
