# SPDX-License-Identifier: 0BSD
"""Crossed flavour-union members through the production runtime path.

A crossed member (process-groups schema 2) is its template's program at
``q[r] = sigma_r * p_member[permutation[r]]`` with member helicities
``lambda[permutation[r]] = rho_r * lambda_template[r]``. Generation does not
emit crossed members yet, so the members here are hand-built on real union
templates with the Python writer, loaded by the Rust reader and evaluated with
``evaluate_assignments``; each must equal its separately generated per-flavour
program at 1e-12 (totals, per helicity and, at leading colour, per flow).
"""

from __future__ import annotations

import importlib.util
import json
import math
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
from pyamplicol.artifacts.process_groups import (
    read_process_groups,
    write_process_groups,
)
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
from pyamplicol.generation.process_grouping import (
    GroupingBasis,
    GroupMember,
    crossing_signs,
)
from pyamplicol.models import BuiltinSMModel

# (accuracy, new member id, template expression, member expression, pattern count)
_CASES = (
    ("lc", "x_uubar_dd", "d d > d d", "u u~ > d d~", 1),
    ("lc", "x_ud_ddbar", "d d~ > d d~", "u d > u d", 1),
    ("lc", "x_dd_ddbar", "d d~ > d d~", "d d > d d", 2),
    ("lc", "x_ug_ddbarg", "d d~ > d d~ g", "u g > d u d~", 1),
    ("lc", "x_dg_ddbarg", "d d~ > d d~ g", "d g > d d d~", 2),
    ("full", "x_uubar_dd", "d d > d d", "u u~ > d d~", 1),
    ("full", "x_ud_ddbar", "d d~ > d d~", "u d > u d", 1),
    ("full", "x_dd_ddbar", "d d~ > d d~", "d d > d d", 2),
)
_UNION_EXPRESSIONS = {"lc": ("p p > j j", "p p > j j j"), "full": ("p p > j j",)}
_NAMES = {"d": 1, "u": 2, "s": 3, "c": 4, "g": 21}
# (sqrt(s), seed, off the beam axis): crossed quarks reach both Weyl branches.
_POINTS = ((600.0, 1, False), (2400.0, 2, True), (13000.0, 3, False), (900.0, 4, True))
Point = tuple[tuple[float, float, float, float], ...]


def _config(accuracy: str, *, unions: bool) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy=accuracy),
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


def _pdgs(expression: str) -> list[int]:
    out = []
    for name in expression.replace(">", " ").split():
        pdg = _NAMES[name.rstrip("~")]
        out.append(-pdg if name.endswith("~") else pdg)
    return out


def _initial(expression: str) -> int:
    return len(expression.split(">")[0].split())


def _crossing(template: str, member: str) -> list[int]:
    """Template slot -> member leg: same side first, then across, by all-outgoing
    quark/antiquark/gluon with flavours erased."""

    tp, mp = _pdgs(template), _pdgs(member)
    tn, mn = _initial(template), _initial(member)

    def key(pdg: int, initial: bool) -> object:
        return "g" if pdg == 21 else (-pdg if initial else pdg) > 0

    free = list(range(len(mp)))
    perm: list[int | None] = [None] * len(tp)
    for same_side in (True, False):
        for r, pdg in enumerate(tp):
            if perm[r] is not None:
                continue
            for m in free:
                if key(mp[m], m < mn) == key(pdg, r < tn) and (
                    not same_side or (m < mn) == (r < tn)
                ):
                    perm[r] = m
                    free.remove(m)
                    break
    return [int(value) for value in perm]  # type: ignore[arg-type]


def _normalization(expression: str) -> Fraction:
    pdgs, n = _pdgs(expression), _initial(expression)
    average, identical = BuiltinSMModel().external_state_normalization(
        pdgs[:n], pdgs[n:]
    )
    return Fraction(average) * Fraction(identical)


def _rambo(count: int, sqrt_s: float, rng: random.Random) -> list[list[float]]:
    qs = []
    for _ in range(count):
        cos, phi = 2 * rng.random() - 1, 2 * math.pi * rng.random()
        energy = -math.log(rng.random() * rng.random())
        sin = math.sqrt(1 - cos * cos)
        qs.append(
            [
                energy,
                energy * sin * math.cos(phi),
                energy * sin * math.sin(phi),
                energy * cos,
            ]
        )
    total = [sum(q[i] for q in qs) for i in range(4)]
    mass = math.sqrt(total[0] ** 2 - sum(total[i] ** 2 for i in (1, 2, 3)))
    b = [-total[i] / mass for i in (1, 2, 3)]
    x, gamma = sqrt_s / mass, total[0] / mass
    a = 1 / (1 + gamma)
    out = []
    for q in qs:
        bq = sum(b[i] * q[i + 1] for i in range(3))
        out.append(
            [x * (gamma * q[0] + bq)]
            + [x * (q[i + 1] + b[i] * q[0] + a * bq * b[i]) for i in range(3)]
        )
    return out


def _boosted(point: list[list[float]], rng: random.Random) -> list[list[float]]:
    """A generic rotation, then a boost along a generic axis."""

    a, b, c = (2 * math.pi * rng.random() for _ in range(3))
    ca, sa, cb, sb, cc, sc = (math.cos(a), math.sin(a), math.cos(b), math.sin(b),
                              math.cos(c), math.sin(c))  # fmt: skip
    rotation = [
        [ca * cc - sa * cb * sc, -ca * sc - sa * cb * cc, sa * sb],
        [sa * cc + ca * cb * sc, -sa * sc + ca * cb * cc, -ca * sb],
        [sb * sc, sb * cc, cb],
    ]
    beta = [0.6 * (2 * rng.random() - 1) for _ in range(3)]
    beta2 = sum(v * v for v in beta)
    gamma = 1 / math.sqrt(1 - beta2)
    out = []
    for p in point:
        v = [sum(rotation[i][j] * p[j + 1] for j in range(3)) for i in range(3)]
        bv = sum(beta[i] * v[i] for i in range(3))
        scale = (gamma - 1) * bv / beta2 + gamma * p[0]
        out.append([gamma * (p[0] + bv)] + [v[i] + scale * beta[i] for i in range(3)])
    return out


def _member_points(expression: str) -> list[Point]:
    final = len(expression.split(">")[1].split())
    out = []
    for sqrt_s, seed, off_axis in _POINTS:
        rng = random.Random(seed)
        half = sqrt_s / 2
        point = [[half, 0.0, 0.0, half], [half, 0.0, 0.0, -half]]
        point += _rambo(final, sqrt_s, rng)
        point = _boosted(point, rng) if off_axis else point
        out.append(tuple(tuple(leg) for leg in point))  # type: ignore[misc]
    return out


def _signed(point: Point, permutation: Sequence[int], sigma: Sequence[int]) -> Point:
    return tuple(
        tuple(s * x for x in point[leg])  # type: ignore[misc]
        for leg, s in zip(permutation, sigma, strict=True)
    )


def _add_crossed_members(artifact: Path, accuracy: str) -> None:
    """Append the crossed members of ``accuracy`` to their templates' groups."""

    group_map = read_process_groups(artifact)
    assert group_map is not None
    groups = {
        group.representative.expression: group for group in group_map.groups.values()
    }
    for case_accuracy, member_id, template, member, count in _CASES:
        if case_accuracy != accuracy:
            continue
        group = groups[template]
        assert group.pattern_pairs is not None
        perm = _crossing(template, member)
        pdgs = _pdgs(member)
        sigma = crossing_signs(perm, _initial(template), _initial(member))
        patterns = tuple(
            index
            for index, pairs in enumerate(group.pattern_pairs)
            if all(abs(pdgs[perm[f]]) == abs(pdgs[perm[a]]) for f, a in pairs)
        )
        assert len(patterns) == count, (template, member, patterns)
        group.members.append(
            GroupMember(
                process_id=member_id,
                expression=member,
                pdgs=tuple(pdgs),
                initial_count=_initial(member),
                factor=_normalization(template) / _normalization(member),
                permutation=tuple(perm),
                basis=GroupingBasis.FLAVOUR_UNION,
                generated=False,
                patterns=patterns,
                pattern_signs=(1,) * len(patterns),
                crossing_momentum_signs=sigma,
                # Crossed quarks flip their helicity label, gluons keep it.
                crossing_helicity_signs=tuple(
                    -1 if s == -1 and pdgs[perm[r]] != 21 else 1
                    for r, s in enumerate(sigma)
                ),
            )
        )
    write_process_groups(artifact, group_map)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Path]]:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    root = tmp_path_factory.mktemp("flavour-union-crossing")
    paths = {}
    try:
        for accuracy, expressions in _UNION_EXPRESSIONS.items():
            union = root / f"{accuracy}_union"
            Generator(_config(accuracy, unions=True)).generate(
                ProcessSet.from_expressions(expressions), union
            )
            _add_crossed_members(union, accuracy)
            reference = root / f"{accuracy}_reference"
            members = tuple(sorted({case[3] for case in _CASES if case[0] == accuracy}))
            Generator(_config(accuracy, unions=False)).generate(
                ProcessSet.from_expressions(members), reference
            )
            paths[f"{accuracy}_union"] = union
            paths[f"{accuracy}_reference"] = reference
        yield paths
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _template(artifact: Path, expression: str) -> Runtime:
    raw = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    process = next(p["id"] for p in raw["processes"] if p["expression"] == expression)
    return Runtime.load(artifact, process=process)


def _index(runtime: Runtime, member_id: str) -> int:
    ids = [entry["process_id"] for entry in runtime.flavour_union_assignments]
    return ids.index(member_id)


def _strings(word: Sequence[int], expression: str) -> frozenset[tuple[int, ...]]:
    """An LC word (1-based legs) as its set of open strings q ... q~ (all-outgoing)."""

    pdgs, n = _pdgs(expression), _initial(expression)
    out, current = [], []
    for label in word:
        current.append(label)
        pdg = pdgs[label - 1]
        outgoing = -pdg if label - 1 < n else pdg
        if pdg != 21 and outgoing < 0:
            out.append(tuple(current))
            current = []
    assert not current, (word, expression)
    return frozenset(out)


def _close(actual: float, expected: float, scale: float, label: object) -> None:
    # below 1e-7 of the scale: roundoff of cancellations, 1e-12 of the scale
    tolerance = 1.0e-12 * (abs(expected) if abs(expected) > 1.0e-7 * scale else scale)
    assert abs(actual - expected) <= tolerance, (label, actual, expected)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: f"{case[0]}-{case[1]}")
def test_crossed_members_equal_their_own_programs(
    artifacts: dict[str, Path], case: tuple[str, str, str, str, int]
) -> None:
    accuracy, member_id, template_expression, member_expression, _count = case
    template = _template(artifacts[f"{accuracy}_union"], template_expression)
    reference = Runtime.load(
        artifacts[f"{accuracy}_reference"], process=member_expression
    )
    index = _index(template, member_id)
    entry = template.flavour_union_assignments[index]
    perm = [int(value) for value in entry["permutation"]]
    sigma = [int(value) for value in entry["momentum_signs"]]
    rho = [int(value) for value in entry["helicity_signs"]]
    # The core re-derived sigma and rho from the sides and the slot families.
    assert perm == _crossing(template_expression, member_expression)
    assert sigma == list(
        crossing_signs(perm, _initial(template_expression), _initial(member_expression))
    )
    assert int(entry["crossing"]) > 0 and -1 in sigma
    points = _member_points(member_expression)
    signed = [_signed(point, perm, sigma) for point in points]
    totals = [row[0] for row in template.evaluate_assignments(signed, [[index]] * 4)]
    expected = [complex(value).real for value in reference.evaluate(points)]
    for total, value in zip(totals, expected, strict=True):
        _close(total, value, abs(value), (member_id, "total"))

    # Per helicity (and LC flow): template labels relabelled by rho and pi.
    member_helicity = {
        tuple(h.values): i for i, h in enumerate(reference.physics.helicities)
    }
    mapped = []
    for helicity in template.physics.helicities:
        vector = [0] * len(perm)
        for r, value in enumerate(helicity.values):
            vector[perm[r]] = rho[r] * value
        mapped.append(member_helicity[tuple(vector)])
    assert sorted(mapped) == list(range(len(member_helicity)))
    if accuracy == "lc":
        member_flows = {
            _strings(flow.word, member_expression): i
            for i, flow in enumerate(reference.physics.color_flows)
        }
        flows = [
            member_flows[
                _strings([perm[x - 1] + 1 for x in flow.word], member_expression)
            ]
            for flow in template.physics.color_flows
        ]
    else:
        flows = [0]
    rows = template.evaluate_assignments(signed, [[index]] * 4, output="helicity-flow")
    resolved = reference.evaluate_resolved(points)
    for point, row in enumerate(rows):
        values = resolved.values[point]
        scale = max(abs(complex(v).real) for line in values for v in line)
        for h, line in enumerate(row[0]):
            for c, value in enumerate(line):
                own = complex(values[mapped[h]][flows[c]]).real
                _close(value, own, scale, (member_id, point, h, c))


def test_crossed_members_are_reachable_only_as_assignments(
    artifacts: dict[str, Path],
) -> None:
    raw = json.loads(
        (artifacts["lc_union"] / "process-groups.json").read_text(encoding="utf-8")
    )
    assert raw["schema_version"] == 2
    with pytest.raises(Exception, match="crossed flavour-union member"):
        Runtime.load(artifacts["lc_union"], process="x_ud_ddbar")
    # Uncrossed members still load as before.
    uncrossed = Runtime.load(artifacts["lc_union"], process="u u > u u")
    assert uncrossed.evaluate(_member_points("u u > u u")[:1])


def test_crossed_requests_check_signs_and_batch_across_crossings(
    artifacts: dict[str, Path],
) -> None:
    template = _template(artifacts["lc_union"], "d d > d d")
    index = _index(template, "x_uubar_dd")
    entry = template.flavour_union_assignments[index]
    perm, sigma = list(entry["permutation"]), list(entry["momentum_signs"])
    member_points = _member_points("u u~ > d d~")
    signed = [_signed(point, perm, sigma) for point in member_points]
    plain = _member_points("d d > d d")
    with pytest.raises(Exception, match="energy"):
        template.evaluate_assignments(plain[:1], [[index]])
    with pytest.raises(Exception, match="energy"):
        template.evaluate_assignments(signed[:1], [[0]])
    with pytest.raises(Exception, match="different crossings"):
        template.evaluate_assignments(signed[:1], [[index, 0]])
    # Points of different crossings in one call equal separate calls.
    mixed = template.evaluate_assignments(
        [signed[0], plain[0], signed[1], plain[1]], [[index], [0], [index, index], [0]]
    )
    alone = template.evaluate_assignments(signed[:2], [[index]] * 2)
    uncrossed = template.evaluate_assignments(plain[:2], [[0]] * 2)
    assert mixed[0][0] == pytest.approx(alone[0][0], rel=1.0e-14)
    assert mixed[2] == pytest.approx((alone[1][0],) * 2, rel=1.0e-14)
    assert [mixed[1][0], mixed[3][0]] == pytest.approx(
        [uncrossed[0][0], uncrossed[1][0]], rel=1.0e-14
    )
    # The binary64 kinematics improvement refuses crossed assignments only.
    with pytest.raises(Exception, match="kinematics improvement"):
        template.evaluate_assignments(
            signed[:1], [[index]], kinematics_improvement_binary64=True
        )
    assert template.evaluate_assignments(
        plain[:1], [[0]], kinematics_improvement_binary64=True
    )


def test_relabelled_crossed_assignments_recompute_their_crossing(
    artifacts: dict[str, Path],
) -> None:
    template = _template(artifacts["lc_union"], "d d > d d")
    reference = Runtime.load(artifacts["lc_reference"], process="u u~ > d d~")
    index = _index(template, "x_uubar_dd")
    points = _member_points("u u~ > d d~")
    expected = [complex(value).real for value in reference.evaluate(points)]
    for symmetry in ([1, 0, 2, 3], [0, 1, 3, 2], [1, 0, 3, 2]):
        relabelled = template.relabel_assignment(index, symmetry)
        entry = template.flavour_union_assignments[relabelled]
        perm = [int(value) for value in entry["permutation"]]
        sigma = [int(value) for value in entry["momentum_signs"]]
        # Independent oracle: the relabelled assignment at its own signed gather.
        assert sigma == list(crossing_signs(perm, 2, 2))
        signed = [_signed(point, perm, sigma) for point in points]
        values = [
            row[0] for row in template.evaluate_assignments(signed, [[relabelled]] * 4)
        ]
        for value, own in zip(values, expected, strict=True):
            _close(value, own, abs(own), (symmetry, value))


def _member_helicity_values(
    template: Runtime, index: int, signed: Sequence[Point]
) -> list[dict[tuple[int, ...], float]]:
    """Per-helicity values of assignment ``index``, keyed by member helicity."""

    entry = template.flavour_union_assignments[index]
    perm = [int(value) for value in entry["permutation"]]
    rho = [int(value) for value in entry["helicity_signs"]]
    keys = []
    for helicity in template.physics.helicities:
        vector = [0] * len(perm)
        for r, value in enumerate(helicity.values):
            vector[perm[r]] = rho[r] * value
        keys.append(tuple(vector))
    rows = template.evaluate_assignments(
        signed, [[index]] * len(signed), output="helicity"
    )
    return [dict(zip(keys, row[0], strict=True)) for row in rows]


@pytest.mark.parametrize("expression", ["d d~ > d d~", "d d~ > d d~ g"])
def test_cross_side_relabellings_equal_their_base(
    artifacts: dict[str, Path], expression: str
) -> None:
    """rusticol.flavour-union-relabelling.v2: a quark slot moved to the other side
    as the antiquark of the same all-outgoing flavour gives a crossed id that equals
    its base at the same member point, in total and per member helicity."""

    template = _template(artifacts["lc_union"], expression)
    # d in <-> d~ out, d~ in <-> d out, and both.
    symmetries = [[3, 1, 2, 0], [0, 2, 1, 3], [3, 2, 1, 0]]
    symmetries = [perm + list(range(4, len(_pdgs(expression)))) for perm in symmetries]
    compared = refused = 0
    for index in range(len(template.flavour_union_assignments)):
        base = template.flavour_union_assignments[index]
        points = _member_points(str(base["expression"]))
        for symmetry in symmetries:
            try:
                relabelled = template.relabel_assignment(index, symmetry)
            except Exception as error:
                # Conjugated bases may not be crossed.
                assert base.get("conjugated"), (index, symmetry, error)
                assert "crossed and conjugated" in str(error), error
                refused += 1
                continue
            entry = template.flavour_union_assignments[relabelled]
            assert entry["expression"] == base["expression"]
            signed = {
                which: [
                    _signed(
                        point,
                        [int(v) for v in item["permutation"]],
                        [int(v) for v in item["momentum_signs"]],
                    )
                    for point in points
                ]
                for which, item in (("base", base), ("relabelled", entry))
            }
            own = _member_helicity_values(template, index, signed["base"])
            other = _member_helicity_values(template, relabelled, signed["relabelled"])
            for point, (left, right) in enumerate(zip(own, other, strict=True)):
                assert left.keys() == right.keys()
                total = sum(left.values())
                _close(sum(right.values()), total, abs(total), (index, symmetry, point))
                for key, value in left.items():
                    _close(right[key], value, abs(total), (index, symmetry, key))
            compared += 1
    assert compared >= len(symmetries), (compared, refused)


def test_full_colour_lanes_refuse_cross_side_relabellings(
    artifacts: dict[str, Path],
) -> None:
    template = _template(artifacts["full_union"], "d d~ > d d~")
    count = len(template.flavour_union_assignments)
    with pytest.raises(Exception, match="full-colour crossing is unsupported"):
        template.relabel_assignment(0, [3, 1, 2, 0])
    assert len(template.flavour_union_assignments) == count


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
    size_t index = (size_t)atoi(argv[3]);
    uint32_t crossing = 0;
    if (rusticol_runtime_assignment_crossing(handle, index, &crossing) != 0)
        fail("crossing");
    printf("%u\n", crossing);
    int32_t values[64];
    size_t required = 0;
    if (rusticol_runtime_assignment_momentum_signs(
            handle, index, values, 64, &required) != 0) fail("momentum signs");
    for (size_t i = 0; i < required; ++i) printf("%d ", values[i]);
    printf("\n");
    if (rusticol_runtime_assignment_helicity_signs(
            handle, index, values, 64, &required) != 0) fail("helicity signs");
    for (size_t i = 0; i < required; ++i) printf("%d ", values[i]);
    printf("\n");
    size_t count = 0;
    if (rusticol_runtime_helicity_count(handle, &count) != 0) fail("helicity count");
    for (size_t h = 0; h < count; ++h) {
        if (rusticol_runtime_assignment_helicity_vector(
                handle, index, h, values, 64, &required) != 0) fail("helicity vector");
        for (size_t i = 0; i < required; ++i) printf("%d ", values[i]);
        printf("\n");
    }
    if (rusticol_runtime_set_kinematics_improvement_binary64(handle, 1) != 0)
        fail("binary64");
    size_t legs = required;
    double *momenta = calloc(legs * 4, sizeof(double));
    for (size_t i = 0; i < legs; ++i) momenta[4 * i] = -1.0;
    size_t offsets[2] = {0, 1};
    uint32_t ids[1] = {(uint32_t)index};
    double output[1] = {7.0};
    int status = rusticol_runtime_evaluate_assignments_f64(
        handle, momenta, legs * 4, 1, offsets, 2, ids, 1, NULL, 0, NULL, 0, 0,
        output, 1);
    char message[2048];
    rusticol_last_error_message(message, sizeof message, &required);
    printf("%d %g %s\n", status, output[0], message);
    rusticol_runtime_free(handle);
    return 0;
}
"""


def test_c_abi_exposes_crossing_signs_and_member_helicities(
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
    source = tmp_path / "crossing.c"
    source.write_text(textwrap.dedent(_C_PROBE), encoding="utf-8")
    binary = tmp_path / "crossing"
    subprocess.run(
        [cc, "-std=c11", *sdk["cflags"], str(source), "-o", str(binary),
         *sdk["link_flags"]],
        check=True,
        timeout=300,
    )  # fmt: skip
    template = _template(artifacts["lc_union"], "d d~ > d d~ g")
    index = _index(template, "x_ug_ddbarg")
    entry = template.flavour_union_assignments[index]
    raw = json.loads((artifacts["lc_union"] / "artifact.json").read_text())
    process = next(
        p["id"] for p in raw["processes"] if p["expression"] == "d d~ > d d~ g"
    )
    lines = subprocess.run(
        [str(binary), str(artifacts["lc_union"]), process, str(index)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    ).stdout.splitlines()
    assert int(lines[0]) == int(entry["crossing"]) > 0
    assert [int(v) for v in lines[1].split()] == entry["momentum_signs"]
    assert [int(v) for v in lines[2].split()] == entry["helicity_signs"]
    perm, rho = entry["permutation"], entry["helicity_signs"]
    reference = Runtime.load(artifacts["lc_reference"], process="u g > d u d~")
    count = len(template.physics.helicities)
    vectors = [tuple(int(v) for v in line.split()) for line in lines[3 : 3 + count]]
    for helicity, vector in zip(template.physics.helicities, vectors, strict=True):
        expected = [0] * len(perm)
        for r, value in enumerate(helicity.values):
            expected[perm[r]] = rho[r] * value
        assert vector == tuple(expected)
    # Over all template indices: exactly the dedicated member's helicity set.
    assert sorted(vectors) == sorted(
        tuple(h.values) for h in reference.physics.helicities
    )
    status, canary, message = lines[3 + count].split(" ", 2)
    assert int(status) != 0 and float(canary) == 7.0
    assert "kinematics improvement" in message
