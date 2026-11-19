# SPDX-License-Identifier: 0BSD
"""Crossing prototype gate (WP-B): a flavour-union template evaluated at
caller-signed momenta q[r] = sigma_r * p[pi_r], with helicities relabelled by
rho, against separately generated per-flavour artifacts (LC and full colour).

    python crossing_prototype.py generate ART
    python crossing_prototype.py cases ART CASES.json OUT.json
    RUSTICOL_CROSSING_PROTOTYPE_CASES=CASES.json \
        cargo test -p rusticol-core --lib crossing_prototype -- --nocapture
    python crossing_prototype.py compare ART CASES.json OUT.json

The Rust test (engine/union_crossing_prototype_tests.rs) appends each crossed
assignment through a test-only hook and writes the template's per-helicity and
per-flow values; compare checks them at 1e-12 per (helicity, flow) after
the |M|^2 factor (avg_T id_T)/(avg_M id_M).
"""

from __future__ import annotations

import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

ANGLES = (0.0, 0.7, 2.1, 4.4)

# (name, union artifact, template expression, reference artifact, member expression,
#  expected pattern count)
CASES = [
    ("lc dd~>dd~ <- dd>dd", "lc2_union", "d d~ > d d~", "lc2_ref", "d d > d d", 2),
    (
        "lc dd~>dd~ <- d~d~>d~d~",
        "lc2_union",
        "d d~ > d d~",
        "lc2_ref",
        "d~ d~ > d~ d~",
        2,
    ),
    ("lc dd~>dd~ <- ud>ud", "lc2_union", "d d~ > d d~", "lc2_ref", "u d > u d", 1),
    ("lc dd>dd <- uu~>dd~", "lc2_union", "d d > d d", "lc2_ref", "u u~ > d d~", 1),
    ("lc dd>dd <- dd~>dd~", "lc2_union", "d d > d d", "lc2_ref", "d d~ > d d~", 2),
    ("lc dd>dd <- su~>su~", "lc2_union", "d d > d d", "lc2_ref", "s u~ > s u~", 1),
    ("fc dd~>dd~ <- dd>dd", "fc2_union", "d d~ > d d~", "fc2_ref", "d d > d d", 2),
    ("fc dd>dd <- uu~>dd~", "fc2_union", "d d > d d", "fc2_ref", "u u~ > d d~", 1),
    ("fc dd~>dd~ <- ud>ud", "fc2_union", "d d~ > d d~", "fc2_ref", "u d > u d", 1),
    (
        "lc dd~>dd~g <- dg>ddd~",
        "lc3_union",
        "d d~ > d d~ g",
        "lc3_ref",
        "d g > d d d~",
        2,
    ),
    (
        "lc dd~>dd~g <- ug>dud~",
        "lc3_union",
        "d d~ > d d~ g",
        "lc3_ref",
        "u g > d u d~",
        1,
    ),
    (
        "lc dd>ddg <- gd~>dd~d~",
        "lc3_union",
        "d d > d d g",
        "lc3_ref",
        "g d~ > d d~ d~",
        2,
    ),
    (
        "fc dd~>dd~g <- dg>ddd~",
        "fc3_union",
        "d d~ > d d~ g",
        "fc3_ref",
        "d g > d d d~",
        2,
    ),
    (
        "fc dd~>dd~g <- ug>dud~",
        "fc3_union",
        "d d~ > d d~ g",
        "fc3_ref",
        "u g > d u d~",
        1,
    ),
]

NAMES = {"d": 1, "u": 2, "s": 3, "c": 4, "g": 21}


def pdgs(expression):
    out = []
    for name in expression.replace(">", " ").split():
        anti = name.endswith("~")
        out.append(-NAMES[name.rstrip("~")] if anti else NAMES[name])
    return out


def initial_count(expression):
    return len(expression.split(">")[0].split())


def outgoing(pdg, initial):
    return -pdg if initial else pdg


def crossing(template, member):
    """pi (template slot -> member leg), sigma: same-side first, then crossed, by
    all-outgoing quark-vs-antiquark (flavours erased)."""

    tp, mp = pdgs(template), pdgs(member)
    tn, mn = initial_count(template), initial_count(member)

    def key(p, initial):
        return "g" if p == 21 else outgoing(p, initial) > 0

    key_t = [key(p, r < tn) for r, p in enumerate(tp)]
    key_m = [key(p, m < mn) for m, p in enumerate(mp)]
    free = set(range(len(mp)))
    perm = [None] * len(tp)
    for r in range(len(tp)):
        for m in sorted(free):
            if key_m[m] == key_t[r] and (m < mn) == (r < tn):
                perm[r] = m
                free.discard(m)
                break
    for r in range(len(tp)):
        if perm[r] is None:
            m = next(m for m in sorted(free) if key_m[m] == key_t[r])
            perm[r] = m
            free.discard(m)
    sigma = [-1 if (r < tn) != (perm[r] < mn) else 1 for r in range(len(tp))]
    return perm, sigma


def member_points(expression):
    from pyamplicol.models.builtin.validation import generic_validation_point

    base = [p.momentum for p in generic_validation_point(expression)]
    out = []
    for angle in ANGLES:
        c, s = math.cos(angle), math.sin(angle)
        out.append(
            [
                [
                    float(p[0]),
                    c * float(p[1]) - s * float(p[2]),
                    s * float(p[1]) + c * float(p[2]),
                    float(p[3]),
                ]
                for p in base
            ]
        )
    return out


def template_id(artifact, expression):
    raw = json.loads((artifact / "artifact.json").read_text())
    return next(p["id"] for p in raw["processes"] if p["expression"] == expression)


def make_cases(root, case_file, out_file):
    cases = []
    for name, union, template, _ref, member, count in CASES:
        perm, sigma = crossing(template, member)
        points = member_points(member)
        signed = [
            [[sigma[r] * x for x in p[perm[r]]] for r in range(len(perm))]
            for p in points
        ]
        cases.append(
            {
                "name": name,
                "artifact": str(root / union),
                "template": template_id(root / union, template),
                "template_expression": template,
                "member_expression": member,
                "member_pdgs": pdgs(member),
                "permutation": perm,
                "sigma": sigma,
                "expected_pattern_count": count,
                "member_points": points,
                "momenta": signed,
            }
        )
    Path(case_file).write_text(
        json.dumps({"cases": cases, "output": str(out_file)}, indent=1)
    )
    for case in cases:
        print(case["name"], case["permutation"], case["sigma"])


def factor(template, member):
    from pyamplicol.models import BuiltinSMModel

    model = BuiltinSMModel()

    def norm(expression):
        p, n = pdgs(expression), initial_count(expression)
        average, identical = model.external_state_normalization(p[:n], p[n:])
        return average * identical

    return Fraction(norm(template), norm(member))


def strings(word, expression):
    """An LC word (1-based legs) as its set of open strings q ... q~ (all-outgoing)."""

    p, n = pdgs(expression), initial_count(expression)
    out, current = [], []
    for label in word:
        current.append(label)
        o = outgoing(p[label - 1], label - 1 < n)
        if o != 21 and o < 0:
            out.append(tuple(current))
            current = []
    assert not current, (word, expression)
    return frozenset(out)


def compare(root, case_file, out_file):
    from pyamplicol import Runtime

    cases = {c["name"]: c for c in json.loads(Path(case_file).read_text())["cases"]}
    refs = {c[0]: c for c in CASES}
    worst = 0.0
    ok = True
    for result in json.loads(Path(out_file).read_text())["results"]:
        case = cases[result["name"]]
        ref_artifact = root / refs[result["name"]][3]
        member = Runtime.load(ref_artifact, process=case["member_expression"])
        perm, sigma = case["permutation"], case["sigma"]
        n = len(perm)
        rho = [
            -1 if sigma[r] == -1 and abs(case["member_pdgs"][perm[r]]) <= 6 else 1
            for r in range(n)
        ]
        f = factor(case["template_expression"], case["member_expression"])
        hel_t = [tuple(h) for h in result["helicities"]]
        cols_t = result["colors"]
        hel_m = {tuple(h.values): i for i, h in enumerate(member.physics.helicities)}
        flows_m = member.physics.color_flows
        if flows_m:
            member_expression = case["member_expression"]
            col_m = {
                strings(c.word, member_expression): i for i, c in enumerate(flows_m)
            }
            col_map = []
            for c in cols_t:
                mapped = [perm[x - 1] + 1 for x in c["word"]]
                col_map.append(col_m[strings(mapped, member_expression)])
        else:
            assert len(cols_t) == 1 == len(member.physics.contracted_color_components)
            col_map = [0]
        resolved = member.evaluate_resolved(
            [tuple(map(tuple, p)) for p in case["member_points"]]
        )
        totals_m = [
            complex(v).real
            for v in member.evaluate(
                [tuple(map(tuple, p)) for p in case["member_points"]]
            )
        ]
        values = result["values"]
        width = len(hel_t) * len(cols_t)
        case_worst, compared, missing = 0.0, 0, 0
        for point in range(len(case["member_points"])):
            m_values = resolved.values[point]
            scale = max(abs(complex(v).real) for row in m_values for v in row)
            for h, vector in enumerate(hel_t):
                member_vector = [0] * n
                for r in range(n):
                    member_vector[perm[r]] = rho[r] * vector[r]
                hm = hel_m.get(tuple(member_vector))
                for c in range(len(cols_t)):
                    t = values[point * width + h * len(cols_t) + c] * float(f)
                    if hm is None:
                        missing += 1
                        dev = abs(t) / scale
                    else:
                        m = complex(m_values[hm][col_map[c]]).real
                        dev = (
                            abs(t - m) / max(abs(m), 1e-300)
                            if abs(m) > 1e-9 * scale
                            else abs(t - m) / scale
                        )
                        compared += 1
                    case_worst = max(case_worst, dev)
            total_t = result["totals"][point] * float(f)
            case_worst = max(
                case_worst, abs(total_t - totals_m[point]) / abs(totals_m[point])
            )
        worst = max(worst, case_worst)
        good = case_worst <= 1e-12
        ok &= good
        print(
            f"{'PASS' if good else 'FAIL'} {result['name']}: pi={perm} sigma={sigma} rho={rho} "
            f"factor={f} patterns={result['patterns']} compared={compared} "
            f"absent-member-helicities={missing} max-rel-dev={case_worst:.2e}"
        )
    print(f"overall {'PASS' if ok else 'FAIL'} worst={worst:.2e}")
    return ok


def config(accuracy: str, unions: bool):
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


JOBS = {
    "lc2_union": ("lc", True, "p p > j j"),
    "lc2_ref": ("lc", False, "p p > j j"),
    "fc2_union": ("full", True, "p p > j j"),
    "fc2_ref": ("full", False, "p p > j j"),
    "lc3_union": ("lc", True, "p p > j j j"),
    "lc3_ref": ("lc", False, "p p > j j j"),
    "fc3_union": ("full", True, "p p > j j j"),
    "fc3_ref": ("full", False, "p p > j j j"),
}


def generate(root: Path) -> None:
    from pyamplicol import Generator, ProcessSet

    for name, (accuracy, unions, expression) in JOBS.items():
        path = root / name
        if path.exists():
            continue
        started = time.perf_counter()
        Generator(config(accuracy, unions)).generate(
            ProcessSet.from_expressions((expression,)), path
        )
        print(f"{name}: {time.perf_counter() - started:.1f} s", flush=True)


if __name__ == "__main__":
    mode, root = sys.argv[1], Path(sys.argv[2])
    if mode == "generate":
        generate(root)
    elif mode == "cases":
        make_cases(root, sys.argv[3], sys.argv[4])
    else:
        sys.exit(0 if compare(root, sys.argv[3], sys.argv[4]) else 1)
