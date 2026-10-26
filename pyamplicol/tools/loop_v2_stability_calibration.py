#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Calibration of the v2 stability estimates against Float(256) (WP2.3, O15).

``run`` evaluates a deterministic sample of phase-space points of one process
with the native v2 executor (``loops.v2.plan_export.NativeExecutor``):

* in binary64 (optionally also double-double) with every check of the
  calibration configuration: the reduction-direction check and two rotations;
* in 256-bit arbitrary precision (``Precision.arbitrary(57)``: the native
  ``Float`` carrier works at 57 + 20 guard digits = 77 decimal digits = 256
  bits), the truth.

Each point is one JSON line: the point, every check's absolute budget and the
absolute sums S per squared-order sector and Laurent coefficient (so that any
subset of checks can be recombined, ``StabilityEstimate.subset``), the
virtual in each carrier and the timings. A point whose truth fails (a reduction
lane failed, e.g. ``ImpreciseResult`` of a 256-bit master) is recorded with its
failure and excluded from the rates.

``report`` aggregates the lines of a directory: per process, carrier and
check configuration, the false-negative rate (points whose true error exceeds
the estimate), for the Laurent vector (``StabilityEstimate.relative``) and for
the finite part alone (``StabilityEstimate.finite``); the distribution of
estimate / true error; the smallest safety factor that would have bracketed
each point; and the rate of points an estimate flags above given thresholds.

Points: 2 -> n massless points at sqrt(s) = 1 TeV from RAMBO, rounded to
exactly on-shell, exactly momentum-conserving rationals (rational directions
on the sphere by inverse stereographic projection; the last momentum keeps
the transverse balance, the incoming momenta follow from the total), with
p_T > 20 GeV for every outgoing leg and Delta R > 0.4 between any two of
them; mu_R = 1 TeV, alpha_s = 0.118, M_T = 173 GeV, M_B = 0 (loop_sm).

Usage:
    python tools/loop_v2_stability_calibration.py run PROCESS START STOP OUTDIR
        [--batch N] [--double-double] [--model-cache DIR] [--ufo PATH]
    python tools/loop_v2_stability_calibration.py report OUTDIR [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # the ``tools`` package, run as a script
    sys.path.insert(0, str(ROOT))
UFO = Path(
    os.environ.get(
        "PYAMPLICOL_NLO_UFO_REFERENCE",
        ROOT / "FOR_REFERENCE_ONLY" / "MG5_aMC_v3_7_0" / "models" / "loop_sm",
    )
)
ENERGY = 1000.0
SCALE_SQUARED = "1000000"
PARAMETERS = {"aS": "0.118", "MT": "173"}
FIXED = {"MB": "0"}
PT_MIN = 20.0
DELTA_R_MIN = 0.4
# Float(256): the native carrier runs at the digits plus 20 guard digits.
TRUTH_DIGITS = 57
RESCALED_SUFFIX = "-rescaled"
CARRIERS = ("double", "double-double")
LAURENT = ("finite", "single_pole", "double_pole")


@dataclass(frozen=True)
class Process:
    process: str
    born: Mapping[str, int]
    # Photon exchange only, as calibrated (the Z on loop quark lines was
    # admitted later, WP3.1 I-AX).
    ignored: tuple[int, ...] = ()

    @property
    def outgoing(self) -> int:
        return len(self.process.split(">")[1].split())


PROCESSES = {
    "gg_ggg": Process("g g > g g g", {"QCD": 3, "QED": 0}),
    "uux_ddxg": Process("u u~ > d d~ g", {"QCD": 3, "QED": 0}),
    "uux_emepddxg": Process("u u~ > e- e+ d d~ g", {"QCD": 3, "QED": 2}, (23,)),
    "gg_gggg": Process("g g > g g g g", {"QCD": 4, "QED": 0}),
}


def rotations():
    """The calibration's two rotation checks: generic axes and angles, so the
    rotated momenta are non-terminating rationals."""

    from pyamplicol.loops.v2.stability import Rotation

    return (
        Rotation.from_axis_angle((1, 1, 0), 0.9),
        Rotation.from_axis_angle((-1, 2, 2), 2.3),
    )


# ------------------------------------------------------------------ points


def _rational_direction(theta: float, phi: float, resolution: int):
    """A rational unit 3-vector near (theta, phi): the inverse stereographic
    projection of a rational point of the plane (from the south pole)."""

    z = math.cos(theta)
    scale = math.sin(theta) / (1 + z) if 1 + z > 1e-12 else 1e6
    u = Fraction(scale * math.cos(phi)).limit_denominator(resolution)
    v = Fraction(scale * math.sin(phi)).limit_denominator(resolution)
    n = 1 + u * u + v * v
    return (2 * u / n, 2 * v / n, (1 - u * u - v * v) / n)


def _rambo(n_out: int, energy: float, rng: random.Random):
    """RAMBO: n massless momenta in the centre-of-mass frame (binary64)."""

    qs = []
    for _ in range(n_out):
        c = 2 * rng.random() - 1
        phi = 2 * math.pi * rng.random()
        e = -math.log(rng.random() * rng.random())
        s = math.sqrt(1 - c * c)
        qs.append((e, e * s * math.cos(phi), e * s * math.sin(phi), e * c))
    total = [sum(q[k] for q in qs) for k in range(4)]
    mass = math.sqrt(total[0] ** 2 - sum(x * x for x in total[1:]))
    b = [-x / mass for x in total[1:]]
    gamma = total[0] / mass
    a = 1 / (1 + gamma)
    x = energy / mass
    out = []
    for q in qs:
        bq = sum(bi * qi for bi, qi in zip(b, q[1:], strict=True))
        space = [
            x * (qi + bi * (q[0] + a * bq)) for qi, bi in zip(q[1:], b, strict=True)
        ]
        out.append((x * (gamma * q[0] + bq), *space))
    return out


def exact_point(n_out: int, energy: float, rng: random.Random, resolution=10**6):
    """A 2 -> n_out massless point of exact rationals: every leg exactly
    massless, momentum exactly conserved, incoming momenta along +-z."""

    floats = _rambo(n_out, energy, rng)
    outgoing = []
    for e, px, py, pz in floats[:-1]:
        theta = math.acos(max(-1.0, min(1.0, pz / e)))
        direction = _rational_direction(theta, math.atan2(py, px), resolution)
        rounded = Fraction(e).limit_denominator(resolution)
        outgoing.append((rounded, *(rounded * c for c in direction)))
    px = -sum(p[1] for p in outgoing)
    py = -sum(p[2] for p in outgoing)
    transverse = px * px + py * py
    e, _, _, pz = floats[-1]
    # E - p_z = transverse / (E + p_z) keeps the last leg exactly massless.
    plus = max(Fraction(e + pz).limit_denominator(resolution), Fraction(1, resolution))
    outgoing.append(
        ((plus + transverse / plus) / 2, px, py, (plus - transverse / plus) / 2)
    )
    total_e = sum(p[0] for p in outgoing)
    total_z = sum(p[3] for p in outgoing)
    zero = Fraction(0)
    p1 = ((total_e + total_z) / 2, zero, zero, (total_e + total_z) / 2)
    p2 = ((total_e - total_z) / 2, zero, zero, -(total_e - total_z) / 2)
    return [p1, p2, *outgoing]


def _passes_cuts(point) -> bool:
    def rapidity_azimuth(p):
        _e, x, y, z = (float(c) for c in p)
        modulus = math.sqrt(x * x + y * y + z * z)
        return 0.5 * math.log((modulus + z) / (modulus - z)), math.atan2(y, x)

    outgoing = point[2:]
    if any(math.hypot(float(p[1]), float(p[2])) < PT_MIN for p in outgoing):
        return False
    for a, b in combinations(outgoing, 2):
        (ea, fa), (eb, fb) = rapidity_azimuth(a), rapidity_azimuth(b)
        dphi = abs(fa - fb)
        if math.hypot(ea - eb, min(dphi, 2 * math.pi - dphi)) < DELTA_R_MIN:
            return False
    return True


def sample_point(name: str, index: int) -> list[tuple[Fraction, ...]]:
    """Point ``index`` of the process's deterministic sample."""

    rng = random.Random(f"{name}:{index}")
    while True:
        point = exact_point(PROCESSES[name].outgoing, ENERGY, rng)
        if _passes_cuts(point):
            return point


# ------------------------------------------------------------------ slices


def compile_loop_sm(ufo: Path, cache: Path) -> Any:
    """loop_sm compiled once into ``cache`` with the checked scalar-arity
    view (as the tests); later calls load it."""

    from pyamplicol.models.loading import compile_model_source, load_compiled_model
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    path = cache / "loop-sm.json.pyAmplicol-model.json"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        model = compile_model_source(ufo, cache_dir=cache / "model-cache")
        model = replace(
            model,
            raw_loop_model=prepare_loop_sm_scalar_declarations(
                model.raw_loop_model
            ).derived,
            _serialized_path=None,
            _prepared_bundle=None,
        )
        os.replace(model.write(cache / f"loop-sm.{os.getpid()}.json"), path)
    return load_compiled_model(path)


def _furry_zeros(generate):
    """``generate_counterterm_sites`` with Furry's certified zeros: a closed
    quark loop with one photon and two gluons is C-odd, so its UV and R2
    parts vanish. The model certifier (WP3.5) will supply them; until then
    the photon processes need them here (as ``test_loop_v2_executor``)."""

    from pyamplicol.loops.v2 import ct_sites

    def with_zeros(families, table, rules, data, selection, **options):
        zeros = [
            ct_sites.CertifiedZero(
                signature,
                species,
                "Furry: C-odd closed quark loop (one photon, two gluons)",
            )
            for signature, sets in ct_sites.divergent_witnesses(families, data).items()
            if signature.fields == ("A", "G", "G")
            for species in sets
            if all(1 <= pdg <= 6 for pdg in species)
        ]
        return generate(families, table, rules, data, selection, zeros=zeros, **options)

    return with_zeros


def build_slice(name: str, compiled: Any) -> Any:
    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2 import assemble

    entry = PROCESSES[name]
    loop = {k: v + (2 if k == "QCD" else 0) for k, v in entry.born.items()}
    request = assemble.SliceRequest(
        entry.process,
        dict(entry.born),
        loop,
        fixed_parameters=FIXED,
        particle_veto=entry.ignored,
    )
    generate = assemble.generate_counterterm_sites
    if entry.born.get("QED"):
        assemble.generate_counterterm_sites = _furry_zeros(generate)
    try:
        return assemble.LoopSlice.generate(compiled, request, load_loop_native())
    finally:
        assemble.generate_counterterm_sites = generate


# ------------------------------------------------------------------ run


def _sector(sector) -> str:
    return " ".join(f"{name}^{power}" for name, power in sector) or "1"


def _values(components) -> dict[str, list[str]]:
    return {
        _sector(sector): [str(x) for x in laurent.coefficients]
        for sector, laurent in components.items()
    }


def _budgets(budgets) -> dict[str, list[float]]:
    return {
        _sector(sector): [float(x) for x in budget.coefficients]
        for sector, budget in budgets.items()
    }


def _record(result, seconds: float) -> dict[str, Any]:
    estimate = result.stability
    record = {
        "virtual": _values(result.components["virtual"]),
        "seconds": seconds,
    }
    if estimate is not None:
        record.update(
            budgets={name: _budgets(b) for name, b in estimate.budgets.items()},
            scale=_budgets(estimate.scale),
            roundoff=estimate.roundoff,
        )
    return record


def _evaluate(executor, points, precision, checks):
    """Evaluate a batch; a failed point is recorded with its reasons."""

    from pyamplicol.loops.v2.plan_export import FailedPoints

    scales = [SCALE_SQUARED] * len(points)
    start = time.perf_counter()
    failures = {}
    try:
        results = executor.evaluate(
            points, scales, parameters=PARAMETERS, precision=precision, stability=checks
        )
    except FailedPoints as failed:
        results, failures = failed.results, failed.failures
    seconds = (time.perf_counter() - start) / len(points)
    return [
        {"failure": "; ".join(failures[index])}
        if result is None
        else _record(result, seconds)
        for index, result in enumerate(results)
    ]


def run(options) -> int:
    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from pyamplicol.loops.v2.stability import StabilityChecks

    name = options.process
    compiled = compile_loop_sm(options.ufo, options.model_cache)
    executor = NativeExecutor(build_slice(name, compiled), threads=options.threads)
    checks = StabilityChecks(rotations(), True)
    # The rescaled reduction check is a second evaluation (one reduction
    # check per evaluation); ``report`` merges its budget by point.
    rescaled = StabilityChecks((), True, rescaled=True)
    precisions = [("double", Precision.double())]
    if options.double_double:
        precisions.append(("double-double", Precision.double_double()))
    carriers = [
        (f"{carrier}{RESCALED_SUFFIX}", precision, rescaled)
        for carrier, precision in precisions
    ]
    suffix = ".rescaled" if options.rescaled_only else ""
    if not options.rescaled_only:
        carriers = [(carrier, precision, checks) for carrier, precision in precisions]
        carriers += [(f"{c}{RESCALED_SUFFIX}", p, rescaled) for c, p in precisions]
        carriers.append(("truth", Precision.arbitrary(TRUTH_DIGITS), None))
    options.outdir.mkdir(parents=True, exist_ok=True)
    path = options.outdir / f"{name}.{options.start}-{options.stop}{suffix}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["index"] for line in path.read_text().splitlines()}
    todo = [i for i in range(options.start, options.stop) if i not in done]
    with path.open("a") as out:
        for first in range(0, len(todo), options.batch):
            indices = todo[first : first + options.batch]
            points = [
                [[str(x) for x in p] for p in sample_point(name, i)] for i in indices
            ]
            records = [{"process": name, "index": i} for i in indices]
            for carrier, precision, configured in carriers:
                for record, value in zip(
                    records,
                    _evaluate(executor, points, precision, configured),
                    strict=True,
                ):
                    record[carrier] = value
            for record, point in zip(records, points, strict=True):
                if not options.rescaled_only:
                    record["point"] = point
                out.write(json.dumps(record) + "\n")
            out.flush()
    return 0


# ------------------------------------------------------------------ report


@dataclass(frozen=True)
class _Laurent:
    coefficients: tuple[Any, ...]


def _configurations(names: Sequence[str]) -> list[tuple[str, ...]]:
    """Every subset of the checks one evaluation can run (one reduction
    check: the reversed or the rescaled frame)."""

    from pyamplicol.loops.v2.stability import FRAMES

    return [
        combination
        for size in range(1, len(names) + 1)
        for combination in combinations(sorted(names), size)
        if len(set(combination) & set(FRAMES)) <= 1
    ]


def _records(outdir: Path) -> list[dict[str, Any]]:
    """The points of a directory with the budget of their rescaled reduction
    check merged into each carrier's budgets (same binary64 evaluation: the
    virtuals must agree exactly)."""

    from pyamplicol.loops.v2.stability import RESCALED

    records: dict[tuple[str, int], dict[str, Any]] = {}
    extra: list[dict[str, Any]] = []
    for path in sorted(outdir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if "truth" in record:
                records[(record["process"], record["index"])] = record
            extra.append(record)
    for record in extra:
        target = records.get((record["process"], record["index"]))
        if target is None:
            continue
        for carrier in CARRIERS:
            source = record.get(f"{carrier}{RESCALED_SUFFIX}")
            data = target.get(carrier)
            if not source or "failure" in source or not data or "failure" in data:
                continue
            if source["virtual"] != data["virtual"]:
                raise ValueError(
                    f"{record['process']} {record['index']}: virtuals differ"
                )
            data["budgets"][RESCALED] = source["budgets"][RESCALED]
    return list(records.values())


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {
        "min": ordered[0],
        "q01": at(0.01),
        "q10": at(0.1),
        "median": at(0.5),
        "q90": at(0.9),
        "q99": at(0.99),
        "max": ordered[-1],
    }


THRESHOLDS = (1e-12, 1e-9, 1e-6, 1e-3)


def _point_report(record, carrier, truth):
    """Per configuration: (true relative error, estimate) for the Laurent
    vector and for the finite part, and the safety factor it needed.

    The estimate's absolute error E is measured against the truth's scale,
    as the true error is: a point whose value has no correct digit has a
    relative estimate of order 1 or more against its own (wrong) value,
    which does not compare with an error relative to the truth.
    """

    from pyamplicol.loops.v2 import stability

    data = record[carrier]
    reference = {
        sector: _Laurent(tuple(Decimal(x) for x in values))
        for sector, values in data["virtual"].items()
    }
    floats = {
        sector: _Laurent(tuple(float(x) for x in values.coefficients))
        for sector, values in reference.items()
    }
    width = len(next(iter(reference.values())).coefficients)
    exact = {
        sector: _Laurent(tuple(Decimal(x) for x in truth[sector][:width]))
        for sector in reference
    }
    errors = {
        sector: stability.Budget(
            tuple(
                float(abs(a - b))
                for a, b in zip(
                    reference[sector].coefficients,
                    exact[sector].coefficients,
                    strict=True,
                )
            )
        )
        for sector in reference
    }
    exact_floats = {
        sector: _Laurent(tuple(float(x) for x in value.coefficients))
        for sector, value in exact.items()
    }
    vector = stability.relative(exact_floats, errors)
    finite = stability.per_coefficient(exact_floats, errors)[0]
    budgets = {
        name: {s: stability.Budget(tuple(v)) for s, v in b.items()}
        for name, b in data["budgets"].items()
    }
    scale = {s: stability.Budget(tuple(v)) for s, v in data["scale"].items()}
    out = {}
    for names in _configurations(list(budgets)):
        estimate = stability.StabilityEstimate.from_checks(
            floats,
            {name: budgets[name] for name in names},
            scale=scale,
            roundoff=data["roundoff"],
        )
        out[names] = (
            vector,
            stability.relative(exact_floats, estimate.absolute),
            finite,
            stability.per_coefficient(exact_floats, estimate.absolute)[0],
        )
    raw = {name: stability.relative(exact_floats, b) for name, b in budgets.items()}
    return out, raw


def report(options) -> int:
    from pyamplicol.loops.v2 import stability

    rows: dict[tuple[str, str], list] = defaultdict(list)
    failures: dict[str, list] = defaultdict(list)
    seconds: dict[tuple[str, str], list] = defaultdict(list)
    for record in _records(options.outdir):
        name = record["process"]
        truth = record["truth"]
        if "failure" in truth:
            failures[name].append((record["index"], truth["failure"]))
            continue
        for carrier in CARRIERS:
            data = record.get(carrier)
            if data is None:
                continue
            if "failure" in data:
                failures[f"{name} {carrier}"].append((record["index"], data["failure"]))
                continue
            rows[(name, carrier)].append(
                _point_report(record, carrier, truth["virtual"])
            )
            seconds[(name, carrier)].append(data["seconds"])
    payload: dict[str, Any] = {
        "safety": stability.SAFETY,
        "floor_units": stability.FLOOR_UNITS,
        "truth_bits": 256,
        "processes": {},
    }
    for (name, carrier), points in sorted(rows.items()):
        configurations = {}
        names_seen = sorted({names for p in points for names in p[0]}, key=len)
        for names in names_seen:
            pairs = [p[0][names] for p in points if names in p[0]]
            vector_misses = sum(error > estimate for error, estimate, _, _ in pairs)
            finite_misses = sum(error > estimate for _, _, error, estimate in pairs)
            configurations[" + ".join(names)] = {
                "points": len(pairs),
                "false_negative_rate": vector_misses / len(pairs),
                "false_negative_rate_finite": finite_misses / len(pairs),
                "false_negatives": vector_misses,
                "false_negatives_finite": finite_misses,
                "estimate_over_error": _quantiles(
                    [e / r for r, e, _, _ in pairs if r > 0]
                ),
                "flagged": {
                    f"{t:.0e}": sum(e > t for _, e, _, _ in pairs) / len(pairs)
                    for t in THRESHOLDS
                },
                "true_above": {
                    f"{t:.0e}": sum(r > t for r, _, _, _ in pairs) / len(pairs)
                    for t in THRESHOLDS
                },
            }
        raw = {}
        for check in sorted({check for p in points for check in p[1]}):
            # (raw check budget, true error) of the points that ran it.
            pairs = [(p[1][check], p[0][(check,)][0]) for p in points if check in p[1]]
            raw[check] = {
                "points": len(pairs),
                "check_over_error": _quantiles([b / e for b, e in pairs if e > 0]),
                "false_negative_rate_unscaled": sum(b < e for b, e in pairs)
                / len(pairs),
            }
        errors = [p[0][next(iter(p[0]))][0] for p in points]
        payload["processes"].setdefault(name, {})[carrier] = {
            "points": len(points),
            "failed_truth": len(failures.get(name, [])),
            "failed": len(failures.get(f"{name} {carrier}", [])),
            "seconds_per_point": _quantiles(seconds[(name, carrier)]),
            "true_relative_error": _quantiles(errors),
            "configurations": configurations,
            "checks": raw,
        }
    payload["failures"] = {k: v[:20] for k, v in failures.items()}
    text = json.dumps(payload, indent=1) + "\n"
    if options.output:
        options.output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("process", choices=sorted(PROCESSES))
    runner.add_argument("start", type=int)
    runner.add_argument("stop", type=int)
    runner.add_argument("outdir", type=Path)
    runner.add_argument("--batch", type=int, default=8)
    runner.add_argument("--threads", type=int, default=1)
    runner.add_argument("--double-double", action="store_true")
    runner.add_argument(
        "--rescaled-only",
        action="store_true",
        help="only the rescaled reduction check (a supplement to a finished run)",
    )
    runner.add_argument("--ufo", type=Path, default=UFO)
    runner.add_argument(
        "--model-cache", type=Path, default=Path.cwd() / ".loop-sm-model"
    )
    reporter = commands.add_parser("report")
    reporter.add_argument("outdir", type=Path)
    reporter.add_argument("--output", type=Path)
    options = parser.parse_args(argv)
    return run(options) if options.command == "run" else report(options)


if __name__ == "__main__":
    sys.exit(main())
