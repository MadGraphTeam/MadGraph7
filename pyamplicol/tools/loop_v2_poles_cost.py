#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Cost of the pole selections of the v2 virtual (design §12.4).

Runs the complete native evaluation of exported plan v4 programs
(``tools/loop_v2_schedules.py export``; schedule B, binary64, no stability
check, random block rows, Born components and kernel constants at RAMBO
points, as ``loop_v2_schedules.py measure``) with

* the fast engine: finite only (``none``), combined poles (``all``) and the
  split columns of the ``uv`` and ``ir`` selections (``split``: the combined
  Laurent plus each lane's UV residue);
* NinjaRS's generic binary64 engine with oneloopmaster's master selection
  (``generic-both``, ``generic-uv``): the upstream route to separated poles,
  which retains the scaleless bubbles the fast engine drops.

The points are exactly light-like rational momenta (every on-shell invariant is
exactly zero, which the generic engine's masters need); block rows, Born
components and kernel constants are random. Per plan and configuration it
records the best of ``--reps`` runs (after one untimed run) of the native
seconds per point and of the reduction stage, and writes them as JSON. The
``uv`` and ``ir`` selections also dress the certified-scaleless families
(``LoopSlice.scaleless``, a plan of their own): ``export`` writes that plan,
``<stem>.scaleless.json``, for the processes of ``loop_v2_schedules.py``, and
``measure --scaleless DIR`` times it (``split``) next to each plan.

Usage:
    python tools/loop_v2_poles_cost.py export --output DIR [--process KEY ...]
    python tools/loop_v2_poles_cost.py measure PLAN.json [...] --json OUT.json
        [--scaleless DIR] [--points 16] [--lanes 16] [--reps 3] [--threads 1]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.loop_v2_operation_counts import UFO  # noqa: E402
from tools.loop_v2_schedules import (  # noqa: E402
    NATIVE_STAGES,
    PARAMETERS,
    PROCESSES,
    SCALE_SQUARED,
    _text,
    compile_loop_sm,
    generate,
)

# name: (engine, poles, master poles)
CONFIGURATIONS = {
    "none": ("fast", "none", "both"),
    "all": ("fast", "all", "both"),
    "split": ("fast", "split", "both"),
    "generic-both": ("generic", "all", "both"),
    "generic-uv": ("generic", "all", "uv"),
}


def _light_like_point(outgoing: int, rng: random.Random) -> list[list[Fraction]]:
    """A 2 -> ``outgoing`` point of exactly light-like rational momenta
    (rational unit directions by stereographic projection), so that every
    on-shell invariant is exactly zero in every carrier: the generic engine's
    masters take the invariants as given (no on-shell snapping)."""

    def direction() -> tuple[Fraction, Fraction, Fraction]:
        u = Fraction(rng.randint(-40, 40), rng.randint(5, 40))
        v = Fraction(rng.randint(-40, 40), rng.randint(5, 40))
        d = 1 + u * u + v * v
        return 2 * u / d, 2 * v / d, (1 - u * u - v * v) / d

    while True:
        final = []
        for _ in range(outgoing - 1):
            energy = Fraction(rng.randint(100, 400))
            x, y, z = direction()
            final.append([energy, energy * x, energy * y, energy * z])
        a = -sum(p[0] for p in final)
        c = -sum(p[3] for p in final)
        bx = -sum(p[1] for p in final)
        by = -sum(p[2] for p in final)
        # p_n = (E1 + E2 + a, bx, by, E1 - E2 + c) is light-like if
        # (2 E1 + a + c)(2 E2 + a - c) = bx^2 + by^2.
        t = Fraction(rng.randint(100, 400))
        e1 = -(a + c) / 2 + t
        e2 = ((bx * bx + by * by) / (2 * t) - a + c) / 2
        last = [e1 + e2 + a, bx, by, e1 - e2 + c]
        if e1 > 0 and e2 > 0 and last[0] > 0:
            return [[e1, 0, 0, e1], [e2, 0, 0, -e2], *final, last]


def _inputs(program: dict, masses: list[str], points: int, seed: int) -> dict:
    """``loop_v2_schedules._inputs`` at light-like rational points."""

    rng = random.Random(seed)

    def value() -> list[float]:
        return [rng.uniform(-1, 1), rng.uniform(-1, 1)]

    momenta = []
    for _ in range(points):
        point = _light_like_point(program["legs"] - 2, rng)
        incoming = [*point[:2], *[[-x for x in p] for p in point[2:]]]
        if any(sum(p[mu] for p in incoming) != 0 for mu in range(4)):
            raise AssertionError("a point that does not conserve momentum")
        if any(p[0] ** 2 != p[1] ** 2 + p[2] ** 2 + p[3] ** 2 for p in incoming):
            raise AssertionError("a momentum that is not light-like")
        momenta.append([_text(Fraction(x)) for p in incoming for x in p])
    block = sum(program["block_rows"])
    born = program["rows"] * program["born_components"]
    return {
        "momenta": momenta,
        "scales": [SCALE_SQUARED] * points,
        "masses": masses,
        "constants": [value() for _ in range(program["constants"])],
        "block_rows": [[value() for _ in range(block)] for _ in range(points)],
        "born": [[value() for _ in range(born)] for _ in range(points)],
    }


def export(options: argparse.Namespace) -> int:
    """The scaleless companion plan of each process (full colour)."""

    import tempfile

    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2.assemble import Precision, _mass_values
    from pyamplicol.loops.v2.plan_export import export_plan

    options.output.mkdir(parents=True, exist_ok=True)
    native = load_loop_native()
    with tempfile.TemporaryDirectory() as scratch:
        compiled = compile_loop_sm(options.ufo, Path(scratch))
        for key in options.process:
            start = time.perf_counter()
            slice_ = generate(compiled, native, key, "full")
            seconds = time.perf_counter() - start
            twin = slice_.scaleless
            stem = f"{key}-FC.scaleless"
            if twin is None:
                print(f"{key}: no certified-scaleless family", flush=True)
                continue
            exported = export_plan(twin, forward=False)
            (options.output / f"{stem}.json").write_text(exported.json())
            squared = _mass_values(
                twin.compiled,
                PARAMETERS,
                exported.masses,
                Precision.double().working_digits,
            )
            summary = {
                "process": PROCESSES[key][0],
                "accuracy": "full",
                "families": len(twin.plan.schedules),
                "main_families": len(slice_.plan.schedules),
                "masses": [
                    [None if mass.massless else mass.mass, str(squared[mass])]
                    for mass in exported.masses
                ],
                "seconds": {"slice": seconds},
            }
            (options.output / f"{stem}.summary.json").write_text(
                json.dumps(summary, indent=1) + "\n"
            )
            print(stem, json.dumps(summary), flush=True)
    return 0


def _run(native, path: Path, engine: str, poles, masters: str, options) -> dict:
    """Best of ``options.reps`` runs of one plan and configuration."""

    text = path.read_text()
    program = json.loads(text)
    summary = json.loads(path.with_name(f"{path.stem}.summary.json").read_text())
    masses = [squared for _name, squared in summary["masses"]]
    inputs = _inputs(program, masses, options.points, options.seed)
    executor = native.V2Executor(
        text,
        "double",
        None,
        lanes=options.lanes,
        threads=options.threads,
        engine=engine,
    )
    best = None
    for rep in range(options.reps + 1):
        start = time.perf_counter()
        output = executor.evaluate(
            inputs, poles=poles, check=False, master_poles=masters
        )
        wall = time.perf_counter() - start
        seconds = output["report"]["seconds"]
        total = sum(seconds.get(stage, 0.0) for stage in NATIVE_STAGES)
        if rep and (best is None or total < best[0]):
            best = total, wall, seconds, output
    total, wall, seconds, output = best
    return {
        "seconds_per_point": total / options.points,
        "reduction_per_point": seconds.get("reduction", 0.0) / options.points,
        "wall_seconds_per_point": wall / options.points,
        "failed_points": sum(1 for f in output["failures"] if f),
    }


def measure(options: argparse.Namespace) -> int:
    from pyamplicol.loops.native import load_loop_native

    native = load_loop_native()
    records = []
    for path in options.plans:
        if options.scaleless is not None:
            twin = options.scaleless / f"{path.stem}.scaleless.json"
            if twin.exists():
                record = _run(native, twin, "fast", "split", "both", options)
                print(
                    f"{twin.stem} split: {record['seconds_per_point'] * 1e3:.3f} ms/pt",
                    flush=True,
                )
                records.append({"plan": str(twin), "split": record})
        program = json.loads(path.read_text())
        summary = json.loads(path.with_name(f"{path.stem}.summary.json").read_text())
        record = {
            "plan": str(path),
            "process": summary["process"],
            "accuracy": summary["accuracy"],
            "families": len(program["families"]),
            "points": options.points,
            "lanes": options.lanes,
            "threads": options.threads,
        }
        del program
        for name in options.configurations:
            engine, poles, masters = CONFIGURATIONS[name]
            record[name] = _run(native, path, engine, poles, masters, options)
            print(
                f"{path.stem} {name}: {record[name]['seconds_per_point'] * 1e3:.2f} "
                f"ms/pt (reduction {record[name]['reduction_per_point'] * 1e3:.2f}), "
                f"failed points {record[name]['failed_points']}",
                flush=True,
            )
        records.append(record)
        if options.json:
            options.json.write_text(json.dumps(records, indent=1) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    exporting = commands.add_parser("export")
    exporting.add_argument("--output", type=Path, required=True)
    exporting.add_argument(
        "--process", nargs="+", choices=tuple(PROCESSES), default=["gg_ggg"]
    )
    exporting.add_argument("--ufo", type=Path, default=UFO)
    measuring = commands.add_parser("measure")
    measuring.add_argument("plans", nargs="+", type=Path)
    measuring.add_argument("--json", type=Path)
    measuring.add_argument("--scaleless", type=Path)
    measuring.add_argument("--points", type=int, default=16)
    measuring.add_argument("--lanes", type=int, default=16)
    measuring.add_argument("--reps", type=int, default=3)
    measuring.add_argument("--threads", type=int, default=1)
    measuring.add_argument("--seed", type=int, default=20261023)
    measuring.add_argument(
        "--configurations",
        nargs="+",
        choices=tuple(CONFIGURATIONS),
        default=list(CONFIGURATIONS),
    )
    options = parser.parse_args(argv)
    return export(options) if options.command == "export" else measure(options)


if __name__ == "__main__":
    raise SystemExit(main())
