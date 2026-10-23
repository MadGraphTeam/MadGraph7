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

Per plan and configuration it records the best of ``--reps`` runs (after one
untimed run) of the native seconds per point and of the reduction stage, and
writes them as JSON.

Usage:
    python tools/loop_v2_poles_cost.py PLAN.json [...] --json OUT.json
        [--points 16] [--lanes 16] [--reps 3] [--threads 1]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.loop_v2_schedules import NATIVE_STAGES, _inputs  # noqa: E402

# name: (engine, poles, master poles)
CONFIGURATIONS = {
    "none": ("fast", False, "both"),
    "all": ("fast", True, "both"),
    "split": ("fast", "split", "both"),
    "generic-both": ("generic", True, "both"),
    "generic-uv": ("generic", True, "uv"),
}


def measure(options: argparse.Namespace) -> int:
    from pyamplicol.loops.native import load_loop_native

    native = load_loop_native()
    records = []
    for path in options.plans:
        text = path.read_text()
        program = json.loads(text)
        summary = json.loads(path.with_name(f"{path.stem}.summary.json").read_text())
        masses = [squared for _name, squared in summary["masses"]]
        inputs = _inputs(program, masses, options.points, options.seed)
        record = {
            "plan": str(path),
            "process": summary["process"],
            "accuracy": summary["accuracy"],
            "families": len(program["families"]),
            "points": options.points,
            "lanes": options.lanes,
            "threads": options.threads,
        }
        executors = {
            engine: native.V2Executor(
                text,
                "double",
                None,
                lanes=options.lanes,
                threads=options.threads,
                engine=engine,
            )
            for engine in ("fast", "generic")
        }
        for name in options.configurations:
            engine, poles, masters = CONFIGURATIONS[name]
            best = None
            for rep in range(options.reps + 1):
                start = time.perf_counter()
                output = executors[engine].evaluate(
                    inputs, poles=poles, check=False, master_poles=masters
                )
                wall = time.perf_counter() - start
                seconds = output["report"]["seconds"]
                total = sum(seconds.get(stage, 0.0) for stage in NATIVE_STAGES)
                if rep and (best is None or total < best[0]):
                    best = total, wall, seconds, output
            total, wall, seconds, output = best
            record[name] = {
                "seconds_per_point": total / options.points,
                "reduction_per_point": seconds.get("reduction", 0.0) / options.points,
                "wall_seconds_per_point": wall / options.points,
                "failed_points": sum(1 for f in output["failures"] if f),
            }
            print(
                f"{path.stem} {name}: {total / options.points * 1e3:.2f} ms/pt "
                f"(reduction {record[name]['reduction_per_point'] * 1e3:.2f}), "
                f"failed points {record[name]['failed_points']}",
                flush=True,
            )
        records.append(record)
        if options.json:
            options.json.write_text(json.dumps(records, indent=1) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("plans", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--points", type=int, default=16)
    parser.add_argument("--lanes", type=int, default=16)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=tuple(CONFIGURATIONS),
        default=list(CONFIGURATIONS),
    )
    return measure(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
