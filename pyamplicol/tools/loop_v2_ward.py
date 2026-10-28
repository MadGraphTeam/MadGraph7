#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Ward identities of a benchmark process on the native per-point input path
(M-QCD MQ-WARD; REQUIREMENTS_ADDENDUM H: "Ward identities probed by
replacing one external gluon's polarisation vector by its momentum").

The process of a ``tools/loop_benchmark`` key (its request, parameters and
frozen point stream, as the G1 harness generates it; g g > g g g g g is
NJet's top-free five-flavour QCD) is generated on the full tree kernels
(``loops.v2.ward.ward_slice``) and evaluated natively with native inputs
(``NativeExecutor(..., inputs="native")``, every pole). Per point and
carrier one batch holds the physical evaluation and, for every chosen gluon
leg and factor c (1 and i), the evaluation with epsilon_j -> c p_j / p_j^0
on the loop side (``ward.gauge_ward``; ``ward=`` one value per batch
entry). Binary64 runs with the reduction-direction check, which its
criterion reads (``ward.WARD_CHECKS``); double-double runs without checks
unless ``--dd-checks``.

Each row is ``ward.ward_row`` (the criterion: the Born and the virtual of
every sector and Laurent coefficient vanish to the carrier's rounding
relative to the physical scale) plus the lanes' absolute sum over the scale
(the conditioning of the cancellation, ``ward.lane_scales``). The record is
written after every batch (``--json``); the exit status is 0 when every row
passed.

Usage:
    python tools/loop_v2_ward.py gg_ggggg --points 0 --legs 1 4 7 \\
        --precisions double double-double --threads 16 \\
        --scratch DIR --json OUT.json
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FACTORS = {"1": (1, 0), "i": (0, 1)}
CARRIERS = ("double", "double-double")


def _peak_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("process", help="a tools/loop_benchmark process key")
    parser.add_argument("--points", type=int, nargs="+", default=[0])
    parser.add_argument("--legs", type=int, nargs="+", help="default: every gluon")
    parser.add_argument(
        "--factors", nargs="+", choices=tuple(FACTORS), default=["1", "i"]
    )
    parser.add_argument("--precisions", nargs="+", choices=CARRIERS, default=CARRIERS)
    parser.add_argument("--dd-checks", action="store_true")
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--scratch", type=Path, help="parent of the model directory")
    parser.add_argument("--json", type=Path)
    options = parser.parse_args(argv)

    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2 import ward
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from tools.developer.prepare_source_runtime import _native_build_inputs_digest
    from tools.loop_benchmark import pyamplicol_v2, streams
    from tools.loop_v2_operation_counts import UFO, compile_loop_sm

    record: dict[str, Any] = {
        "kind": "ward-native",
        "revision": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        # The checkout's native build-input digest (the staged native's, which
        # the source runtime verifies at import).
        "native_digest": _native_build_inputs_digest(ROOT),
        "threads": options.threads,
        "lanes": options.lanes,
        "process": options.process,
        "criterion": {
            "born": ward.BORN_BOUNDS,
            "virtual": ward.VIRTUAL_BOUNDS,
            "estimate_factor": ward.ESTIMATE_FACTOR,
            "binary64_scope": ward.BINARY64_SCOPE,
            "binary64_floor": ward.BINARY64_FLOOR,
            "vector": "c p_j / p_j^0",
            "checks": {"double": "reduction-direction"},
        },
        "stages": {},
        "points": [],
    }
    checks = record["criterion"]["checks"]
    if options.dd_checks:
        checks["double-double"] = checks["double"]

    def write() -> None:
        if options.json:
            options.json.parent.mkdir(parents=True, exist_ok=True)
            options.json.write_text(json.dumps(record, indent=1) + "\n")

    native = load_loop_native()
    scratch = Path(tempfile.mkdtemp(prefix="ward-model-", dir=options.scratch))
    start = time.perf_counter()
    compiled = compile_loop_sm(UFO, scratch)
    request = pyamplicol_v2._request(options.process, "full", compiled)
    record["process_text"] = request.process
    record["particle_veto"] = list(request.particle_veto)
    record["stages"]["model"] = time.perf_counter() - start
    start = time.perf_counter()
    slice_ = ward.ward_slice(compiled, request, native)
    record["stages"]["generate"] = time.perf_counter() - start
    record["stages"]["generate_peak_gib"] = _peak_gib()
    print(options.process, "generated", record["stages"], flush=True)
    start = time.perf_counter()
    executor = NativeExecutor(
        slice_,
        inputs="native",
        forward=False,
        lanes=options.lanes,
        threads=options.threads,
    )
    record["stages"]["export"] = time.perf_counter() - start
    record["stages"]["export_peak_gib"] = _peak_gib()
    legs = tuple(options.legs or ward.gauge_legs(slice_))
    record["legs"] = list(legs)
    write()
    parameters = pyamplicol_v2._parameters(options.process)
    stream = streams.load(options.process)
    average = slice_.normalisation.average
    for index in options.points:
        point = streams.exact(stream, index)
        scale = str(Fraction(stream["mu_r"][index]) ** 2)
        entry: dict[str, Any] = {"index": index, "carriers": {}}
        record["points"].append(entry)
        names = {FACTORS[name]: name for name in options.factors}
        for carrier in options.precisions:
            checked = carrier == "double" or options.dd_checks
            begin = time.perf_counter()
            check = ward.ward_check(
                executor,
                point,
                scale,
                carrier,
                legs=legs,
                factors=list(names),
                parameters=parameters,
                checks=checked,
            )
            seconds = time.perf_counter() - begin
            physical = check["physical"]
            scales = ward.ward_scales(physical)
            rows = []
            for row, result in zip(check["rows"], check["results"], strict=True):
                lanes = ward.lane_scales(result, average)
                row.update(
                    factor=names[row["factor"]],
                    lanes=[v / s for v, s in zip(lanes, scales, strict=True)],
                )
                rows.append(row)
            width = len(scales)
            summary = {
                "seconds": seconds,
                "batch": len(rows) + 1,
                "checks": checked,
                "born": float(sum(complex(v).real for v in physical.born.values())),
                # The physical virtual per sector and Laurent coefficient
                # (against the production slice's values at the point).
                "virtual": {
                    repr(sector): [
                        [complex(x).real, complex(x).imag] for x in laurent.coefficients
                    ]
                    for sector, laurent in physical.components["virtual"].items()
                },
                "scale": scales,
                "physical_estimate": (
                    None
                    if physical.stability is None
                    else list(physical.stability.coefficients)
                ),
                "rows": rows,
                "passed": bool(check["passed"]),
                "max_born": max(row["born"] for row in rows),
                "max_virtual": [
                    max(row["parts"]["virtual"][k] for row in rows)
                    for k in range(width)
                ],
            }
            entry["carriers"][carrier] = summary
            print(
                options.process,
                index,
                carrier,
                f"{seconds:.1f}s",
                "passed" if summary["passed"] else "FAILED",
                "max |V|/scale",
                [f"{x:.1e}" for x in summary["max_virtual"]],
                "max born ratio",
                f"{summary['max_born']:.1e}",
                flush=True,
            )
            write()
    record["max_rss_gib"] = _peak_gib()
    record["passed"] = all(
        carrier["passed"]
        for entry in record["points"]
        for carrier in entry["carriers"].values()
    )
    write()
    status = "passed" if record["passed"] else "FAILED"
    print(options.process, "DONE", status, flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
