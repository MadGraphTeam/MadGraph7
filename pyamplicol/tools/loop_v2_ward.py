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
leg and factor c (1 and i), the evaluation with epsilon_j -> c p_j on the
loop side (``ward=`` one value per batch entry). A row keeps, per Laurent
coefficient, the largest magnitude over the sectors of every component
divided by the physical scale (the largest physical component,
``ward.ward_scales``), the lanes' absolute sum over that scale (the
conditioning of the cancellation, ``ward.lane_scales``) and the loop-side
Born over the physical Born (``ward.born_ratio``).

The bound of each row is O10's (``tests/unit/test_loop_v2_observables.py``,
``assert_ward_identity``), per Laurent coefficient for the virtual, the
local UV part and the external fields: in binary64 1e-11 of the scale (the
physical one, or the R2 part where it is larger: it cancels a bare
violation of its own size) plus 1e-12 of the lanes' absolute sum; in
double-double 1e-25 of the scale. The Born ratio is reported. The record is
written after every batch (``--json``).

Usage:
    python tools/loop_v2_ward.py gg_ggggg --points 0 --legs 1 4 7 \\
        --precisions double double-double --threads 16 \\
        --scratch DIR --json OUT.json
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# O10's bounds (``assert_ward_identity``): binary64 relative to the scale and
# to the lanes' absolute sum; double-double relative to the scale.
BOUNDS = {"double": (1e-11, 1e-12), "double-double": (1e-25, 0.0)}
PARTS = ("virtual", "bare", "r2", "uv_local", "uv_external_fields")
FACTORS = {"1": (1, 0), "i": (0, 1)}


def _peak_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _precision(name: str) -> Any:
    from pyamplicol.loops.v2.assemble import Precision

    if name == "double":
        return Precision.double()
    if name == "double-double":
        return Precision.double_double()
    raise SystemExit(f"unknown precision {name!r}")


def rows_of(
    physical: Any,
    results: list[tuple[int, str, Any]],
    average: int,
    precision: str,
) -> list[dict[str, Any]]:
    """The Ward rows of one point (``results``: (leg, factor, result))
    against its physical evaluation, with O10's bound per row."""

    from pyamplicol.loops.v2 import ward

    scale = ward.ward_scales(physical)
    relative, lanes_bound = BOUNDS[precision]
    rows = []
    for leg, factor, result in results:
        parts = {name: ward.largest(result, name) for name in PARTS}
        lanes = ward.lane_scales(result, average)
        size = [max(s, r2) for s, r2 in zip(scale, parts["r2"], strict=True)]
        bound = [
            relative * a + lanes_bound * b for a, b in zip(size, lanes, strict=True)
        ]
        worst = max(
            parts[name][k] / bound[k]
            for name in ward.WARD_PARTS
            for k in range(len(scale))
        )
        rows.append(
            {
                "leg": leg,
                "factor": factor,
                "born": ward.born_ratio(result, physical),
                "parts": {
                    name: [v / s for v, s in zip(values, scale, strict=True)]
                    for name, values in parts.items()
                },
                "lanes": [v / s for v, s in zip(lanes, scale, strict=True)],
                "bound": [v / s for v, s in zip(bound, scale, strict=True)],
                "worst_over_bound": worst,
                "passed": worst <= 1,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("process", help="a tools/loop_benchmark process key")
    parser.add_argument("--points", type=int, nargs="+", default=[0])
    parser.add_argument("--legs", type=int, nargs="+", help="default: every gluon")
    parser.add_argument(
        "--factors", nargs="+", choices=tuple(FACTORS), default=["1", "i"]
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=tuple(BOUNDS),
        default=["double", "double-double"],
    )
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--scratch", type=Path, help="parent of the model directory")
    parser.add_argument("--json", type=Path)
    options = parser.parse_args(argv)

    import subprocess

    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2 import ward
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from tools.developer.prepare_source_runtime import _native_build_inputs_digest
    from tools.loop_benchmark import pyamplicol_v2, streams
    from tools.loop_v2_operation_counts import UFO, compile_loop_sm

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    record: dict[str, Any] = {
        "kind": "ward-native",
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        # The checkout's native build-input digest (the staged native's, which
        # the source runtime verifies at import).
        "native_digest": _native_build_inputs_digest(ROOT),
        "threads": options.threads,
        "lanes": options.lanes,
        "process": options.process,
        "bounds": {k: list(v) for k, v in BOUNDS.items()},
        "stages": {},
        "points": [],
    }

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
        labels = [(leg, name) for leg in legs for name in options.factors]
        configurations = [None] + [(leg, FACTORS[name]) for leg, name in labels]
        entry: dict[str, Any] = {"index": index, "carriers": {}}
        record["points"].append(entry)
        for name in options.precisions:
            begin = time.perf_counter()
            results = executor.evaluate(
                [point] * len(configurations),
                [scale] * len(configurations),
                parameters=parameters,
                precision=_precision(name),
                ward=list(configurations),
            )
            seconds = time.perf_counter() - begin
            physical = results[0]
            labelled = [
                (leg, factor, result)
                for (leg, factor), result in zip(labels, results[1:], strict=True)
            ]
            rows = rows_of(physical, labelled, average, name)
            entry["carriers"][name] = {
                "seconds": seconds,
                "batch": len(configurations),
                "born": float(sum(complex(v).real for v in physical.born.values())),
                "scale": ward.ward_scales(physical),
                "rows": rows,
                "passed": all(row["passed"] for row in rows),
                "max_born": max(row["born"] for row in rows),
            }
            print(
                options.process,
                index,
                name,
                f"{seconds:.1f}s",
                "passed" if entry["carriers"][name]["passed"] else "FAILED",
                "max worst/bound",
                f"{max(row['worst_over_bound'] for row in rows):.2e}",
                "max born ratio",
                f"{entry['carriers'][name]['max_born']:.2e}",
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
    print(
        options.process, "DONE", "passed" if record["passed"] else "FAILED", flush=True
    )
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
